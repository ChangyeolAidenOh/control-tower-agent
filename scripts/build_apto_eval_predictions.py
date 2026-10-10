"""Stage 6 (ST6-11 (5), ST6-12): A-PTO evaluation-replay predictions.

Same contract as build_eval_predictions, CatBoost path: origins are the
decision days [test_start, test_end]; demand features are conditioned on
the fold's deterministic scenario path through a preloaded DemandHistory
(ST3 D5 -- demand flows only through observed_through, price/calendar/
static come from mart and are scenario-independent). The input-layer
scenario gate is identical to build_eval_predictions.

A-PTO has no committed scoring parquet, so the prediction-layer crosscheck
is replaced by the ST6-11 (5) reproducibility gate: the full feature +
predict pass runs TWICE per fold inside one invocation and must match
bitwise before the parquet is written.

Negative raw predictions are clipped to zero AT STORE TIME: max(0, dhat)
is part of the ST6-11 (3) policy formula, and the point schema requires
nonnegative yhat; the clip share is recorded per fold.

Lineage gates: fold model .cbm sha256 must match fold_models_manifest.json
(frozen GPU artifacts, ST6-12 (4)); wide feature columns must match the
tuning staging manifest; scales are the retrain fold's scales_f{K}.json.

Run:
  python -m scripts.build_apto_eval_predictions --folds 1-12
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from core_pipeline.data import db
from core_pipeline.data.mart_reader import fetch_demand_by_bundle, fetch_folds
from core_pipeline.forecast import apto, point_store
from core_pipeline.forecast.common import file_sha256, write_json
from core_pipeline.forecast.dataset import (
    DatasetConfig, DemandHistory, FoldDatasetBuilder, MartPanelSource,
    retrain_fold_for,
)
from core_pipeline.simulator.demand_paths import build_fold_paths, draw_shock
from core_pipeline.simulator.engine import load_sim_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "outputs" / "predictions_eval"
APTO_DIR = PROJECT_ROOT / "outputs" / "apto"
STAGING_DIR = PROJECT_ROOT / "outputs" / "apto_staging"
MODEL = "apto"
KIND = "pto_point"
SCENARIO = "core"
SCENARIO_P = 0.0
N_H = 28

def full_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def scenario_for_fold(demand: dict, fold_row: dict, sim_cfg):
    """Deterministic scenario paths + reconstructed shock draw + gates
    (identical to build_eval_predictions)."""
    fold = fold_row["fold"]
    ts, te = fold_row["test_start_idx"], fold_row["test_end_idx"]
    paths = build_fold_paths(
        demand, fold=fold,
        warmup_start_row=fold_row["warmup_start_idx"] - 1,
        test_start_row=ts - 1, test_end_row=te - 1,
        p=SCENARIO_P, cfg=sim_cfg,
    )
    group_ids = sorted(demand)
    draw = draw_shock(group_ids, te - ts + 1, sim_cfg, fold)
    sl = slice(draw.start_offset, draw.start_offset + draw.duration_days)
    sha = hashlib.sha256()
    for gid in group_ids:
        bp = paths[gid]
        _, mat = demand[gid]
        orig = mat[ts - 1:te, :]
        if bp.shocked != (gid == draw.group_id):
            raise RuntimeError(f"fold {fold}: shock flag mismatch at {gid}")
        if bp.shocked:
            mask = np.ones(orig.shape[0], dtype=bool)
            mask[sl] = False
            if not np.array_equal(bp.test[mask], orig[mask]):
                raise RuntimeError(
                    f"fold {fold} {gid}: scenario differs outside shock days")
            if not np.array_equal(bp.test[sl], orig[sl] * draw.multiplier):
                raise RuntimeError(
                    f"fold {fold} {gid}: shock days != mart * multiplier")
        else:
            if not np.array_equal(bp.test, orig):
                raise RuntimeError(
                    f"fold {fold} {gid}: non-shocked bundle altered")
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(bp.test, dtype=np.float64).tobytes())
    return paths, draw, sha.hexdigest()


def scenario_history(dp: pd.DataFrame, demand: dict, paths: dict,
                     fold_row: dict, static: pd.DataFrame) -> DemandHistory:
    """DemandHistory over period 1..test_end whose test window carries the
    scenario path; everything earlier is raw mart. Series order = sorted
    series_id (matches FoldDatasetBuilder's static order)."""
    ts, te = fold_row["test_start_idx"], fold_row["test_end_idx"]
    days = np.arange(ts, te + 1, dtype=np.int64)
    reps = []
    for gid in sorted(demand):
        series, _ = demand[gid]
        bp = paths[gid]
        reps.append(pd.DataFrame({
            "series_id": np.repeat(np.asarray(series, dtype=object),
                                   days.size),
            "period_idx": np.tile(days, len(series)),
            "sales_scn": bp.test.T.reshape(-1),
        }))
    rep = pd.concat(reps, ignore_index=True)
    out = dp.merge(rep, on=["series_id", "period_idx"], how="left")
    mask = out["sales_scn"].notna()
    if int(mask.sum()) != len(rep):
        raise RuntimeError(
            f"fold {fold_row['fold']}: scenario cells missing from panel")
    out.loc[mask, "sales"] = out.loc[mask, "sales_scn"].astype(
        out["sales"].dtype)
    wide = out.pivot(index="series_id", columns="period_idx", values="sales")
    wide = wide.sort_index().sort_index(axis=1)
    idx_full = np.arange(int(wide.columns.min()), te + 1)
    wide = wide.reindex(columns=idx_full)
    series_sorted = tuple(wide.index.astype(str))
    if series_sorted != tuple(static.sort_values("series_id")["series_id"]):
        raise RuntimeError("series order mismatch vs static table")
    mat = wide.to_numpy(dtype=np.float32)
    if np.isnan(mat).any():
        raise RuntimeError("NaN sales cells inside 1..test_end")
    return DemandHistory(series_sorted, int(idx_full[0]), mat,
                         current_idx=te)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for col in a.columns:
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            return False
    return True


def predict_fold(builder, cfg, hist, origins, model, scales,
                 feat_cols, cat_cols) -> tuple[np.ndarray, list, list, list]:
    """One full feature+predict pass. Returns (yhat_raw stacked, sid, org,
    hor long arrays) in deterministic order."""
    sids, orgs, hors, mats = [], [], [], []
    for origin in origins:
        frame = builder.inference_frame(hist, origin)
        wide = apto.to_wide(frame, cfg)
        X = wide.X
        if list(X.columns) != feat_cols:
            raise RuntimeError("wide feature columns != tuning staging")
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(int)
        yhat_scaled = np.asarray(model.predict(X), dtype=np.float64)
        if yhat_scaled.shape != (len(X), N_H):
            raise RuntimeError(f"predict shape {yhat_scaled.shape}")
        s = wide.meta["series_id"].map(scales).to_numpy(np.float64)
        if not np.isfinite(s).all() or (s <= 0).any():
            raise RuntimeError("invalid series scale at predict time")
        yhat_raw = yhat_scaled * s[:, None]
        n = len(X)
        sids.append(np.repeat(wide.meta["series_id"].to_numpy(), N_H))
        orgs.append(np.full(n * N_H, origin, dtype=np.int64))
        hors.append(np.tile(np.arange(1, N_H + 1, dtype=np.int64), n))
        mats.append(yhat_raw.reshape(-1))
    return (np.concatenate(mats), np.concatenate(sids),
            np.concatenate(orgs), np.concatenate(hors))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1-12")
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    tune_man = json.loads(
        (STAGING_DIR / "staging_manifest.json").read_text())
    fold_models = json.loads(
        (APTO_DIR / "fold_models_manifest.json").read_text())
    feat_cols = list(tune_man["feature_columns"])
    cat_cols = list(tune_man["categorical"])

    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    cfg = DatasetConfig(include_series_id=False)
    builder = FoldDatasetBuilder(cfg, source)
    static = source.load_static()
    dp = source.load_demand_price(max_period_idx=source.max_demand_idx())
    folds_all = {f["fold"]: f for f in fetch_folds(conn)}
    max_te = max(folds_all[f]["test_end_idx"] for f in fold_list)
    demand = fetch_demand_by_bundle(conn, max_te)
    print(f"staged rows={len(dp)} folds={fold_list}")

    models_cache: dict[int, tuple] = {}

    def fold_model(rf: int):
        if rf not in models_cache:
            mp = APTO_DIR / f"apto_f{rf}.cbm"
            sha = full_sha256(mp)
            want = fold_models["folds"][str(rf)]["model_sha256"]
            if sha != want:
                raise RuntimeError(
                    f"{mp.name}: sha {sha[:12]} != frozen {want[:12]} "
                    "(ST6-12 (4) artifact freeze)")
            m = CatBoostRegressor()
            m.load_model(str(mp))
            scales = json.loads(
                (STAGING_DIR / f"scales_f{rf}.json").read_text())
            models_cache[rf] = (m, scales, sha)
        return models_cache[rf]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for fold in fold_list:
        fr = folds_all[fold]
        paths, draw, path_sha = scenario_for_fold(demand, fr, sim_cfg)
        print(f"fold {fold:02d} scenario ok shock={draw.group_id} "
              f"start_offset={draw.start_offset}")
        hist = scenario_history(dp, demand, paths, fr, static)
        origins = list(range(fr["test_start_idx"], fr["test_end_idx"] + 1))
        rf = retrain_fold_for(fold)
        model, scales, model_sha = fold_model(rf)

        t0 = time.time()
        y1, sid, org, hor = predict_fold(
            builder, cfg, hist, origins, model, scales, feat_cols, cat_cols)
        y2, sid2, org2, hor2 = predict_fold(
            builder, cfg, hist, origins, model, scales, feat_cols, cat_cols)
        if not (np.array_equal(y1, y2) and np.array_equal(sid, sid2)
                and np.array_equal(org, org2) and np.array_equal(hor, hor2)):
            raise RuntimeError(
                f"fold {fold}: repeat pass NOT bitwise equal "
                "(ST6-11 (5) reproducibility gate)")

        n_neg = int((y1 < 0).sum())
        y_store = np.maximum(y1, 0.0)
        frame = point_store.build_point_frame(sid, org, hor, y_store)
        stats = point_store.validate_point_frame(frame, KIND)
        out_path = EVAL_DIR / f"{MODEL}_{SCENARIO}_f{fold}.parquet"
        frame.to_parquet(out_path, index=False)
        back = pd.read_parquet(out_path)
        if not frames_equal(frame.reset_index(drop=True), back):
            raise RuntimeError(f"reload mismatch for {out_path}")
        entries.append({
            "fold": fold,
            "file": out_path.name,
            "prediction_file_sha256": file_sha256(out_path),
            **stats,
            "origin_start": origins[0],
            "origin_end": origins[-1],
            "origin_count": len(origins),
            "retrain_fold": rf,
            "model_file": f"apto_f{rf}.cbm",
            "model_sha256": model_sha,
            "scales_file": f"scales_f{rf}.json",
            "neg_raw_clipped": n_neg,
            "neg_raw_share": round(n_neg / len(y1), 6),
            "repeat_pass_bitwise_equal": True,
            "demand_path_sha256": path_sha,
            "shock": {
                "group_id": draw.group_id,
                "start_offset": draw.start_offset,
                "shock_start_idx": fr["test_start_idx"] + draw.start_offset,
                "duration_days": draw.duration_days,
                "multiplier": draw.multiplier,
            },
        })
        print(f"fold {fold:02d} apto_f{rf} origins={len(origins)} "
              f"neg_clipped={n_neg} fold_s={time.time() - t0:.0f}")

    conn.close()
    write_json(EVAL_DIR / f"{MODEL}_{SCENARIO}_manifest.json", {
        "model": MODEL,
        "model_id": "apto",
        "prediction_kind": KIND,
        "schema_version": point_store.POINT_SCHEMA_VERSION,
        "prediction_purpose": "rolling_decision_replay",
        "scenario_id": {"name": SCENARIO, "p": SCENARIO_P,
                        "shock_rule": "pre-registered fold-level draw (seed D)"},
        "history_observation_rule": "exogenous_latent",
        "forecast_horizon": N_H,
        "framework": "catboost_multirmse_h28 (ST6-11/12)",
        "negatives": "raw yhat clipped to 0 at store time (ST6-11 (3) "
                     "max(0, dhat) policy formula)",
        "crosscheck": "no committed counterpart; reproducibility gate = "
                      "double in-process pass bitwise equal (ST6-11 (5))",
        "spec_hash": tune_man["spec_hash"],
        "encoder_hash": tune_man["encoder_hash"],
        "tuning_staging_sha": tune_man["files_sha256"],
        "folds": entries,
    })
    print(f"manifest written for {MODEL}_{SCENARIO}")


if __name__ == "__main__":
    main()