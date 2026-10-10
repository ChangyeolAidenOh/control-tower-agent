"""Stage 6 (ST6-02/ST6-03): evaluation-replay predictions for rolling decisions.

Committed Stage 3/4 parquets (origins = train_end..train_end+27, conditioned on
mart history) are scoring artifacts and stay untouched. Rolling evaluation
consumes a separate replay parquet per (model, scenario, fold): origins are the
decision days [test_start, test_end], and model inputs are conditioned on the
same deterministic scenario demand path the evaluation engine replays
(build_fold_paths, frozen seeds), so every policy observes one exogenous
latent-demand history (spec sections 5-6, ST6-03 (1)).

Offline model forward is allowed here only; the evaluation runtime does a
strict replay of these files (ST6-01 (2), ST6-02 (5)).

Hard gates before any parquet is written (ST6-02 (3), ST6-03 (2)):
  input layer      per bundle, scenario test matrix == mart matrix bitwise
                   outside the drawn shock, and == mart * multiplier inside it;
  prediction layer on origins shared with the committed parquet
                   (test_start..train_end+27): non-shocked bundles and
                   pre-shock origins of the shocked bundle must match exactly,
                   with a pre-fixed rel/abs 1e-6 fallback that is recorded;
                   post-shock differences are reported, never required.

Run:
  python -m scripts.build_eval_predictions --folds 1-12 --models b_r3,b_r5
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.data.mart_reader import fetch_demand_by_bundle, fetch_folds
from core_pipeline.forecast import point_store
from core_pipeline.forecast.common import file_sha256, write_json
from core_pipeline.forecast.dataset import MartPanelSource
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND, TSMixerConfig, checkpoint_fold_for, checkpoint_key,
)
from core_pipeline.forecast.tsmixer.staging import _canonical_hash
from core_pipeline.simulator.demand_paths import build_fold_paths, draw_shock
from core_pipeline.simulator.engine import load_sim_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DIR = PROJECT_ROOT / "outputs" / "predictions"
EVAL_DIR = PROJECT_ROOT / "outputs" / "predictions_eval"
SCENARIO = "core"
SCENARIO_P = 0.0
TOL = 1e-6


def parse_models(spec: str) -> list[tuple[str, int]]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if "_r" not in token:
            raise ValueError(f"model token {token!r} must carry _r<r>")
        mid, r = token.split("_r")
        checkpoint_key(mid, int(r))
        out.append((mid, int(r)))
    return out


def load_encoder(staging_root: Path, retrain_fold: int) -> dict:
    path = staging_root / f"fold_{retrain_fold:02d}" / "encoder.json"
    enc = json.loads(path.read_text())
    recomputed = _canonical_hash(
        {k: enc[k] for k in ("event_type", "static", "cardinalities")})
    if recomputed != enc["hash"]:
        raise ValueError(f"encoder.json content/hash mismatch at {path}")
    return enc


def stage_from_db(max_test_end: int):
    """Phase A: everything that touches PostgreSQL; then close."""
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    dp = source.load_demand_price(max_period_idx=source.max_demand_idx())
    cal = source.load_calendar(1, source.max_calendar_idx())
    static = source.load_static()
    max_cal = source.max_calendar_idx()
    folds = fetch_folds(conn)
    demand = fetch_demand_by_bundle(conn, max_test_end)
    conn.close()
    return dp, cal, static, max_cal, folds, demand


def scenario_for_fold(demand: dict, fold_row: dict, sim_cfg):
    """Deterministic scenario paths + reconstructed shock draw + gates."""
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
            if not np.array_equal(bp.test[sl],
                                  orig[sl] * draw.multiplier):
                raise RuntimeError(
                    f"fold {fold} {gid}: shock days != mart * multiplier")
        else:
            if not np.array_equal(bp.test, orig):
                raise RuntimeError(
                    f"fold {fold} {gid}: non-shocked bundle altered")
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(bp.test, dtype=np.float64).tobytes())
    return paths, draw, sha.hexdigest()


def fold_dp(dp: pd.DataFrame, demand: dict, paths: dict,
            fold_row: dict) -> pd.DataFrame:
    """Replace test-window sales in the long panel with the scenario path."""
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
            f"fold {fold_row['fold']}: {len(rep) - int(mask.sum())} "
            "scenario cells missing from the staged panel")
    out.loc[mask, "sales"] = out.loc[mask, "sales_scn"].astype(
        out["sales"].dtype)
    return out.drop(columns=["sales_scn"])


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for col in a.columns:
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            return False
    return True


def crosscheck(frame: pd.DataFrame, key: str, fold: int,
               series_group: dict, shocked_gid: str,
               shock_start_idx: int) -> dict:
    committed_path = COMMITTED_DIR / f"tsmixer_{key}_f{fold}.parquet"
    committed = pd.read_parquet(committed_path)
    merged = frame.merge(
        committed, on=["series_id", "origin_idx", "horizon"],
        suffixes=("_new", "_old"))
    if merged.empty:
        raise RuntimeError(f"{key} fold {fold}: no common origins to check")
    gids = merged["series_id"].map(series_group)
    if gids.isna().any():
        raise RuntimeError(f"{key} fold {fold}: unmapped series in crosscheck")
    strict = (gids != shocked_gid) | (merged["origin_idx"] < shock_start_idx)
    new = merged["yhat_new"].to_numpy(np.float64)
    old = merged["yhat_old"].to_numpy(np.float64)
    diff = np.abs(new - old)
    s_diff = diff[strict.to_numpy()]
    strict_max = float(s_diff.max()) if s_diff.size else 0.0
    strict_exact = bool((s_diff == 0.0).all())
    if not strict_exact:
        ok = np.isclose(new[strict.to_numpy()], old[strict.to_numpy()],
                        rtol=TOL, atol=TOL).all()
        if not ok:
            raise RuntimeError(
                f"{key} fold {fold}: strict crosscheck failed, "
                f"max_abs_diff {strict_max:.3e} beyond rel/abs {TOL}")
    post = (~strict).to_numpy()
    p_diff = diff[post]
    return {
        "committed_file": committed_path.name,
        "common_origins": int(merged["origin_idx"].nunique()),
        "strict_rows": int(strict.sum()),
        "strict_exact": strict_exact,
        "strict_max_abs_diff": strict_max,
        "post_shock_rows": int(post.sum()),
        "post_shock_max_abs_diff": float(p_diff.max()) if p_diff.size else 0.0,
        "post_shock_mean_abs_diff": (float(p_diff.mean())
                                     if p_diff.size else 0.0),
        "post_shock_changed_share": (float((p_diff > 0).mean())
                                     if p_diff.size else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1-12")
    ap.add_argument("--models", type=str, default="b_r3,b_r5")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--staging-root", default="outputs/staging/tsmixer")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))
    models = parse_models(args.models)

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise RuntimeError("kappa_selected missing in configs/simulator.yaml")

    t0 = time.time()
    all_folds_meta = None
    dp = cal = static = None
    # Phase A: all DB work, connection closed before any torch import.
    conn_probe = None
    del conn_probe
    # max test_end is needed before staging; read folds first via a light pass
    db.load_dotenv()
    conn = db.connect(db.load_config())
    all_folds_meta = fetch_folds(conn)
    conn.close()
    fold_rows = {f["fold"]: f for f in all_folds_meta}
    max_test_end = max(fold_rows[f]["test_end_idx"] for f in fold_list)
    dp, cal, static, max_cal, _, demand = stage_from_db(max_test_end)
    series_group = dict(zip(static["series_id"].astype(str),
                            static["group_id"].astype(str)))
    print(f"staged rows={len(dp)} folds={fold_list} "
          f"stage_s={time.time() - t0:.0f}")

    scenarios = {}
    for f in fold_list:
        scenarios[f] = scenario_for_fold(demand, fold_rows[f], sim_cfg)
        draw = scenarios[f][1]
        print(f"fold {f:02d} scenario ok shock={draw.group_id} "
              f"start_offset={draw.start_offset}")

    # Phase B: torch only from here on.
    from core_pipeline.forecast.tsmixer.checkpoint import load_checkpoint
    from core_pipeline.forecast.tsmixer.infer import (
        inference_pairs, long_arrays, predict,
    )
    from core_pipeline.forecast.tsmixer.panel import (
        build_panel, build_static_inputs,
    )

    cfg = TSMixerConfig()
    ckpt_dir = Path(args.ckpt_dir)
    staging_root = Path(args.staging_root)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    entries = {checkpoint_key(m, r): [] for m, r in models}
    runtime = {"device": "cpu", "batch_size": args.batch_size}

    by_retrain: dict[int, list[int]] = {}
    for fold in fold_list:
        by_retrain.setdefault(checkpoint_fold_for(fold), []).append(fold)

    for retrain_fold, folds in sorted(by_retrain.items()):
        encoder = load_encoder(staging_root, retrain_fold)
        loaded = {}
        ref_scale = None
        for mid, r in models:
            model, aux, manifest = load_checkpoint(
                ckpt_dir, mid, retrain_fold, r,
                expected_encoder_hash=encoder["hash"])
            if ref_scale is None:
                ref_scale = aux["scale"]
            elif not np.array_equal(ref_scale, aux["scale"]):
                raise ValueError(
                    f"scale mismatch across checkpoints at retrain fold "
                    f"{retrain_fold} ({checkpoint_key(mid, r)})")
            loaded[checkpoint_key(mid, r)] = (model, aux, manifest)
        first_aux = next(iter(loaded.values()))[1]
        static_in = build_static_inputs(static, encoder)

        for fold in folds:
            fr = fold_rows[fold]
            paths, draw, path_sha = scenarios[fold]
            shock_start_idx = fr["test_start_idx"] + draw.start_offset
            dpf = fold_dp(dp, demand, paths, fr)
            panel = build_panel(dpf, cal, static, encoder,
                                first_aux["scale"], first_aux["price_mean"],
                                idx_start=1, idx_end=max_cal)
            origins = list(range(fr["test_start_idx"],
                                 fr["test_end_idx"] + 1))
            rows, orgs = inference_pairs(panel, origins, cfg)
            t_fold = time.time()
            for key, (model, aux, manifest) in loaded.items():
                mid = key.split("_")[0]
                raw = predict(model, panel, static_in, rows, orgs, cfg,
                              batch_size=args.batch_size)
                sid, org_l, hor = long_arrays(panel, rows, orgs, cfg)
                frame = point_store.build_point_frame(
                    sid, org_l, hor, raw.reshape(-1))
                stats = point_store.validate_point_frame(
                    frame, MODEL_KIND[mid])
                check = crosscheck(frame, key, fold, series_group,
                                   draw.group_id, shock_start_idx)
                out_path = EVAL_DIR / f"tsmixer_{key}_{SCENARIO}_f{fold}.parquet"
                frame.to_parquet(out_path, index=False)
                back = pd.read_parquet(out_path)
                if not frames_equal(frame.reset_index(drop=True), back):
                    raise RuntimeError(f"reload mismatch for {out_path}")
                entries[key].append({
                    "fold": fold,
                    "file": out_path.name,
                    "prediction_file_sha256": file_sha256(out_path),
                    **stats,
                    "origin_start": origins[0],
                    "origin_end": origins[-1],
                    "origin_count": len(origins),
                    "retrain_fold": retrain_fold,
                    "checkpoint_sha256": manifest["state_dict_sha256"],
                    "probe_max_abs_diff": aux["probe_max_abs_diff"],
                    "encoder_hash": encoder["hash"],
                    "demand_path_sha256": path_sha,
                    "shock": {
                        "group_id": draw.group_id,
                        "start_offset": draw.start_offset,
                        "shock_start_idx": shock_start_idx,
                        "duration_days": draw.duration_days,
                        "multiplier": draw.multiplier,
                    },
                    "crosscheck": check,
                })
            print(f"fold {fold:02d} ckpt_f{retrain_fold} "
                  f"origins={len(origins)} fold_s={time.time() - t_fold:.0f}")

    for (mid, r) in models:
        key = checkpoint_key(mid, r)
        write_json(EVAL_DIR / f"tsmixer_{key}_{SCENARIO}_manifest.json", {
            "model": f"tsmixer_{key}",
            "model_id": mid,
            "r_train": r,
            "prediction_kind": MODEL_KIND[mid],
            "schema_version": point_store.POINT_SCHEMA_VERSION,
            "prediction_purpose": "rolling_decision_replay",
            "scenario_id": {
                "name": SCENARIO,
                "p": SCENARIO_P,
                "shock_rule": "pre-registered fold-level draw (seed D)",
            },
            "history_observation_rule": "exogenous_latent",
            "forecast_horizon": 28,
            "input_normalization": "checkpoint_aux(scale, price_mean)",
            "crosscheck_tolerance": {"rtol": TOL, "atol": TOL},
            "folds": entries[key],
            "runtime": runtime,
        })
        print(f"manifest written for tsmixer_{key}_{SCENARIO}")


if __name__ == "__main__":
    main()