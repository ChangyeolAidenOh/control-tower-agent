"""ST7-02 P0-1: agent-selection A3 replay parquet (weekly candidate origins).

Weekly candidates are selected at the close of day t_w = T - 1 + 7w
(w = 0..3, T = fold test_start_idx), BEFORE the first action day's demand
is observed (ST7-02 D6/P0-1). The Stage 6 evaluation replay parquet keeps
decision days [T, T_end] only (ST6-07 design), so the w = 0 origin T - 1
is absent there. This script regenerates ALL FOUR selection origins per
fold with the SAME machinery as build_eval_predictions (scenario gates ->
fold_dp -> frozen checkpoints -> monotone quantiles) and hard-gates every
row against pre-existing artifacts before the parquet is written:

  origin T-1        vs the committed Stage 3/4 scoring parquet
                    (outputs/predictions/tsmixer_a3_f{fold}.parquet).
                    Structural identity holds: the core scenario has p = 0
                    and the shock starts inside [T, T_end] (draw_shock
                    start_offset >= 0), so the conditioning history up to
                    T - 1 equals the mart history; future sales are never
                    model inputs. The gate turns this argument into a
                    numeric check (the reviewer's reuse condition).
  origins T+6/13/20 vs the Stage 6 evaluation replay parquet
                    (outputs/predictions_eval/tsmixer_a3_core_f{fold}).

Both gates require bit-exact rows; the pre-fixed rel/abs 1e-6 fallback of
ST6-02 (3) is recorded if ever used, and anything beyond it hard-fails.
Selection-only artifact: used for candidate gating/ranking, never for
H3/H5 cost measurement (realization stays bp.test, ST7-01).

Run:
  python -m scripts.build_agent_selection_predictions --folds 1-12
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.data.mart_reader import fetch_folds
from core_pipeline.forecast import prediction_store as qstore
from core_pipeline.forecast.common import file_sha256, write_json
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND,
    QUANTILES_9,
    TSMixerConfig,
    checkpoint_fold_for,
    checkpoint_key,
)
from core_pipeline.simulator.engine import load_sim_config
from scripts.build_eval_predictions import (
    COMMITTED_DIR,
    EVAL_DIR,
    SCENARIO_P,
    TOL,
    fold_dp,
    frames_equal,
    load_encoder,
    scenario_for_fold,
    stage_from_db,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "outputs" / "predictions_agent"
SCENARIO = "agent"
MODEL_ID = "a3"
N_WEEKS = 4
WEEK_DAYS = 7
FORECAST_DAYS = 28


def selection_origins(fold_row: dict) -> list[int]:
    """t_w = test_start - 1 + 7w for w = 0..3 (ST7-02 D6)."""
    ts = fold_row["test_start_idx"]
    return [ts - 1 + WEEK_DAYS * w for w in range(N_WEEKS)]


def gate_rows(frame: pd.DataFrame, ref: pd.DataFrame, origins: list[int],
              label: str, fold: int) -> dict:
    """Every (series, origin, horizon) row of `frame` restricted to `origins`
    must exist in `ref` and match on all 9 quantile columns: bit-exact
    preferred, rel/abs TOL fallback recorded, beyond that -> hard fail."""
    qcols = qstore.quantile_columns(QUANTILES_9)
    sub = frame[frame["origin_idx"].isin(origins)]
    if sub.empty:
        raise RuntimeError(f"{label} fold {fold}: no rows at origins {origins}")
    merged = sub.merge(ref, on=["series_id", "origin_idx", "horizon"],
                       suffixes=("_new", "_ref"))
    if len(merged) != len(sub):
        raise RuntimeError(
            f"{label} fold {fold}: {len(sub) - len(merged)} rows missing "
            f"from the reference parquet at origins {origins}")
    new = merged[[c + "_new" for c in qcols]].to_numpy(np.float64)
    old = merged[[c + "_ref" for c in qcols]].to_numpy(np.float64)
    diff = np.abs(new - old)
    exact = bool((diff == 0.0).all())
    max_diff = float(diff.max())
    if not exact and not np.isclose(new, old, rtol=TOL, atol=TOL).all():
        raise RuntimeError(
            f"{label} fold {fold}: gate failed, max_abs_diff "
            f"{max_diff:.3e} beyond rel/abs {TOL}")
    return {
        "label": label,
        "origins": origins,
        "rows": int(len(sub)),
        "bit_exact": exact,
        "max_abs_diff": max_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1-12")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--staging-root", default="outputs/staging/tsmixer")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise RuntimeError("kappa_selected missing in configs/simulator.yaml")

    t0 = time.time()
    db.load_dotenv()
    conn = db.connect(db.load_config())
    all_folds_meta = fetch_folds(conn)
    conn.close()
    fold_rows = {f["fold"]: f for f in all_folds_meta}
    max_test_end = max(fold_rows[f]["test_end_idx"] for f in fold_list)
    dp, cal, static, max_cal, _, demand = stage_from_db(max_test_end)
    n_series = sum(len(demand[gid][0]) for gid in demand)
    print(f"staged rows={len(dp)} folds={fold_list} "
          f"stage_s={time.time() - t0:.0f}")

    scenarios = {}
    for f in fold_list:
        scenarios[f] = scenario_for_fold(demand, fold_rows[f], sim_cfg)
        draw = scenarios[f][1]
        shock_start_idx = fold_rows[f]["test_start_idx"] + draw.start_offset
        t_minus_1 = fold_rows[f]["test_start_idx"] - 1
        if t_minus_1 >= shock_start_idx:
            raise RuntimeError(
                f"fold {f}: shock_start {shock_start_idx} <= T-1 {t_minus_1}")
        print(f"fold {f:02d} scenario ok shock={draw.group_id} "
              f"start_offset={draw.start_offset}")

    from core_pipeline.forecast.tsmixer.checkpoint import load_checkpoint
    from core_pipeline.forecast.tsmixer.infer import (
        inference_pairs, long_arrays, predict,
    )
    from core_pipeline.forecast.tsmixer.panel import (
        build_panel, build_static_inputs,
    )
    from core_pipeline.forecast.tsmixer.postproc import finalize_quantiles

    cfg = TSMixerConfig()
    ckpt_dir = Path(args.ckpt_dir)
    staging_root = Path(args.staging_root)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    key = checkpoint_key(MODEL_ID, None)
    entries = []
    runtime = {"device": "cpu", "batch_size": args.batch_size}

    by_retrain: dict[int, list[int]] = {}
    for fold in fold_list:
        by_retrain.setdefault(checkpoint_fold_for(fold), []).append(fold)

    for retrain_fold, folds in sorted(by_retrain.items()):
        encoder = load_encoder(staging_root, retrain_fold)
        model, aux, manifest = load_checkpoint(
            ckpt_dir, MODEL_ID, retrain_fold, None,
            expected_encoder_hash=encoder["hash"])
        static_in = build_static_inputs(static, encoder)

        for fold in folds:
            fr = fold_rows[fold]
            paths, draw, path_sha = scenarios[fold]
            shock_start_idx = fr["test_start_idx"] + draw.start_offset
            dpf = fold_dp(dp, demand, paths, fr)
            panel = build_panel(dpf, cal, static, encoder,
                                aux["scale"], aux["price_mean"],
                                idx_start=1, idx_end=max_cal)
            origins = selection_origins(fr)
            rows, orgs = inference_pairs(panel, origins, cfg)
            t_fold = time.time()
            raw = predict(model, panel, static_in, rows, orgs, cfg,
                          batch_size=args.batch_size)
            sid, org_l, hor = long_arrays(panel, rows, orgs, cfg)
            fixed, crossing = finalize_quantiles(raw)
            frame = qstore.build_frame(
                sid, org_l, hor,
                fixed.reshape(-1, len(QUANTILES_9)), QUANTILES_9)
            stats = qstore.validate_frame(frame, QUANTILES_9)
            stats["crossing_rate_before"] = crossing
            expected = len(origins) * n_series * FORECAST_DAYS
            if len(frame) != expected:
                raise RuntimeError(
                    f"fold {fold}: frame rows {len(frame)} != {expected}")

            committed = pd.read_parquet(
                COMMITTED_DIR / f"tsmixer_{key}_f{fold}.parquet")
            eval_ref = pd.read_parquet(
                EVAL_DIR / f"tsmixer_{key}_core_f{fold}.parquet")
            gate_t = gate_rows(frame, committed, origins[:1],
                               "vs_committed_t_minus_1", fold)
            gate_w = gate_rows(frame, eval_ref, origins[1:],
                               "vs_eval_weekly", fold)

            out_path = AGENT_DIR / f"tsmixer_{key}_{SCENARIO}_f{fold}.parquet"
            frame.to_parquet(out_path, index=False)
            back = pd.read_parquet(out_path)
            if not frames_equal(frame.reset_index(drop=True), back):
                raise RuntimeError(f"reload mismatch for {out_path}")
            entries.append({
                "fold": fold,
                "file": out_path.name,
                "prediction_file_sha256": file_sha256(out_path),
                **stats,
                "selection_origins": origins,
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
                "gates": {"t_minus_1": gate_t, "weekly": gate_w},
            })
            print(f"fold {fold:02d} ckpt_f{retrain_fold} "
                  f"origins={origins} exact={gate_t['bit_exact']}/"
                  f"{gate_w['bit_exact']} fold_s={time.time() - t_fold:.0f}")

    write_json(AGENT_DIR / f"tsmixer_{key}_{SCENARIO}_manifest.json", {
        "model": f"tsmixer_{key}",
        "model_id": MODEL_ID,
        "r_train": None,
        "prediction_kind": MODEL_KIND[MODEL_ID],
        "schema_version": qstore.SCHEMA_VERSION
        if hasattr(qstore, "SCHEMA_VERSION") else "see_eval_manifest",
        "prediction_purpose": "agent_candidate_selection_replay",
        "scenario_id": {
            "name": SCENARIO,
            "p": SCENARIO_P,
            "shock_rule": "pre-registered fold-level draw (seed D)",
            "note": "same core scenario demand; origins = weekly selection "
                    "days T-1+7w, w=0..3 (ST7-02 D6/P0-1)",
        },
        "history_observation_rule": "exogenous_latent",
        "forecast_horizon": FORECAST_DAYS,
        "input_normalization": "checkpoint_aux(scale, price_mean)",
        "crosscheck_tolerance": {"rtol": TOL, "atol": TOL},
        "folds": entries,
        "runtime": runtime,
    })
    print(f"manifest written for tsmixer_{key}_{SCENARIO}")


if __name__ == "__main__":
    main()