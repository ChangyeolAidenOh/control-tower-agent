"""Stage 4 comparison report (ST4-08). Descriptive only: re-aggregates
FROZEN prediction artifacts under labeled evaluation masks and records
the H2 input-model decision. No new methodology, no H1/H2 verdicts
(Stage 6), no interval re-calibration.

Blocks:
  1. decile9 probabilistic table -- lgbm (9 of 11 columns), chronos2s
     (9 of 11, pretraining-overlap caveat), tsmixer_a3 (native 9) --
     WQL on QUANTILES_9, mask stage3_in_window, headline + by_fold +
     by_h (per-h origin sets differ; annotated).
  2. Point table (main) -- tsmixer_a1 (mse_point), tsmixer_a3 q0.50,
     lgbm q0.50, chronos2s q0.50 -- WAPE, nonzero MAPE, MASE (macro over
     series x fold cells; naive trained on the series' active span up to
     the fold train_end, m=7; convention stated, shared denominator per
     cell so cross-model comparison is unaffected).
  3. Cost-aware point table (separate, unranked) -- tsmixer_a2_r3
     (target tau = 0.75), tsmixer_a2_r5 (tau = 5/6): same descriptive
     metrics; never interpreted as policy quality (Stage 5-6 decides).
  4. A3 crossing rates per fold (from the run_tsmixer manifest).
  5. Implementation-equivalence check: tuning val_wql9 formula vs
     metrics.wql on identical inputs (one fold).
  6. H2 input decision (citation only): LightGBM 11-quantile fixed as
     the primary ACI input (ST4-08); prior exposure to Stage 3 coverage
     is disclosed; A3-based ACI deferred to exploratory follow-up.
     Numbers cited verbatim from outputs/intervals/lgbm_aci_summary.json.

Self-checks (hard fail -- these anchor this script to the Stage 3
definitions without re-implement drift):
  - lgbm stage3_in_window row count == 750,288 (ST3-09 freeze)
  - lgbm 11-quantile headline WQL reproduces the frozen 0.37923
Mask discipline: every table carries evaluation_mask_id and, where
quantiles are involved, quantile_set_id. 11Q and decile9 are never
mixed in one table.

Run: python -m scripts.report_stage4
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.forecast import point_store
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES, write_json
from core_pipeline.forecast.metrics import (
    macro_mean, nonzero_mape, seasonal_naive_mae, wape, wql,
)
from core_pipeline.forecast.tsmixer.config import QUANTILES_9

LAST_TRUTH = 1941
FROZEN_LGBM_ROWS = 750_288
FROZEN_LGBM_WQL11 = 0.37923
JSON_OUT = Path("outputs/reports/stage4_report.json")
MD_OUT = Path("docs/stage4_report.md")
CHRONOS_CAVEAT = ("Chronos-2 Small: public pretraining corpus links a "
                  "collection containing M5; overlap cannot be ruled "
                  "out -- reference performance, not verified zero-shot.")
Q9_COLS = [f"q{q:.2f}" for q in QUANTILES_9]
Q11_COLS = store.quantile_columns(QUANTILES)
H2_DECISION = (
    "H2 primary input model: LightGBM 11-quantile (ST4-08). Chosen to "
    "keep the ST3-07 ACI contract unchanged (q0.05/q0.95 base interval, "
    "28-day fixed feedback delay, online update rule, initial state). "
    "The 12-fold WQL ranking between A3 and LightGBM is NOT used for "
    "this selection: those folds are the preregistered final evaluation "
    "set, and selecting on them would couple model selection with H2 "
    "evaluation. Disclosure: Stage 3 already produced and reported the "
    "LightGBM ACI coverage (0.8885 after calibration), so this decision "
    "was not made blind to prior evaluation results. A3-based ACI on the "
    "q0.10/q0.90 base interval is theoretically possible but a different "
    "experimental setting; deferred to exploratory follow-up. Stage 4 "
    "does not re-optimize any interval parameters.")


def stage_from_db():
    """Phase A: truth panel + folds; imports and connection stay local."""
    from core_pipeline.data import db
    from core_pipeline.forecast.dataset import MartPanelSource
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    folds = source.load_folds()
    static = source.load_static()
    dp = source.load_demand_price(max_period_idx=LAST_TRUTH)
    conn.close()
    wide = dp.pivot(index="series_id", columns="period_idx", values="sales")
    series_order = [str(s) for s in static["series_id"]]
    wide = wide.reindex(index=series_order)
    idx_start = int(wide.columns.min())
    truth = np.nan_to_num(wide.to_numpy(dtype=np.float64), nan=0.0)
    sid_rank = {s: i for i, s in enumerate(series_order)}
    first_active = pd.to_numeric(
        static["first_active_idx"]).to_numpy(dtype=np.int64)
    return folds, truth, idx_start, sid_rank, first_active


def attach(df: pd.DataFrame, truth, idx_start, sid_rank) -> pd.DataFrame:
    rows = df["series_id"].map(sid_rank).to_numpy(dtype=np.int64)
    cols = df["target_idx"].to_numpy(dtype=np.int64) - idx_start
    out = df.copy()
    out["y"] = truth[rows, cols]
    return out


def load_frames(model: str, folds: pd.DataFrame, point: bool) -> pd.DataFrame:
    parts = []
    for fold in folds["fold"]:
        df = (point_store.load_point_fold(model, int(fold)) if point
              else store.load_fold(model, int(fold)))
        df["fold"] = int(fold)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    win = folds.set_index("fold")
    lo = out["fold"].map(win["test_start_idx"]).to_numpy()
    hi = out["fold"].map(win["test_end_idx"]).to_numpy()
    out["in_window"] = ((out["target_idx"].to_numpy() >= lo)
                        & (out["target_idx"].to_numpy() <= hi))
    return out


def wql_block(d: pd.DataFrame, qcols, quantiles, set_id: str) -> dict:
    y = d["y"].to_numpy(dtype=np.float64)
    qm = d[qcols].to_numpy(dtype=np.float64)
    by_fold = {
        str(f): round(float(wql(g["y"].to_numpy(),
                                g[qcols].to_numpy(dtype=np.float64),
                                quantiles)), 5)
        for f, g in d.groupby("fold")}
    by_h = {
        str(h): {"wql": round(float(wql(g["y"].to_numpy(),
                                        g[qcols].to_numpy(dtype=np.float64),
                                        quantiles)), 5),
                 "n_origins": int(g["origin_idx"].nunique())}
        for h, g in d.groupby("horizon")}
    return {
        "evaluation_mask_id": "stage3_in_window",
        "quantile_set_id": set_id,
        "rows": int(len(d)),
        "wql": round(float(wql(y, qm, quantiles)), 5),
        "by_fold": by_fold,
        "by_h_note": ("per-h origin sets differ with h; h-curves are not "
                      "pure forecast-distance effects"),
        "by_h": by_h,
    }


def point_block(d: pd.DataFrame, yhat_col: str, folds, truth, idx_start,
                sid_rank, first_active, kind: str) -> dict:
    y = d["y"].to_numpy(dtype=np.float64)
    p = d[yhat_col].to_numpy(dtype=np.float64)
    nz_mape, n_nz = nonzero_mape(y, p)
    win = folds.set_index("fold")
    cells = []
    for (fold, sid), g in d.groupby(["fold", "series_id"]):
        r = sid_rank[str(sid)]
        t_end = int(win.loc[fold, "train_end_idx"])
        lo = max(int(first_active[r]), idx_start)
        train_values = truth[r, lo - idx_start: t_end - idx_start + 1]
        denom = seasonal_naive_mae(train_values, m=7)
        mae = float(np.abs(g["y"].to_numpy() - g[yhat_col].to_numpy()).mean())
        cells.append(mae / denom if denom > 0 else np.nan)
    mase_macro, n_undef = macro_mean(np.asarray(cells, dtype=np.float64))
    return {
        "evaluation_mask_id": "stage3_in_window",
        "prediction_kind": kind,
        "rows": int(len(d)),
        "wape": round(float(wape(y, p)), 5),
        "nonzero_mape": round(float(nz_mape), 5),
        "nonzero_mape_n": int(n_nz),
        "mase_macro": round(float(mase_macro), 5),
        "mase_cells_undef": int(n_undef),
        "mase_note": ("macro over series x fold cells; naive m=7 trained "
                      "on the series' active span through the fold "
                      "train_end -- convention local to this report, "
                      "denominator shared across models per cell"),
    }


def equivalence_check(d_a3: pd.DataFrame) -> dict:
    """val_wql9 implementation vs metrics.wql on identical inputs."""
    g = d_a3[d_a3["fold"] == 1]
    y = g["y"].to_numpy(dtype=np.float64)
    qm = g[Q9_COLS].to_numpy(dtype=np.float64)
    tau = np.asarray(QUANTILES_9, dtype=np.float64).reshape(1, -1)
    diff = y[:, None] - qm
    mine = 2.0 * float(np.maximum(tau * diff, (tau - 1.0) * diff).sum()) \
        / (len(QUANTILES_9) * float(np.abs(y).sum()))
    official = float(wql(y, qm, QUANTILES_9))
    ok = bool(np.isclose(mine, official, rtol=1e-9, atol=1e-12))
    return {"val_wql9_formula": round(mine, 8),
            "metrics_wql": round(official, 8), "match": ok}


def main() -> None:
    folds, truth, idx_start, sid_rank, first_active = stage_from_db()

    frames = {}
    for model in ("lgbm", "chronos2s", "tsmixer_a3"):
        f = load_frames(model, folds, point=False)
        frames[model] = attach(f[f["in_window"]], truth, idx_start, sid_rank)
    for model in ("tsmixer_a1", "tsmixer_a2_r3", "tsmixer_a2_r5"):
        f = load_frames(model, folds, point=True)
        frames[model] = attach(f[f["in_window"]], truth, idx_start, sid_rank)

    # anchors: Stage 3 mask and frozen headline must reproduce exactly
    n_lgbm = len(frames["lgbm"])
    if n_lgbm != FROZEN_LGBM_ROWS:
        raise SystemExit(f"mask drift: lgbm in-window rows {n_lgbm} != "
                         f"{FROZEN_LGBM_ROWS}")
    wql11 = float(wql(frames["lgbm"]["y"].to_numpy(),
                      frames["lgbm"][Q11_COLS].to_numpy(dtype=np.float64),
                      QUANTILES))
    if abs(wql11 - FROZEN_LGBM_WQL11) > 1e-4:
        raise SystemExit(f"frozen-WQL drift: lgbm 11Q {wql11:.6f} != "
                         f"{FROZEN_LGBM_WQL11}")
    print(f"anchors ok: rows={n_lgbm} lgbm_wql11={wql11:.5f}")

    decile9 = {m: wql_block(frames[m], Q9_COLS, QUANTILES_9, "decile9")
               for m in ("lgbm", "chronos2s", "tsmixer_a3")}
    point_main = {
        "tsmixer_a1": point_block(frames["tsmixer_a1"], "yhat", folds,
                                  truth, idx_start, sid_rank, first_active,
                                  "mse_point"),
        "tsmixer_a3_q50": point_block(
            frames["tsmixer_a3"].rename(columns={"q0.50": "p50"}), "p50",
            folds, truth, idx_start, sid_rank, first_active,
            "quantile_grid_median"),
        "lgbm_q50": point_block(
            frames["lgbm"].rename(columns={"q0.50": "p50"}), "p50", folds,
            truth, idx_start, sid_rank, first_active, "quantile_median"),
        "chronos2s_q50": point_block(
            frames["chronos2s"].rename(columns={"q0.50": "p50"}), "p50",
            folds, truth, idx_start, sid_rank, first_active,
            "quantile_median_reference"),
    }
    point_cost = {
        "tsmixer_a2_r3": {**point_block(
            frames["tsmixer_a2_r3"], "yhat", folds, truth, idx_start,
            sid_rank, first_active, "cost_weighted_point"),
            "target_tau": 0.75},
        "tsmixer_a2_r5": {**point_block(
            frames["tsmixer_a2_r5"], "yhat", folds, truth, idx_start,
            sid_rank, first_active, "cost_weighted_point"),
            "target_tau": round(5 / 6, 5)},
        "note": ("cost-weighted statistics target different quantiles; "
                 "unranked -- policy quality is decided by downstream "
                 "cost and service level (Stage 5-6)"),
    }
    a3_manifest = json.loads(
        (store.PRED_DIR / "tsmixer_a3_manifest.json").read_text())
    crossing = {str(e["fold"]): e.get("crossing_rate_before")
                for e in a3_manifest["folds"]}
    aci_path = Path("outputs/intervals/lgbm_aci_summary.json")
    aci_cite = (json.loads(aci_path.read_text()).get("primary_full_period")
                if aci_path.exists() else "summary not found; cite manually")

    report = {
        "decision": "ST4-08",
        "chronos_caveat": CHRONOS_CAVEAT,
        "frozen_anchors": {"lgbm_rows": n_lgbm,
                           "lgbm_wql11": round(wql11, 5),
                           "frozen_reference": FROZEN_LGBM_WQL11},
        "decile9": decile9,
        "point_main": point_main,
        "point_cost_aware": point_cost,
        "a3_crossing_rate_before_by_fold": crossing,
        "equivalence_check": equivalence_check(frames["tsmixer_a3"]),
        "h2_input_decision": H2_DECISION,
        "h2_citation_primary_full_period": aci_cite,
    }
    write_json(JSON_OUT, report)

    lines = ["# Stage 4 comparison report (ST4-08)\n",
             "Descriptive only. No H1/H2 verdicts (Stage 6). Every table "
             "carries evaluation_mask_id; quantile sets are never mixed.\n",
             f"> {CHRONOS_CAVEAT}\n",
             "## 1. decile9 WQL (mask: stage3_in_window, "
             "quantile_set_id: decile9)\n",
             "| model | WQL9 | rows |", "|---|---|---|"]
    for m, b in decile9.items():
        lines.append(f"| {m} | {b['wql']:.5f} | {b['rows']:,} |")
    lines += ["\nFrozen 11-quantile headline (separate set, never mixed): "
              f"lgbm {wql11:.5f} (reference {FROZEN_LGBM_WQL11}).\n",
              "## 2. Point metrics (mask: stage3_in_window)\n",
              "| model | kind | WAPE | nonzero MAPE | MASE |",
              "|---|---|---|---|---|"]
    for m, b in point_main.items():
        lines.append(f"| {m} | {b['prediction_kind']} | {b['wape']:.5f} | "
                     f"{b['nonzero_mape']:.5f} | {b['mase_macro']:.5f} |")
    lines += ["\n## 3. Cost-aware point statistics (separate, unranked)\n",
              "| model | target tau | WAPE | nonzero MAPE | MASE |",
              "|---|---|---|---|---|"]
    for m in ("tsmixer_a2_r3", "tsmixer_a2_r5"):
        b = point_cost[m]
        lines.append(f"| {m} | {b['target_tau']} | {b['wape']:.5f} | "
                     f"{b['nonzero_mape']:.5f} | {b['mase_macro']:.5f} |")
    lines += [f"\n{point_cost['note']}\n",
              "## 4. A3 crossing rate before rearrangement (by fold)\n",
              json.dumps(crossing), "",
              "## 5. Implementation equivalence\n",
              json.dumps(report["equivalence_check"]), "",
              "## 6. H2 input model decision\n", H2_DECISION, "",
              f"Cited coverage (verbatim): {json.dumps(aci_cite)}", ""]
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(lines))
    for m, b in decile9.items():
        print(f"{m}: wql9={b['wql']:.5f}")
    for m, b in point_main.items():
        print(f"{m}: wape={b['wape']:.5f} mase={b['mase_macro']:.5f}")
    print(f"equivalence match={report['equivalence_check']['match']}")
    print(f"wrote {JSON_OUT} and {MD_OUT}")


if __name__ == "__main__":
    main()
