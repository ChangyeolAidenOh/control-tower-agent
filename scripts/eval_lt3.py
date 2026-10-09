"""Evaluate the three Stage 3-E cumulative-demand methods (ST3-08) on
the identical evaluation set E_lt = {(i, o): o + 3 <= 1941} and report
coverage / width / WQL / per-quantile pinball, pooled and by fold.

Interpretation discipline (fixed): differences across methods mix model
estimation error, MC and interpolation error, and coupling assumptions;
they are NOT direct evidence about the temporal dependence structure.
These 12-fold results are post-hoc diagnostics of pre-fixed methods and
are NOT used to select Stage 5 confirmatory policy inputs (the A3
TSMixer-Ext SAA contract stays as preregistered).

Appendix diagnostics (development window <= d_1605 only): lag-1/2
Spearman rank correlation of daily demand and consecutive high-demand
co-occurrence lift -- descriptive only, no seasonality/covariate
control, not causal dependence estimates.

Run: python -m scripts.eval_lt3
"""

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES, write_json
from core_pipeline.forecast.dataset import DemandHistory, MartPanelSource
from core_pipeline.forecast.metrics import pinball, wql

MODELS = ("lgbm_lt3", "lt3_como", "lt3_indep")
LAST_TRUTH = 1941
OUT = "outputs/reports/lt3_eval.json"
DEV_END = 1605


def load_all(model: str, n_folds: int) -> pd.DataFrame:
    parts = []
    for fold in range(1, n_folds + 1):
        df = store.load_fold(model, fold)
        store.validate_lt_frame(df)          # refuses daily frames
        df["fold"] = fold
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["origin_idx", "series_id"], ignore_index=True)


def metric_block(y: np.ndarray, qm: np.ndarray) -> dict:
    lo, hi = qm[:, 0], qm[:, -1]
    return {
        "rows": int(len(y)),
        "coverage90": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width90": float((hi - lo).mean()),
        "wql": wql(y, qm, QUANTILES),
        "pinball_by_q": {f"{q}": float(pinball(y, qm[:, j], q).mean())
                         for j, q in enumerate(QUANTILES)},
    }


def dependence_diagnostics(dev: np.ndarray) -> dict:
    def spearman_lag(lag: int) -> float:
        vals = []
        for row in dev:
            a = pd.Series(row[:-lag]).rank().to_numpy()
            b = pd.Series(row[lag:]).rank().to_numpy()
            if a.std() == 0 or b.std() == 0:
                continue
            vals.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.median(vals))

    thr = np.quantile(dev, 0.9, axis=1, keepdims=True)
    high = dev > thr
    p_high = high.mean(axis=1)
    p_joint = (high[:, :-1] & high[:, 1:]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(p_high > 0, p_joint / p_high**2, np.nan)
    return {
        "window": f"development <= d_{DEV_END}",
        "note": ("descriptive only; no seasonality/trend/covariate "
                 "control; not a causal dependence estimate"),
        "spearman_lag1_median": spearman_lag(1),
        "spearman_lag2_median": spearman_lag(2),
        "consec_high_cooccurrence_lift_median": float(np.nanmedian(lift)),
    }


def main() -> None:
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    full = DemandHistory.from_source(source, max_period_idx=LAST_TRUTH)
    n_folds = int(source.load_folds()["fold"].max())
    conn.close()

    truth = full.observed_through(LAST_TRUTH)
    sid_rank = {s: i for i, s in enumerate(full.series_ids)}
    qcols = store.quantile_columns()

    frames = {m: load_all(m, n_folds) for m in MODELS}
    base_keys = frames[MODELS[0]][["series_id", "origin_idx"]]
    for m in MODELS[1:]:
        if not base_keys.equals(frames[m][["series_id", "origin_idx"]]):
            raise SystemExit(f"{m}: key set differs from {MODELS[0]}")

    report = {"decision": "ST3-08",
              "eval_set": "E_lt = {o + 3 <= 1941}, identical across "
                          "methods",
              "interpretation_note": (
                  "method differences mix estimation error and coupling "
                  "assumptions; not evidence about true temporal "
                  "dependence; not used for Stage 5 confirmatory input "
                  "selection"),
              "methods": {}}
    for m, df in frames.items():
        evalable = df["target_end_idx"].to_numpy() <= LAST_TRUTH
        rows = df[evalable]
        srow = rows["series_id"].map(sid_rank).to_numpy()
        o = rows["origin_idx"].to_numpy()
        cols = o[:, None] + np.arange(1, store.LT_AGG_DAYS + 1)
        y = truth[srow[:, None], cols - full.idx_start].sum(axis=1)
        qm = rows[qcols].to_numpy(dtype=np.float64)
        entry = {"pooled": metric_block(y, qm), "by_fold": {}}
        for f in sorted(rows["fold"].unique()):
            fm = rows["fold"].to_numpy() == f
            entry["by_fold"][str(f)] = metric_block(y[fm], qm[fm])
        report["methods"][m] = entry
        p = entry["pooled"]
        print(f"{m}: rows={p['rows']} cov90={p['coverage90']:.4f} "
              f"width={p['mean_width90']:.3f} wql={p['wql']:.5f}")

    report["dependence_appendix"] = dependence_diagnostics(
        np.asarray(full.observed_through(DEV_END), dtype=np.float64))
    write_json(OUT, report)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
