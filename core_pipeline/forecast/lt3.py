"""Stage 3-E lead-time (protection-window) cumulative demand methods
(ST3-08). Three ways to produce quantiles of Y_o = d_{o+1}+d_{o+2}+d_{o+3}:

A. Direct model (scripts/train_lgbm_lt3.py): LightGBM on the lt_cum
   target -- not built here.
B. comonotone_from_daily: per-tau summation of the daily quantiles,
   Q_tau(Y) ~= sum_h Q_tau(d_{o+h}). Exact iff the days are COMONOTONIC
   (one shared uniform drives all three marginals). This is a COUPLING
   ASSUMPTION, NOT a universal pointwise upper bound on the cumulative
   quantile: with non-overlapping intermittent demand the true sum
   quantile can exceed the comonotonic sum (the convex-order dominance
   result does not give pointwise quantile bounds).
C. independent_mc_from_daily: conditional-independence coupling via
   Monte Carlo from a TRUNCATED piecewise-linear inverse CDF built on
   the 11 stored quantiles. Tails outside [0.05, 0.95] are truncated at
   the end quantile values -- results are conditional on that tail
   assumption; a larger N does not fix a wrong coupling or tail rule.

Interpretation discipline (ST3-08): performance differences across
A/B/C mix model estimation error, MC/interpolation error, and coupling
assumptions. They are NOT direct evidence about the true temporal
dependence structure of demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES

LT_K = store.LT_AGG_DAYS   # 3-day protection window
MC_N = 4096                # pre-fixed MC budget (reproducibility choice)
MC_SEED = 20261009         # base seed; stream = [MC_SEED, origin_idx]


def _pivot_daily(daily: pd.DataFrame, qcols: list[str]):
    """(series_id, origin_idx, qcube (R, K, Q)) from a DAILY frame,
    h = 1..K rows required per key; sorted by (origin, series)."""
    store.assert_daily_frame(daily)
    d = daily[daily["horizon"] <= LT_K]
    sizes = d.groupby(["series_id", "origin_idx"]).size()
    if len(d) == 0 or not (sizes == LT_K).all():
        raise ValueError(f"every (series, origin) needs h=1..{LT_K} rows")
    d = d.sort_values(["origin_idx", "series_id", "horizon"],
                      ignore_index=True)
    q = d[qcols].to_numpy(dtype=np.float64).reshape(-1, LT_K, len(qcols))
    keys = d.iloc[::LT_K]
    return (keys["series_id"].to_numpy(), keys["origin_idx"].to_numpy(),
            q)


def comonotone_from_daily(daily: pd.DataFrame,
                          quantiles=QUANTILES) -> pd.DataFrame:
    qcols = store.quantile_columns(quantiles)
    sid, org, q = _pivot_daily(daily, qcols)
    return store.build_lt_frame(sid, org, q.sum(axis=1), quantiles)


def sample_from_quantiles(qvals: np.ndarray, u: np.ndarray,
                          levels) -> np.ndarray:
    """Truncated piecewise-linear inverse CDF. qvals (R, Q) must be
    row-wise non-decreasing (rearranged store output); u (R, N) uniforms.
    u outside [levels[0], levels[-1]] is clipped (tail truncation)."""
    lv = np.asarray(levels, dtype=np.float64)
    u = np.clip(u, lv[0], lv[-1])
    idx = np.clip(np.searchsorted(lv, u, side="right"), 1, len(lv) - 1)
    w = (u - lv[idx - 1]) / (lv[idx] - lv[idx - 1])
    rows = np.arange(qvals.shape[0])[:, None]
    v_lo = qvals[rows, idx - 1]
    v_hi = qvals[rows, idx]
    return v_lo + w * (v_hi - v_lo)


def independent_mc_from_daily(daily: pd.DataFrame, quantiles=QUANTILES,
                              n: int = MC_N,
                              seed: int = MC_SEED) -> pd.DataFrame:
    qcols = store.quantile_columns(quantiles)
    sid, org, q = _pivot_daily(daily, qcols)
    lv = np.asarray(quantiles, dtype=np.float64)
    out = np.empty((len(sid), len(lv)))
    for o in np.unique(org):
        mask = org == o
        qo = q[mask]                                   # (S, K, Q)
        rng = np.random.default_rng([seed, int(o)])    # per-origin stream
        u = rng.random((qo.shape[0], LT_K, n))
        s = np.zeros((qo.shape[0], n))
        for k in range(LT_K):
            s += sample_from_quantiles(qo[:, k, :], u[:, k, :], lv)
        out[mask] = np.quantile(s, lv, axis=1).T
    return store.build_lt_frame(sid, org, out, quantiles)
