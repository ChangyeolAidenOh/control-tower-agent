"""Forecast evaluation metrics (Stage 3-B contract, decision ST3-03).

Formulas are fixed before any model results are seen (preregistration v1.2
SS3: forecast accuracy is auxiliary reporting, ranked but never claimed).

Evaluation rows E are forecast INSTANCES (series, origin, horizon) whose
target has an observed realization. Under daily 28-day direct forecasting
the same calendar day enters E once per origin that forecast it, so no
metric here may be described as "each day evaluated once". Targets past
the last observed demand day (d_1941) are generated for planning but are
excluded from E.

Zero-denominator rule (no epsilon anywhere): the affected aggregate is
undefined (NaN) and the count of undefined units is reported alongside.
Pooled metrics (one ratio over all of E) and macro averages (mean of
per-unit metrics over defined units) are different numbers and are never
mixed in reporting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> np.ndarray:
    """Elementwise pinball loss rho_q(y - yhat) = max(q*u, (q-1)*u)."""
    u = np.asarray(y, dtype=np.float64) - np.asarray(yhat, dtype=np.float64)
    return np.maximum(q * u, (q - 1.0) * u)


def wql(y: np.ndarray, preds: np.ndarray, quantiles: Sequence[float]) -> float:
    """Weighted quantile loss, AWS Forecast convention (coefficient 2):

        WQL = 2 * sum_q sum_E rho_q(y - yhat_q) / (|Q| * sum_E |y|)

    preds has shape (len(y), len(quantiles)), columns aligned to
    `quantiles`. Identity used as a unit test: for Q = {0.5}, WQL equals
    the WAPE of the median forecast. Returns NaN when sum|y| == 0.
    """
    y = np.asarray(y, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2 or preds.shape != (len(y), len(quantiles)):
        raise ValueError("preds must have shape (len(y), len(quantiles))")
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    num = sum(pinball(y, preds[:, j], q).sum()
              for j, q in enumerate(quantiles))
    return float(2.0 * num / (len(quantiles) * denom))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """sum|y - yhat| / sum|y|; NaN when sum|y| == 0. Point forecast for
    reporting is the tau = 0.5 quantile (ST3-03)."""
    y = np.asarray(y, dtype=np.float64)
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y - np.asarray(yhat, dtype=np.float64)).sum() / denom)


def nonzero_mape(y: np.ndarray, yhat: np.ndarray) -> tuple[float, int]:
    """MAPE over rows with y != 0 (reference only; excluding zero-demand
    days is a selection and must be labelled as such in reports). Returns
    (mape, n_rows_used); NaN if no nonzero rows."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = y != 0
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask]))), n


def monotone_rearrange(preds: np.ndarray) -> tuple[np.ndarray, float]:
    """Fixed post-processing for independently trained quantile models
    (ST3-03): sort each row across quantile columns (rearrangement).
    Returns (sorted copy, crossing_rate) where crossing_rate is the share
    of rows with any strictly decreasing adjacent pair BEFORE sorting.
    Applied uniformly to every evaluation and interval computation; the
    pre-correction crossing rate is always reported."""
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError("preds must be 2-D (rows x quantiles)")
    crossing = (np.diff(preds, axis=1) < 0).any(axis=1)
    return np.sort(preds, axis=1), float(crossing.mean())


def seasonal_naive_mae(train_values: np.ndarray, m: int = 7) -> float:
    """MASE denominator for one series: mean |y_t - y_{t-m}| over the
    series' TRAINING window of the relevant fold. Returns NaN when fewer
    than m + 1 observations."""
    v = np.asarray(train_values, dtype=np.float64)
    if v.size <= m:
        return float("nan")
    return float(np.abs(v[m:] - v[:-m]).mean())


def mase_table(
    y: np.ndarray,
    yhat: np.ndarray,
    series_ids: Sequence[str],
    denominators: dict,
) -> pd.DataFrame:
    """Per-series MASE_i = MAE_i / denominators[series_id].

    A series with a missing, NaN or ZERO denominator is marked
    defined = False with mase = NaN (never epsilon-patched); callers
    report (~defined).sum(). Macro aggregation = mean over defined rows.
    """
    df = pd.DataFrame({
        "series_id": list(series_ids),
        "ae": np.abs(np.asarray(y, dtype=np.float64)
                     - np.asarray(yhat, dtype=np.float64)),
    })
    out = df.groupby("series_id", sort=True)["ae"].agg(
        mae="mean", n_rows="size").reset_index()
    denom = out["series_id"].map(denominators).astype(float)
    defined = denom.notna() & (denom != 0)
    out["mase"] = np.where(defined, out["mae"] / denom, np.nan)
    out["defined"] = defined
    return out[["series_id", "mase", "mae", "n_rows", "defined"]]


def macro_mean(values: np.ndarray) -> tuple[float, int]:
    """Mean over defined (non-NaN) units; returns (mean, n_undefined)."""
    v = np.asarray(values, dtype=np.float64)
    undef = int(np.isnan(v).sum())
    if undef == len(v):
        return float("nan"), undef
    return float(np.nanmean(v)), undef
