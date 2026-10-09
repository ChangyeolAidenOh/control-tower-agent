"""Chronos-2 support helpers (Stage 3-C, ST3-05). No chronos import here,
so these are unit-testable without the package or model download.

Covariate rule (reviewed against the official implementation): columns
present in BOTH the historical df and future_df are treated as
known-future covariates. Both frames are therefore built from ONE
calendar source and split at the origin, which also guarantees identical
encodings on either side of the boundary. `target` never appears in
future_df; future_df spans exactly the 28 target days.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import pandas as pd

M5_D1_DATE = dt.date(2011, 1, 29)  # period_idx 1

COVARIATE_COLUMNS = ("wday", "month", "snap", "is_event", "event_type")


def idx_to_date(period_idx: np.ndarray) -> np.ndarray:
    base = np.datetime64(M5_D1_DATE) - np.timedelta64(1, "D")
    return base + np.asarray(period_idx, dtype=np.int64).astype("timedelta64[D]")


def date_to_idx(ts: pd.Series) -> np.ndarray:
    base = pd.Timestamp(M5_D1_DATE)
    return (pd.to_datetime(ts) - base).dt.days.to_numpy() + 1


def find_quantile_column(columns: Sequence[str], q: float) -> str:
    """Chronos-2 predict_df names quantile columns by their level; the
    exact string format is version-dependent, so resolve defensively."""
    candidates = [str(q), f"{q:.2f}", f"{q:.1f}", f"{q:g}", f"q{q}", f"q{q:.2f}"]
    for cand in candidates:
        if cand in columns:
            return cand
    raise KeyError(
        f"no column for quantile {q}; tried {candidates}; "
        f"available: {sorted(map(str, columns))}"
    )


def split_frames(panel: pd.DataFrame, origin_idx: int,
                 horizon: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """panel: long df with series_id, period_idx, target, covariates for
    period 1..origin+horizon (one calendar source). Returns (df, future_df)
    with timestamps; exposure guard asserts the split is exact."""
    cols = ["series_id", "period_idx", "target", *COVARIATE_COLUMNS]
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    hist = panel[panel["period_idx"] <= origin_idx][cols].copy()
    fut = panel[
        (panel["period_idx"] > origin_idx)
        & (panel["period_idx"] <= origin_idx + horizon)
    ][["series_id", "period_idx", *COVARIATE_COLUMNS]].copy()
    if hist["period_idx"].max() != origin_idx:
        raise ValueError("historical frame does not end at the origin")
    n_series = panel["series_id"].nunique()
    if len(fut) != n_series * horizon:
        raise ValueError("future frame is not exactly horizon days per series")
    if "target" in fut.columns:
        raise ValueError("target must not appear in future_df")
    for df in (hist, fut):
        df["timestamp"] = idx_to_date(df["period_idx"].to_numpy())
    return hist.drop(columns="period_idx"), fut.drop(columns="period_idx")


def extract_quantile_matrix(
    out: pd.DataFrame, origin_idx: int, quantiles: Sequence[float],
    id_column: str = "series_id", timestamp_column: str = "timestamp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From predict_df output, return (series_id, horizon, raw matrix)
    sorted by (series, horizon). `predictions` is intentionally ignored:
    it is the 0.5 quantile under an internal name, not a mean."""
    qcols = [find_quantile_column(out.columns, q) for q in quantiles]
    df = out.copy()
    df["horizon"] = date_to_idx(df[timestamp_column]) - origin_idx
    if df["horizon"].min() < 1:
        raise ValueError("prediction timestamps overlap the history")
    df = df.sort_values([id_column, "horizon"])
    raw = df[qcols].to_numpy(dtype=np.float64)
    return (df[id_column].to_numpy(), df["horizon"].to_numpy(dtype=np.int64),
            raw)
