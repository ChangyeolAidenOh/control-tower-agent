"""Frames -> PanelInputs/StaticInputs (torch-free). ONE implementation
shared by Colab training (bundle frames) and local inference (DB frames),
so encoding, scaling and the calendar broadcast cannot diverge between the
two sides. Mirrors the dataset-layer calendar convention: wday, month,
event_type, is_event are global per period (deduplicated), snap is
state-level and broadcast to series rows.

Unknown event_type values hard-fail (the encoder vocabulary is built from
the full published calendar, so an unknown value means a contract break,
not data to be coerced).
"""

import numpy as np
import pandas as pd

from core_pipeline.forecast.tsmixer.adapters import PanelInputs, StaticInputs
from core_pipeline.forecast.tsmixer.config import STATIC_CATEGORICALS


def encode_values(values, mapping: dict, what: str) -> np.ndarray:
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        key = str(v)
        if key not in mapping:
            raise ValueError(f"unknown {what} value: {key!r}")
        out[i] = mapping[key]
    return out


def build_static_inputs(static_df: pd.DataFrame, encoder: dict) -> StaticInputs:
    codes = np.stack([
        encode_values(static_df[c].tolist(), encoder["static"][c], c)
        for c in STATIC_CATEGORICALS], axis=1)
    cards = tuple(int(encoder["cardinalities"][c]) for c in STATIC_CATEGORICALS)
    return StaticInputs(STATIC_CATEGORICALS, codes, cards)


def build_panel(
    dp: pd.DataFrame,
    calendar: pd.DataFrame,
    static_df: pd.DataFrame,
    encoder: dict,
    scale: np.ndarray,
    price_mean: np.ndarray,
    idx_start: int,
    idx_end: int,
) -> PanelInputs:
    """dp: series_id, period_idx, sales, price_ffill, is_active (rows past
    the observation cap simply absent -- grid cells fill as inactive zero
    and are reachable only through future calendar channels, never as past
    demand). calendar must cover [idx_start, idx_end] completely."""
    series_order = [str(s) for s in static_df["series_id"]]
    idx_full = np.arange(idx_start, idx_end + 1)

    def _wide(col: str) -> np.ndarray:
        w = dp.pivot(index="series_id", columns="period_idx", values=col)
        w = w.reindex(index=series_order, columns=idx_full)
        return w.to_numpy(dtype=np.float64)

    sales = np.nan_to_num(_wide("sales"), nan=0.0)
    price = _wide("price_ffill")  # NaN kept: pre-listing / past-cap cells
    active = np.nan_to_num(
        _wide("is_active") if dp["is_active"].dtype != bool
        else _wide("is_active"), nan=0.0)

    glob = (calendar.drop_duplicates("period_idx")
            .sort_values("period_idx").reset_index(drop=True))
    if not np.array_equal(glob["period_idx"].to_numpy(dtype=np.int64), idx_full):
        raise ValueError("calendar range incomplete for the panel grid")
    snap_wide = calendar.pivot(index="state_id", columns="period_idx",
                               values="snap").reindex(columns=idx_full)
    state_rows = {s: i for i, s in enumerate(snap_wide.index)}
    srow = np.array([state_rows[str(s)] for s in static_df["state_id"]],
                    dtype=np.int64)
    event_global = encode_values(
        glob["event_type"].tolist(), encoder["event_type"], "event_type")
    n_series, n_periods = len(series_order), len(idx_full)

    scale = np.asarray(scale, dtype=np.float64)
    price_mean = np.asarray(price_mean, dtype=np.float64)
    if scale.shape != (n_series,) or price_mean.shape != (n_series,):
        raise ValueError("scale/price_mean length differs from series count")

    return PanelInputs(
        idx_start=int(idx_start),
        series_ids=np.array(series_order),
        sales_scaled=sales / scale[:, None],
        price_rel=price / price_mean[:, None],
        is_active=active,
        snap=snap_wide.to_numpy(dtype=np.float64)[srow],
        is_event=np.tile(glob["is_event"].to_numpy(dtype=np.float64),
                         (n_series, 1)),
        event_code=np.tile(event_global, (n_series, 1)),
        wday=glob["wday"].to_numpy(dtype=np.int64),
        month=glob["month"].to_numpy(dtype=np.int64),
        first_active_idx=pd.to_numeric(
            static_df["first_active_idx"]).to_numpy(dtype=np.int64),
        scale=scale,
    )
