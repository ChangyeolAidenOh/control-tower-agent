"""A-PTO (ST6-11) tabular adapter: long per-horizon frames -> wide
multi-output frames for the CatBoost MultiRMSE predictor.

Contract (decision ST6-11):
  - Reuses the Stage 3 shared information set via FoldDatasetBuilder
    frames built with include_series_id=False (series ID excluded).
  - One wide row per (series, origin) with ALL 28 horizons present;
    origins with an incomplete horizon set are dropped, which enforces
    origin + 28 <= target_hi mechanically (checking origin <= train_end
    alone is insufficient).
  - Origin-level features are taken from the h=1 rows (they are
    horizon-invariant by construction except swd_mean_4 / tgt_*, which
    are horizon-specific and therefore replaced by the deterministic
    28-day known-calendar summary: snap_days_28, event_days_28,
    start_wday -- the lt_cum pattern).
  - Target scaling: per-series divisor s_i = max(1, mean demand over
    [first_active_idx, cutoff]); predictions are inverse-transformed by
    the same s_i. Features are not scaled (tree splits are
    scale-invariant; only the pooled RMSE objective needs it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core_pipeline.forecast.dataset import TrainFrame

N_HORIZONS = 28

WIDE_AGG_COLUMNS = ("snap_days_28", "event_days_28", "start_wday")
WIDE_CAT_COLUMNS = ("store_id", "dept_id", "cat_id", "state_id")


def wide_numeric_columns(cfg) -> tuple[str, ...]:
    origin = tuple(c for c in cfg.origin_numeric_columns
                   if c != "swd_mean_4")
    return origin + WIDE_AGG_COLUMNS


def wide_feature_columns(cfg) -> tuple[str, ...]:
    return wide_numeric_columns(cfg) + WIDE_CAT_COLUMNS


@dataclass
class APTOFrame:
    """Wide multi-output frame: one row per (series, origin)."""

    X: pd.DataFrame
    y: Optional[np.ndarray]       # (n, 28) raw demand; None for inference
    meta: pd.DataFrame            # series_id, origin_idx (row-aligned)
    categorical: tuple[str, ...]
    n_dropped_incomplete: int
    spec_hash: str
    encoder_hash: str


def to_wide(frame: TrainFrame, cfg) -> APTOFrame:
    """Convert a per-horizon long frame (training, tuning or inference)
    into the wide A-PTO frame. Only (series, origin) groups carrying all
    28 horizons survive."""
    if frame.target_mode != "per_horizon":
        raise ValueError(f"expected per_horizon frame, got {frame.target_mode}")
    for col in ("series_id",):
        if col in frame.X.columns:
            raise ValueError(
                "A-PTO frames must be built with include_series_id=False "
                "(ST6-11 (1)); found series_id feature column")

    df = frame.X.reset_index(drop=True).copy()
    meta = frame.meta.reset_index(drop=True)
    df["_series"] = meta["series_id"].to_numpy()
    df["_origin"] = meta["origin_idx"].to_numpy()
    df["_h"] = meta["horizon"].to_numpy()
    has_y = frame.y is not None and len(frame.y) == len(df)
    if has_y:
        df["_y"] = np.asarray(frame.y, dtype=np.float64)

    counts = df.groupby(["_series", "_origin"], sort=False)["_h"].transform("size")
    complete = counts == N_HORIZONS
    n_dropped = int((~complete).sum() // max(1, 1))  # row count of dropped rows
    dfc = df.loc[complete]
    if dfc.empty:
        raise ValueError("no (series, origin) group carries all 28 horizons")
    dup = dfc.groupby(["_series", "_origin"])["_h"].nunique()
    if (dup != N_HORIZONS).any():
        raise ValueError("duplicate horizons inside a (series, origin) group")

    keys = ["_series", "_origin"]
    agg = dfc.groupby(keys, sort=True).agg(
        snap_days_28=("tgt_snap", "sum"),
        event_days_28=("tgt_is_event", "sum"),
    ).reset_index()

    base = (dfc[dfc["_h"] == 1]
            .sort_values(keys)
            .reset_index(drop=True))
    agg = agg.sort_values(keys).reset_index(drop=True)
    if len(base) != len(agg) or not (
        (base["_series"].to_numpy() == agg["_series"].to_numpy()).all()
        and (base["_origin"].to_numpy() == agg["_origin"].to_numpy()).all()
    ):
        raise ValueError("h=1 base rows misaligned with aggregates")

    origin_cols = [c for c in cfg.origin_numeric_columns if c != "swd_mean_4"]
    X = base[origin_cols].copy()
    X["snap_days_28"] = agg["snap_days_28"].to_numpy(np.float32)
    X["event_days_28"] = agg["event_days_28"].to_numpy(np.float32)
    X["start_wday"] = base["tgt_wday"].to_numpy(np.float32)
    for c in WIDE_CAT_COLUMNS:
        X[c] = base[c].to_numpy()
    X = X[list(wide_feature_columns(cfg))].reset_index(drop=True)

    y = None
    if has_y:
        piv = dfc.pivot_table(index=keys, columns="_h", values="_y",
                              aggfunc="first").sort_index()
        piv = piv.reindex(columns=range(1, N_HORIZONS + 1))
        if piv.isna().any().any():
            raise ValueError("missing horizon targets after pivot")
        if len(piv) != len(base):
            raise ValueError("target pivot row count mismatch")
        y = piv.to_numpy(dtype=np.float64)

    out_meta = pd.DataFrame({
        "series_id": base["_series"].to_numpy(),
        "origin_idx": base["_origin"].to_numpy(np.int64),
    })
    return APTOFrame(
        X=X, y=y, meta=out_meta, categorical=WIDE_CAT_COLUMNS,
        n_dropped_incomplete=n_dropped,
        spec_hash=frame.spec_hash, encoder_hash=frame.encoder_hash,
    )


def series_scales(source, cutoff: int) -> dict[str, float]:
    """s_i = max(1, mean demand over [first_active_idx, cutoff]) per
    series, from mart reads capped at `cutoff` (fold-dependent; tuning
    uses d_1437). Deterministic, no randomness."""
    static = source.load_static().sort_values("series_id")
    dp = source.load_demand_price(max_period_idx=cutoff)
    first = dict(zip(static["series_id"], static["first_active_idx"]))
    out: dict[str, float] = {}
    for sid, grp in dp.groupby("series_id", sort=True):
        fa = int(first[str(sid)])
        vals = grp.loc[grp["period_idx"] >= fa, "sales"].to_numpy(np.float64)
        mu = float(vals.mean()) if vals.size else 0.0
        out[str(sid)] = max(1.0, mu)
    missing = set(map(str, static["series_id"])) - set(out)
    if missing:
        raise ValueError(f"series without demand rows: {sorted(missing)[:5]}")
    return out


def scale_targets(y: np.ndarray, meta: pd.DataFrame,
                  scales: dict[str, float]) -> np.ndarray:
    s = meta["series_id"].map(scales).to_numpy(np.float64)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("invalid series scale")
    return y / s[:, None]


def unscale_predictions(yhat: np.ndarray, meta: pd.DataFrame,
                        scales: dict[str, float]) -> np.ndarray:
    s = meta["series_id"].map(scales).to_numpy(np.float64)
    return yhat * s[:, None]