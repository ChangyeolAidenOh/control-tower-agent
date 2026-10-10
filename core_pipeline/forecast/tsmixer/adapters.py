"""TSMixer-Ext adapters (ST4-01): information-set layer -> tensors-to-be.

This module is torch-free on purpose. The layer boundary is numpy: the
export step (DB phase, ST3-06) produces per-fold PanelInputs; everything
here runs identically on the Mac (inference) and inside the Colab training
script. Embedding of integer codes happens in the model, not here.

Window definition (D3): past window is [t - 55, t], origin inclusive; the
future block is [t + 1, t + 28]. An origin t is valid for series i iff
  - t - first_active_idx[i] + 1 >= past_length (56 active days of history),
  - the panel grid covers [t - 55, t + 28],
  - for training additionally t + horizon <= train_end (ST3-01 D3), and the
    known calendar covers t + 28 (guaranteed upstream by
    MissingKnownCalendarError at the information-set layer).
"""

from dataclasses import dataclass

import numpy as np

from core_pipeline.forecast.tsmixer.config import (
    FUTURE_NUM_CHANNELS,
    PAST_NUM_CHANNELS,
    TSMixerConfig,
)


@dataclass
class PanelInputs:
    """Per-fold, model-ready panel on a contiguous period grid.

    All (S, T) matrices share the same grid; column 0 is period_idx
    idx_start. sales_scaled and price_rel are already divided by the
    per-series train statistics (SeriesMeanScaler and the analogous price
    mean); scale carries s_i for inverse transform downstream.
    """

    idx_start: int
    series_ids: np.ndarray        # (S,) object/str, for meta only
    sales_scaled: np.ndarray      # (S, T) float
    price_rel: np.ndarray         # (S, T) float
    is_active: np.ndarray         # (S, T) 0/1
    snap: np.ndarray              # (S, T) 0/1, state broadcast upstream
    is_event: np.ndarray          # (S, T) 0/1
    event_code: np.ndarray        # (S, T) int, 0 = no event
    wday: np.ndarray              # (T,) int 1..7
    month: np.ndarray             # (T,) int 1..12
    first_active_idx: np.ndarray  # (S,) int period_idx
    scale: np.ndarray             # (S,) float s_i

    def __post_init__(self) -> None:
        s, t = self.sales_scaled.shape
        for name in ("price_rel", "is_active", "snap", "is_event", "event_code"):
            if getattr(self, name).shape != (s, t):
                raise ValueError(f"{name} shape differs from sales_scaled")
        if self.wday.shape != (t,) or self.month.shape != (t,):
            raise ValueError("calendar vectors must have length T")
        if self.first_active_idx.shape != (s,) or self.scale.shape != (s,):
            raise ValueError("per-series vectors must have length S")

    @property
    def n_series(self) -> int:
        return self.sales_scaled.shape[0]

    @property
    def n_periods(self) -> int:
        return self.sales_scaled.shape[1]

    @property
    def idx_end(self) -> int:
        return self.idx_start + self.n_periods - 1


@dataclass
class StaticInputs:
    names: tuple[str, ...]
    codes: np.ndarray             # (S, n_cats) int
    cardinalities: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.codes.shape[1] != len(self.names) or len(self.names) != len(self.cardinalities):
            raise ValueError("static names/codes/cardinalities disagree")
        for j, card in enumerate(self.cardinalities):
            col = self.codes[:, j]
            if col.min() < 0 or col.max() >= card:
                raise ValueError(f"static codes out of range for {self.names[j]}")


@dataclass
class WindowBatch:
    """One gathered batch of N samples. Scaled space throughout."""

    x_num: np.ndarray    # (N, L, len(PAST_NUM_CHANNELS)) float32
    x_event: np.ndarray  # (N, L) int64
    z_num: np.ndarray    # (N, H, len(FUTURE_NUM_CHANNELS)) float32
    z_event: np.ndarray  # (N, H) int64
    s_codes: np.ndarray  # (N, n_cats) int64
    scale: np.ndarray    # (N,) float32
    series_rows: np.ndarray  # (N,) int
    origin_idx: np.ndarray   # (N,) int period_idx
    y_scaled: np.ndarray | None = None  # (N, H) float32, training targets


def _calendar_channels(wday: np.ndarray, month: np.ndarray) -> np.ndarray:
    """(T,) wday/month -> (T, 9): month_sin, month_cos, wday one-hot(7)."""
    t = wday.shape[0]
    out = np.zeros((t, 9), dtype=np.float32)
    ang = 2.0 * np.pi * (month.astype(np.float32) - 1.0) / 12.0
    out[:, 0] = np.sin(ang)
    out[:, 1] = np.cos(ang)
    out[np.arange(t), 1 + wday.astype(np.int64)] = 1.0  # cols 2..8 = wday 1..7
    return out


def training_origins(
    panel: PanelInputs, train_end_idx: int, cfg: TSMixerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """All valid (series_row, origin) pairs for training on one fold."""
    lo = panel.idx_start + cfg.past_length - 1
    hi = min(train_end_idx - cfg.horizon, panel.idx_end - cfg.horizon)
    rows_out: list[np.ndarray] = []
    orig_out: list[np.ndarray] = []
    for i in range(panel.n_series):
        start = max(lo, int(panel.first_active_idx[i]) + cfg.past_length - 1)
        if start > hi:
            continue
        origins = np.arange(start, hi + 1)
        rows_out.append(np.full(origins.shape, i, dtype=np.int64))
        orig_out.append(origins)
    if not rows_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(rows_out), np.concatenate(orig_out)


def _gather(mat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return mat[rows[:, None], cols]


def build_batch(
    panel: PanelInputs,
    static: StaticInputs,
    series_rows: np.ndarray,
    origin_idx: np.ndarray,
    cfg: TSMixerConfig,
    with_targets: bool = False,
) -> WindowBatch:
    rows = np.asarray(series_rows, dtype=np.int64)
    origins = np.asarray(origin_idx, dtype=np.int64)
    j = origins - panel.idx_start
    if (j - (cfg.past_length - 1) < 0).any():
        raise ValueError("origin window underruns the panel grid")
    if (j + cfg.horizon > panel.n_periods - 1).any():
        raise ValueError("origin horizon overruns the panel grid")
    hist_ok = origins - panel.first_active_idx[rows] + 1 >= cfg.past_length
    if not hist_ok.all():
        raise ValueError("origin lacks 56 active-history days (ST4-01 D3)")

    past_cols = j[:, None] + np.arange(-(cfg.past_length - 1), 1)[None, :]
    fut_cols = j[:, None] + np.arange(1, cfg.horizon + 1)[None, :]
    cal = _calendar_channels(panel.wday, panel.month)  # (T, 9)

    n = rows.shape[0]
    x_num = np.empty((n, cfg.past_length, len(PAST_NUM_CHANNELS)), dtype=np.float32)
    x_num[:, :, 0] = _gather(panel.sales_scaled, rows, past_cols)
    x_num[:, :, 1] = np.nan_to_num(_gather(panel.price_rel, rows, past_cols), nan=0.0)
    x_num[:, :, 2] = _gather(panel.is_active, rows, past_cols)
    x_num[:, :, 3] = _gather(panel.snap, rows, past_cols)
    x_num[:, :, 4] = _gather(panel.is_event, rows, past_cols)
    x_num[:, :, 5:14] = cal[past_cols]

    z_num = np.empty((n, cfg.horizon, len(FUTURE_NUM_CHANNELS)), dtype=np.float32)
    z_num[:, :, 0] = _gather(panel.snap, rows, fut_cols)
    z_num[:, :, 1] = _gather(panel.is_event, rows, fut_cols)
    z_num[:, :, 2:11] = cal[fut_cols]

    batch = WindowBatch(
        x_num=x_num,
        x_event=_gather(panel.event_code, rows, past_cols).astype(np.int64),
        z_num=z_num,
        z_event=_gather(panel.event_code, rows, fut_cols).astype(np.int64),
        s_codes=static.codes[rows].astype(np.int64),
        scale=panel.scale[rows].astype(np.float32),
        series_rows=rows,
        origin_idx=origins,
        y_scaled=(
            _gather(panel.sales_scaled, rows, fut_cols).astype(np.float32)
            if with_targets else None
        ),
    )
    return batch
