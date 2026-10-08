"""Stage 3-A: fold-wise dataset contract for global forecasting models.

Two-layer contract (3-A verdict, 2026-10-09):

  Shared information-set layer (all models):
    - PanelSource: the only mart access path. Demand/price reads take a hard
      max_period_idx cap; calendar reads are the only reads allowed past a
      cap (D1: published schedule) and touch no demand/price column.
    - DemandHistory: the only demand source for inference features (D5).
      Future access is blocked at the interface (observed_through guard),
      not by feature-function discipline.
    - DatasetConfig window/horizon/tuning constants.

  Model-specific layer:
    - This module's 28-column tabular schema is the LightGBM contract only.
    - TSMixer-Ext (Stage 4) consumes the same shared layer through its own
      adapter (past sequence / future covariate sequence / static inputs);
      A1, A2 and B share that adapter and backbone. The engineered columns
      here are NOT part of the cross-model contract.

Information-set rules (handoff v1.4 SS2-7/SS4, preregistration v1.2 SS2):
  - Training rows satisfy origin + h <= train_end (D3); tuning rows come
    only from tuning_split() (single pre-fold-1 time split, D4). The split
    has no fold argument; the operational re-tuning ban is enforced by the
    3-B runner writing tuned params once to outputs/stage3_lgbm_tuning.json
    and fold runners loading only that file.
  - Retraining only at folds 1/4/7/10; training_frame rejects other folds.
  - Inference always produces the full 28-horizon plan (preregistered
    daily 28-day rolling decision). Demand/price end at max_demand_idx
    (mart: d_1941); the known calendar extends to max_calendar_idx
    (mart.known_future_calendar, d_1969). If the known calendar cannot
    cover origin + 28, MissingKnownCalendarError is raised -- horizons are
    never silently clipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

RETRAIN_FOLDS: tuple[int, ...] = (1, 4, 7, 10)

TARGET_PER_HORIZON = "per_horizon"
TARGET_LT_CUM = "lt_cum"


class FutureAccessError(ValueError):
    """Raised when code asks DemandHistory for observations past current_idx."""


class MissingKnownCalendarError(ValueError):
    """Raised when the known calendar cannot cover origin + max(horizons).
    Horizons are never silently clipped (28-day rolling plan contract)."""


def retrain_fold_for(fold: int) -> int:
    """Map an evaluation fold to the retrain fold whose model it uses."""
    if not 1 <= fold <= 12:
        raise ValueError(f"fold must be in 1..12, got {fold}")
    return max(f for f in RETRAIN_FOLDS if f <= fold)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Shared information-set constants + the LightGBM tabular schema.

    spec_hash() is recorded next to every trained model and output JSON so
    feature drift is detectable. TSMixer-Ext reuses the constants (windows,
    horizons, tuning split, min history), not the engineered columns.
    """

    lags: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
    roll_mean_windows: tuple[int, ...] = (7, 14, 28, 56)
    roll_std_windows: tuple[int, ...] = (7, 28)  # population std (ddof=0)
    same_wday_count: int = 4  # == occurrences of any weekday in 28 days
    zero_share_window: int = 28
    price_rel_window: int = 28
    horizons: tuple[int, ...] = tuple(range(1, 29))
    # Order at D arrives D+3 morning (simulator_spec v1 SS2): lead-time
    # cumulative target = sum of demand over t+1..t+3.
    lead_time_cum_horizon: int = 3
    # D4: single pre-fold-1 split; valid == mart.windows validation window.
    tune_train_target_end: int = 1437
    tune_valid_target_start: int = 1438
    tune_valid_target_end: int = 1605
    # D3: minimum days since the series' first active day at the origin.
    min_history_days: int = 56
    include_series_id: bool = True  # D6

    categorical_columns: tuple[str, ...] = (
        "series_id", "store_id", "dept_id", "cat_id", "state_id",
        "tgt_event_type",
    )

    @property
    def origin_numeric_columns(self) -> tuple[str, ...]:
        cols: list[str] = [f"sales_lag_{k}" for k in self.lags]
        cols += [f"roll_mean_{w}" for w in self.roll_mean_windows]
        cols += [f"roll_std_{w}" for w in self.roll_std_windows]
        cols += ["swd_mean_4", "zero_share_28", "price_now", "price_rel_28"]
        return tuple(cols)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return self.origin_numeric_columns + (
            "h", "tgt_wday", "tgt_month", "tgt_snap", "tgt_is_event",
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self._cats()

    @property
    def lt_cum_numeric_columns(self) -> tuple[str, ...]:
        # swd_mean_4 is horizon-specific; the cumulative target spans
        # t+1..t+K, so it is replaced by aggregated schedule features.
        origin = tuple(c for c in self.origin_numeric_columns if c != "swd_mean_4")
        return origin + ("cum_snap_days", "cum_event_days", "start_wday")

    @property
    def lt_cum_feature_columns(self) -> tuple[str, ...]:
        cats = tuple(c for c in self._cats() if c != "tgt_event_type")
        return self.lt_cum_numeric_columns + cats

    def _cats(self) -> tuple[str, ...]:
        cats = self.categorical_columns
        if not self.include_series_id:
            cats = tuple(c for c in cats if c != "series_id")
        return cats

    def spec_hash(self) -> str:
        payload = json.dumps(
            {k: v for k, v in sorted(self.__dict__.items())},
            sort_keys=True, default=list,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Demand history (shared layer, D5)
# ---------------------------------------------------------------------------


class DemandHistory:
    """Demand observations with interface-level future blocking.

    values may physically contain a preloaded scenario path past
    current_idx (censoring/shock from demand_paths.build_fold_paths), but
    observed_through() refuses to expose anything past current_idx. All
    state transitions return new objects; underlying arrays are shared
    read-only (paired-path safety).
    """

    def __init__(
        self,
        series_ids: Sequence[str],
        idx_start: int,
        values: np.ndarray,
        current_idx: int,
    ):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(series_ids):
            raise ValueError("values must be (n_series, n_days)")
        last_idx = idx_start + values.shape[1] - 1
        if not idx_start - 1 <= current_idx <= last_idx:
            raise ValueError(
                f"current_idx {current_idx} outside [{idx_start - 1}, {last_idx}]"
            )
        values = values.copy()
        values.setflags(write=False)
        self._series_ids = tuple(series_ids)
        self._idx_start = int(idx_start)
        self._values = values
        self._current_idx = int(current_idx)

    @property
    def series_ids(self) -> tuple[str, ...]:
        return self._series_ids

    @property
    def idx_start(self) -> int:
        return self._idx_start

    @property
    def current_idx(self) -> int:
        return self._current_idx

    @property
    def preloaded_end_idx(self) -> int:
        return self._idx_start + self._values.shape[1] - 1

    def observed_through(self, origin_idx: int) -> np.ndarray:
        """Observations for period_idx in [idx_start, origin_idx], shape
        (n_series, origin_idx - idx_start + 1). The only demand accessor."""
        if origin_idx > self._current_idx:
            raise FutureAccessError(
                f"origin {origin_idx} > current_idx {self._current_idx}: "
                "future observations are unavailable"
            )
        if origin_idx < self._idx_start:
            raise ValueError(f"origin {origin_idx} < idx_start {self._idx_start}")
        j = origin_idx - self._idx_start
        out = self._values[:, : j + 1]
        out.setflags(write=False)
        return out

    def advance(self) -> "DemandHistory":
        """Reveal the next preloaded day (scenario replay). New object."""
        if self._current_idx >= self.preloaded_end_idx:
            raise ValueError("no preloaded day to advance into; use appended()")
        clone = object.__new__(DemandHistory)
        clone._series_ids = self._series_ids
        clone._idx_start = self._idx_start
        clone._values = self._values
        clone._current_idx = self._current_idx + 1
        return clone

    def appended(self, day_values: np.ndarray) -> "DemandHistory":
        """Append one newly observed day past the preloaded range."""
        if self._current_idx != self.preloaded_end_idx:
            raise ValueError("cannot append while preloaded days remain")
        day = np.asarray(day_values, dtype=np.float32).reshape(-1, 1)
        if day.shape[0] != len(self._series_ids):
            raise ValueError("day_values length mismatch")
        return DemandHistory(
            self._series_ids,
            self._idx_start,
            np.concatenate([self._values, day], axis=1),
            self._current_idx + 1,
        )

    @classmethod
    def from_source(
        cls, source: "PanelSource", max_period_idx: int
    ) -> "DemandHistory":
        """p = 0 path: load mart sales with a hard cap (current_idx = cap)."""
        wide = source.load_demand_price(max_period_idx)
        mat, series_ids, idx_start = _pivot(wide, "sales")
        return cls(series_ids, idx_start, mat, current_idx=max_period_idx)


# ---------------------------------------------------------------------------
# Panel sources (shared layer)
# ---------------------------------------------------------------------------


class PanelSource:
    """All mart access for Stage 3. Contracts:

    load_static() -> one row per series: series_id, item_id, store_id,
        state_id, dept_id, cat_id, group_id, first_active_idx.
    load_demand_price(max_period_idx) -> long df: series_id, period_idx,
        sales, price_ffill, is_active, for period_idx <= cap. Single
        leakage chokepoint for demand and price.
    load_calendar(start_idx, end_idx) -> long df: state_id, period_idx,
        wday, month, snap, event_type, is_event. Known published schedule,
        demand-free columns only; mart implementation reads
        mart.known_future_calendar (covers d_1..d_1969). event_type applies
        the D1 rule (event_type_1 primary slot, else event_type_2, else
        'none'; is_event = either slot present).
    load_folds() -> mart.folds verbatim.
    load_windows() -> {'selection': (lo, hi), 'validation': (lo, hi)}.
    max_demand_idx() -> last period with demand/price (mart: 1941).
    max_calendar_idx() -> last period with known calendar (mart: 1969).
    """

    def load_static(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_demand_price(self, max_period_idx: int) -> pd.DataFrame:
        raise NotImplementedError

    def load_calendar(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        raise NotImplementedError

    def load_folds(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_windows(self) -> dict[str, tuple[int, int]]:
        raise NotImplementedError

    def max_demand_idx(self) -> int:
        raise NotImplementedError

    def max_calendar_idx(self) -> int:
        raise NotImplementedError


class MartPanelSource(PanelSource):
    """PostgreSQL-backed source (mart on port 5435). Pass a DB-API
    connection built via core_pipeline.data.db:
        db.load_dotenv(); cfg = db.load_config(); conn = db.connect(cfg)
    Calendar comes from mart.known_future_calendar (d_1..d_1969); demand
    and price from mart.sku_daily, always capped. Uses %s placeholders
    (psycopg-family paramstyle)."""

    def __init__(self, conn):
        self._conn = conn
        self._max_demand: Optional[int] = None
        self._max_calendar: Optional[int] = None

    def _df(self, query: str, params: tuple = ()) -> pd.DataFrame:
        cur = self._conn.cursor()
        cur.execute(query, params)
        columns = [c[0] for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)
        cur.close()
        return df

    def _scalar(self, query: str) -> int:
        cur = self._conn.cursor()
        cur.execute(query)
        value = cur.fetchone()[0]
        cur.close()
        if value is None:
            raise ValueError(f"empty result for: {query}")
        return int(value)

    def load_static(self) -> pd.DataFrame:
        df = self._df(
            """
            SELECT series_id, item_id, store_id, state_id, dept_id, cat_id,
                   group_id,
                   MIN(period_idx) FILTER (WHERE is_active) AS first_active_idx
            FROM mart.sku_daily
            GROUP BY series_id, item_id, store_id, state_id, dept_id,
                     cat_id, group_id
            ORDER BY series_id
            """
        )
        df["first_active_idx"] = pd.to_numeric(
            df["first_active_idx"]).astype(np.int64)
        return df

    def load_demand_price(self, max_period_idx: int) -> pd.DataFrame:
        df = self._df(
            """
            SELECT series_id, period_idx, sales, price_ffill, is_active
            FROM mart.sku_daily
            WHERE period_idx <= %s
            ORDER BY series_id, period_idx
            """,
            (int(max_period_idx),),
        )
        df["period_idx"] = pd.to_numeric(df["period_idx"]).astype(np.int64)
        df["sales"] = pd.to_numeric(df["sales"]).astype(np.float32)
        df["price_ffill"] = pd.to_numeric(
            df["price_ffill"], errors="coerce").astype(np.float32)
        df["is_active"] = df["is_active"].astype(bool)
        return df

    def load_calendar(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        df = self._df(
            """
            SELECT state_id, period_idx, wday, month, snap,
                   event_type, is_event
            FROM mart.known_future_calendar
            WHERE period_idx BETWEEN %s AND %s
            ORDER BY state_id, period_idx
            """,
            (int(start_idx), int(end_idx)),
        )
        for col in ("period_idx", "wday", "month"):
            df[col] = pd.to_numeric(df[col]).astype(np.int64)
        for col in ("snap", "is_event"):
            df[col] = pd.to_numeric(df[col]).astype(np.float32)
        df["event_type"] = df["event_type"].astype(str)
        return df

    def load_folds(self) -> pd.DataFrame:
        df = self._df(
            """
            SELECT fold, test_start_idx, test_end_idx, train_end_idx,
                   warmup_start_idx, is_retrain
            FROM mart.folds
            ORDER BY fold
            """
        )
        for col in ("fold", "test_start_idx", "test_end_idx",
                    "train_end_idx", "warmup_start_idx"):
            df[col] = pd.to_numeric(df[col]).astype(np.int64)
        return df

    def load_windows(self) -> dict[str, tuple[int, int]]:
        from core_pipeline.data import mart_reader
        return {
            name: tuple(int(v) for v in
                        mart_reader.fetch_window(self._conn, name))
            for name in ("selection", "validation")
        }

    def max_demand_idx(self) -> int:
        if self._max_demand is None:
            self._max_demand = self._scalar(
                "SELECT MAX(period_idx) FROM mart.sku_daily")
        return self._max_demand

    def max_calendar_idx(self) -> int:
        if self._max_calendar is None:
            self._max_calendar = self._scalar(
                "SELECT MAX(period_idx) FROM mart.known_future_calendar")
        return self._max_calendar


class ArrayPanelSource(PanelSource):
    """In-memory source for tests and small experiments. Enforces the same
    cap semantics as the mart-backed source."""

    def __init__(
        self,
        static: pd.DataFrame,
        demand_price: pd.DataFrame,
        calendar: pd.DataFrame,
        folds: Optional[pd.DataFrame] = None,
        windows: Optional[dict[str, tuple[int, int]]] = None,
    ):
        self._static = static.reset_index(drop=True)
        self._dp = demand_price
        self._cal = calendar
        self._folds = folds
        self._windows = windows or {}

    def load_static(self) -> pd.DataFrame:
        return self._static.copy()

    def load_demand_price(self, max_period_idx: int) -> pd.DataFrame:
        return self._dp[self._dp["period_idx"] <= max_period_idx].copy()

    def load_calendar(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        m = (self._cal["period_idx"] >= start_idx) & (self._cal["period_idx"] <= end_idx)
        return self._cal[m].copy()

    def load_folds(self) -> pd.DataFrame:
        if self._folds is None:
            raise ValueError("no folds configured")
        return self._folds.copy()

    def load_windows(self) -> dict[str, tuple[int, int]]:
        return dict(self._windows)

    def max_demand_idx(self) -> int:
        return int(self._dp["period_idx"].max())

    def max_calendar_idx(self) -> int:
        return int(self._cal["period_idx"].max())


# ---------------------------------------------------------------------------
# Categorical encoder (D1 detail: one dictionary for train/valid/inference)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalEncoder:
    mappings: dict  # col -> {value: int32 code}

    @classmethod
    def build(cls, static: pd.DataFrame, event_types: Sequence[str]) -> "CategoricalEncoder":
        maps: dict = {}
        for col in ("series_id", "store_id", "dept_id", "cat_id", "state_id"):
            vals = sorted(static[col].astype(str).unique())
            maps[col] = {v: i for i, v in enumerate(vals)}
        ev = sorted(set(str(e) for e in event_types) | {"none"})
        ev.remove("none")
        maps["tgt_event_type"] = {"none": 0, **{v: i + 1 for i, v in enumerate(ev)}}
        return cls(mappings=maps)

    def encode(self, col: str, values: Sequence[str]) -> np.ndarray:
        m = self.mappings[col]
        try:
            return np.fromiter((m[str(v)] for v in values), dtype=np.int32,
                               count=len(values))
        except KeyError as e:
            raise KeyError(f"unseen category {e} for column {col}") from e

    def encoder_hash(self) -> str:
        payload = json.dumps(self.mappings, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@dataclass
class TrainFrame:
    """Model-ready frame. meta carries (series_id, origin_idx, horizon,
    target_idx) per row so WQL/MASE and 3-E coverage can be computed without
    re-deriving alignment. For inference frames y is empty."""

    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    categorical: tuple[str, ...]
    spec_hash: str
    encoder_hash: str
    target_mode: str


# ---------------------------------------------------------------------------
# Internal panel container and feature math
# ---------------------------------------------------------------------------


@dataclass
class _Panel:
    series_ids: tuple[str, ...]
    idx_start: int
    sales: np.ndarray       # (S, T)
    price: np.ndarray       # (S, T), NaN where unlisted
    active: np.ndarray      # (S, T) bool
    snap: np.ndarray        # (S, T) float32 per-series state SNAP
    wday: np.ndarray        # (T,) int
    month: np.ndarray       # (T,) int
    event_code: np.ndarray  # (T,) int32 encoded tgt_event_type
    is_event: np.ndarray    # (T,) float32
    first_active: np.ndarray  # (S,) period_idx of first active day

    def col(self, period_idx: int) -> int:
        return period_idx - self.idx_start


def _pivot(long_df: pd.DataFrame, value_col: str):
    wide = long_df.pivot(index="series_id", columns="period_idx", values=value_col)
    wide = wide.sort_index().sort_index(axis=1)
    idx_full = np.arange(int(wide.columns.min()), int(wide.columns.max()) + 1)
    wide = wide.reindex(columns=idx_full)
    return (
        wide.to_numpy(dtype=np.float32),
        tuple(wide.index.astype(str)),
        int(idx_full[0]),
    )


def origin_features(
    sales_tail: np.ndarray,   # (S, >=56) ending at origin inclusive
    price_tail: np.ndarray,   # (S, >=28) ending at origin inclusive
    wday_tail: np.ndarray,    # (28,) weekdays of the last 28 days
    cfg: DatasetConfig,
) -> dict:
    """Origin-anchored features from trailing observations only (<= t).
    Single implementation shared by training and inference paths."""
    if sales_tail.shape[1] < cfg.min_history_days:
        raise ValueError("insufficient trailing history")
    out: dict = {}
    for k in cfg.lags:
        out[f"sales_lag_{k}"] = sales_tail[:, -k]
    for w in cfg.roll_mean_windows:
        out[f"roll_mean_{w}"] = sales_tail[:, -w:].mean(axis=1)
    for w in cfg.roll_std_windows:
        out[f"roll_std_{w}"] = sales_tail[:, -w:].std(axis=1)  # ddof=0
    t28 = sales_tail[:, -cfg.zero_share_window:]
    out["zero_share_28"] = (t28 == 0).mean(axis=1)
    p28 = price_tail[:, -cfg.price_rel_window:]
    price_now = p28[:, -1]
    denom = np.nanmean(p28, axis=1)
    rel = np.where(
        np.isfinite(denom) & (denom > 0), price_now / denom, 1.0
    ).astype(np.float32)
    out["price_now"] = price_now
    out["price_rel_28"] = rel
    # Last-4 same-weekday means: any weekday occurs exactly 4 times in the
    # trailing 28 days, so swd[w] = mean of those 4 observations.
    swd = np.empty((7, sales_tail.shape[0]), dtype=np.float32)
    s28 = sales_tail[:, -28:]
    for w in range(1, 8):
        mask = wday_tail == w
        swd[w - 1] = s28[:, mask].mean(axis=1) if mask.any() else np.nan
    out["_swd_by_wday"] = swd
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class FoldDatasetBuilder:
    """Builds training/tuning/inference frames under the fold information set.

    Leakage contract enforced in code:
      - training rows satisfy origin + h <= train_end(fold)
      - tuning rows satisfy target_idx inside the pre-fold-1 split ranges
      - inference demand comes only from DemandHistory.observed_through
      - every demand/price read is capped at the relevant origin/train_end
    """

    def __init__(self, cfg: DatasetConfig, source: PanelSource):
        self.cfg = cfg
        self.source = source
        self._check_windows()
        static = source.load_static()
        cal_all = source.load_calendar(1, source.max_calendar_idx())
        self.encoder = CategoricalEncoder.build(
            static, cal_all["event_type"].unique()
        )
        self._static = static.sort_values("series_id").reset_index(drop=True)
        # Static categorical codes are fixed per builder; cache once instead
        # of re-encoding on every (origin, horizon) emit.
        self._static_codes = {
            col: self.encoder.encode(col, self._static[col].tolist())
            for col in self.cfg._cats() if col != "tgt_event_type"
        }

    # -- public API --------------------------------------------------------

    def training_frame(self, fold: int, target_mode: str = TARGET_PER_HORIZON) -> TrainFrame:
        if fold not in RETRAIN_FOLDS:
            raise ValueError(
                f"fold {fold} is not a retrain fold {RETRAIN_FOLDS}; "
                f"use the model of fold {retrain_fold_for(fold)}"
            )
        folds = self.source.load_folds().set_index("fold")
        train_end = int(folds.loc[fold, "train_end_idx"])
        panel = self._load_panel(cap=train_end)
        return self._build_frame(panel, target_lo=1, target_hi=train_end,
                                 target_mode=target_mode)

    def tuning_split(self, target_mode: str = TARGET_PER_HORIZON) -> tuple[TrainFrame, TrainFrame]:
        """Single pre-fold-1 time split (D4). No fold argument by design;
        the operational re-tuning ban lives in the 3-B runner contract."""
        cfg = self.cfg
        panel = self._load_panel(cap=cfg.tune_valid_target_end)
        tr = self._build_frame(panel, 1, cfg.tune_train_target_end, target_mode)
        va = self._build_frame(panel, cfg.tune_valid_target_start,
                               cfg.tune_valid_target_end, target_mode)
        return tr, va

    def inference_frame(
        self,
        history: DemandHistory,
        origin_idx: int,
        horizons: Optional[Sequence[int]] = None,
    ) -> TrainFrame:
        """Features for every series at one origin, always over the full
        requested horizons (default h = 1..28; the preregistered 28-day
        rolling plan is never silently shortened). Demand comes exclusively
        from `history` (guarded); price from the source capped at the
        origin; target calendar from the known schedule. Raises
        MissingKnownCalendarError if the known calendar cannot cover
        origin + max(horizons)."""
        cfg = self.cfg
        hs = tuple(horizons) if horizons is not None else cfg.horizons
        need_end = origin_idx + max(hs)
        if need_end > self.source.max_calendar_idx():
            raise MissingKnownCalendarError(
                f"known calendar ends at {self.source.max_calendar_idx()}, "
                f"origin {origin_idx} needs {need_end}"
            )
        cap = need_end

        obs = history.observed_through(origin_idx)
        if obs.shape[1] < cfg.min_history_days:
            raise ValueError("history shorter than min_history_days")
        if tuple(history.series_ids) != tuple(self._static["series_id"]):
            raise ValueError("history series order must match static table")

        dp = self.source.load_demand_price(max_period_idx=origin_idx)
        price_mat, p_series, p_start = _pivot(dp, "price_ffill")
        if p_series != tuple(self._static["series_id"]):
            raise ValueError("price series order mismatch")
        price_tail = price_mat[:, -cfg.price_rel_window:]

        cal = self._calendar_arrays(
            max(1, origin_idx - 27), cap
        )
        off = lambda idx: idx - cal["idx_start"]
        wday_tail = cal["wday"][off(origin_idx - 27): off(origin_idx) + 1]

        feats = origin_features(
            obs[:, -cfg.min_history_days:], price_tail, wday_tail, cfg
        )
        return self._assemble_per_horizon(
            feats, cal, origin_idx, hs, panel_snap=cal["snap_by_series"]
        )

    def iter_test_origins(self, fold: int) -> Iterator[int]:
        """Origins test_start-1 .. test_end-1 (3-F standalone evaluation)."""
        folds = self.source.load_folds().set_index("fold")
        start = int(folds.loc[fold, "test_start_idx"])
        end = int(folds.loc[fold, "test_end_idx"])
        yield from range(start - 1, end)

    # -- internals ---------------------------------------------------------

    def _check_windows(self) -> None:
        win = self.source.load_windows()
        if "validation" in win:
            lo, hi = win["validation"]
            if (lo, hi) != (self.cfg.tune_valid_target_start,
                            self.cfg.tune_valid_target_end):
                raise ValueError(
                    "tuning validation constants disagree with mart.windows: "
                    f"cfg=({self.cfg.tune_valid_target_start}, "
                    f"{self.cfg.tune_valid_target_end}) mart=({lo}, {hi})"
                )

    def _calendar_arrays(self, start_idx: int, end_idx: int) -> dict:
        cal = self.source.load_calendar(start_idx, end_idx)
        glob = (
            cal.drop_duplicates("period_idx")
            .sort_values("period_idx")
            .reset_index(drop=True)
        )
        idx_full = np.arange(start_idx, end_idx + 1)
        if not np.array_equal(glob["period_idx"].to_numpy(), idx_full):
            raise ValueError("calendar range incomplete")
        snap_wide = cal.pivot(index="state_id", columns="period_idx", values="snap")
        snap_wide = snap_wide.reindex(columns=idx_full)
        state_rows = {s: i for i, s in enumerate(snap_wide.index)}
        srow = np.array(
            [state_rows[s] for s in self._static["state_id"]], dtype=np.int64
        )
        return {
            "idx_start": start_idx,
            "wday": glob["wday"].to_numpy(dtype=np.int64),
            "month": glob["month"].to_numpy(dtype=np.int64),
            "event_code": self.encoder.encode(
                "tgt_event_type", glob["event_type"].tolist()
            ),
            "is_event": glob["is_event"].to_numpy(dtype=np.float32),
            "snap_by_series": snap_wide.to_numpy(dtype=np.float32)[srow],
        }

    def _load_panel(self, cap: int) -> _Panel:
        dp = self.source.load_demand_price(max_period_idx=cap)
        if int(dp["period_idx"].max()) > cap:
            raise ValueError("source returned rows past the cap")
        sales, series_ids, idx_start = _pivot(dp, "sales")
        price, _, _ = _pivot(dp, "price_ffill")
        active, _, _ = _pivot(dp.assign(_a=dp["is_active"].astype(float)), "_a")
        if series_ids != tuple(self._static["series_id"]):
            raise ValueError("series order mismatch between panel and static")
        cal = self._calendar_arrays(idx_start, cap)
        return _Panel(
            series_ids=series_ids,
            idx_start=idx_start,
            sales=sales,
            price=price,
            active=active.astype(bool),
            snap=cal["snap_by_series"],
            wday=cal["wday"],
            month=cal["month"],
            event_code=cal["event_code"],
            is_event=cal["is_event"],
            first_active=self._static["first_active_idx"].to_numpy(dtype=np.int64),
        )

    def _valid_series_at(self, panel: _Panel, t: int) -> np.ndarray:
        j = panel.col(t)
        if j < self.cfg.min_history_days - 1:
            return np.zeros(len(panel.series_ids), dtype=bool)
        ok = panel.active[:, j].copy()
        ok &= np.isfinite(panel.price[:, j])
        ok &= (t - panel.first_active) >= self.cfg.min_history_days
        return ok

    def _build_frame(
        self, panel: _Panel, target_lo: int, target_hi: int, target_mode: str
    ) -> TrainFrame:
        cfg = self.cfg
        if target_mode == TARGET_PER_HORIZON:
            h_max = max(cfg.horizons)
            t_min = panel.idx_start + cfg.min_history_days - 1
            t_max = target_hi - 1
        elif target_mode == TARGET_LT_CUM:
            h_max = cfg.lead_time_cum_horizon
            t_min = panel.idx_start + cfg.min_history_days - 1
            t_max = target_hi - cfg.lead_time_cum_horizon
        else:
            raise ValueError(f"unknown target_mode {target_mode}")

        cols: dict[str, list] = {}
        metas: dict[str, list] = {"series_idx": [], "origin_idx": [],
                                  "horizon": [], "target_idx": []}
        ys: list[np.ndarray] = []

        for t in range(t_min, t_max + 1):
            valid = self._valid_series_at(panel, t)
            if not valid.any():
                continue
            j = panel.col(t)
            sales_tail = panel.sales[valid, j - cfg.min_history_days + 1: j + 1]
            price_tail = panel.price[valid, j - cfg.price_rel_window + 1: j + 1]
            wday_tail = panel.wday[j - 27: j + 1]
            feats = origin_features(sales_tail, price_tail, wday_tail, cfg)
            sidx = np.flatnonzero(valid)

            if target_mode == TARGET_PER_HORIZON:
                self._emit_per_horizon_rows(
                    panel, cols, metas, ys, feats, sidx, t,
                    target_lo, target_hi, h_max,
                )
            else:
                self._emit_lt_cum_rows(
                    panel, cols, metas, ys, feats, sidx, t, target_lo,
                )

        return self._finalize(cols, metas, ys, target_mode)

    def _emit_per_horizon_rows(
        self, panel, cols, metas, ys, feats, sidx, t, target_lo, target_hi, h_max
    ):
        cfg = self.cfg
        swd = feats["_swd_by_wday"]
        for h in cfg.horizons:
            tgt = t + h
            if tgt < target_lo or tgt > target_hi:
                continue
            k = panel.col(tgt)
            n = len(sidx)
            for name in cfg.origin_numeric_columns:
                if name == "swd_mean_4":
                    cols.setdefault(name, []).append(swd[panel.wday[k] - 1])
                else:
                    cols.setdefault(name, []).append(feats[name])
            cols.setdefault("h", []).append(np.full(n, h, dtype=np.float32))
            cols.setdefault("tgt_wday", []).append(
                np.full(n, panel.wday[k], dtype=np.float32))
            cols.setdefault("tgt_month", []).append(
                np.full(n, panel.month[k], dtype=np.float32))
            cols.setdefault("tgt_snap", []).append(panel.snap[sidx, k])
            cols.setdefault("tgt_is_event", []).append(
                np.full(n, panel.is_event[k], dtype=np.float32))
            cols.setdefault("tgt_event_type", []).append(
                np.full(n, panel.event_code[k], dtype=np.int32))
            self._emit_statics(cols, sidx)
            metas["series_idx"].append(sidx)
            metas["origin_idx"].append(np.full(n, t, dtype=np.int64))
            metas["horizon"].append(np.full(n, h, dtype=np.int64))
            metas["target_idx"].append(np.full(n, tgt, dtype=np.int64))
            ys.append(panel.sales[sidx, k])

    def _emit_lt_cum_rows(self, panel, cols, metas, ys, feats, sidx, t, target_lo):
        cfg = self.cfg
        K = cfg.lead_time_cum_horizon
        if t + 1 < target_lo:
            return
        j = panel.col(t)
        n = len(sidx)
        for name in cfg.lt_cum_numeric_columns:
            if name in feats:
                cols.setdefault(name, []).append(feats[name])
        cols.setdefault("cum_snap_days", []).append(
            panel.snap[sidx, j + 1: j + K + 1].sum(axis=1))
        cols.setdefault("cum_event_days", []).append(
            np.full(n, panel.is_event[j + 1: j + K + 1].sum(), dtype=np.float32))
        cols.setdefault("start_wday", []).append(
            np.full(n, panel.wday[j + 1], dtype=np.float32))
        self._emit_statics(cols, sidx)
        metas["series_idx"].append(sidx)
        metas["origin_idx"].append(np.full(n, t, dtype=np.int64))
        metas["horizon"].append(np.full(n, K, dtype=np.int64))
        metas["target_idx"].append(np.full(n, t + K, dtype=np.int64))
        ys.append(panel.sales[sidx, j + 1: j + K + 1].sum(axis=1))

    def _emit_statics(self, cols, sidx):
        for col, codes in self._static_codes.items():
            cols.setdefault(col, []).append(codes[sidx])

    def _assemble_per_horizon(
        self, feats, cal, origin_idx, horizons, panel_snap
    ) -> TrainFrame:
        """Inference-path assembly (all series, one origin)."""
        cfg = self.cfg
        n = len(self._static)
        sidx = np.arange(n)
        swd = feats["_swd_by_wday"]
        off = lambda idx: idx - cal["idx_start"]
        cols: dict[str, list] = {}
        metas: dict[str, list] = {"series_idx": [], "origin_idx": [],
                                  "horizon": [], "target_idx": []}
        for h in horizons:
            k = off(origin_idx + h)
            for name in cfg.origin_numeric_columns:
                if name == "swd_mean_4":
                    cols.setdefault(name, []).append(swd[cal["wday"][k] - 1])
                else:
                    cols.setdefault(name, []).append(feats[name])
            cols.setdefault("h", []).append(np.full(n, h, dtype=np.float32))
            cols.setdefault("tgt_wday", []).append(
                np.full(n, cal["wday"][k], dtype=np.float32))
            cols.setdefault("tgt_month", []).append(
                np.full(n, cal["month"][k], dtype=np.float32))
            cols.setdefault("tgt_snap", []).append(panel_snap[:, k])
            cols.setdefault("tgt_is_event", []).append(
                np.full(n, cal["is_event"][k], dtype=np.float32))
            cols.setdefault("tgt_event_type", []).append(
                np.full(n, cal["event_code"][k], dtype=np.int32))
            self._emit_statics(cols, sidx)
            metas["series_idx"].append(sidx)
            metas["origin_idx"].append(np.full(n, origin_idx, dtype=np.int64))
            metas["horizon"].append(np.full(n, h, dtype=np.int64))
            metas["target_idx"].append(np.full(n, origin_idx + h, dtype=np.int64))
        return self._finalize(cols, metas, ys=None,
                              target_mode=TARGET_PER_HORIZON)

    def _finalize(self, cols, metas, ys, target_mode: str) -> TrainFrame:
        cfg = self.cfg
        order = (cfg.feature_columns if target_mode == TARGET_PER_HORIZON
                 else cfg.lt_cum_feature_columns)
        cats = tuple(c for c in cfg._cats()
                     if target_mode == TARGET_PER_HORIZON or c != "tgt_event_type")
        data = {}
        for name in order:
            arrs = cols.get(name, [])
            if not arrs:
                raise ValueError(f"missing feature column {name}")
            merged = np.concatenate(arrs)
            data[name] = merged.astype(
                np.int32 if name in cats else np.float32, copy=False
            )
        X = pd.DataFrame(data, columns=list(order))
        series_idx = np.concatenate(metas["series_idx"])
        meta = pd.DataFrame({
            "series_id": self._static["series_id"].to_numpy()[series_idx],
            "origin_idx": np.concatenate(metas["origin_idx"]),
            "horizon": np.concatenate(metas["horizon"]),
            "target_idx": np.concatenate(metas["target_idx"]),
        })
        y = (np.concatenate(ys).astype(np.float32) if ys
             else np.empty(0, dtype=np.float32))
        return TrainFrame(
            X=X, y=y, meta=meta, categorical=cats,
            spec_hash=cfg.spec_hash(),
            encoder_hash=self.encoder.encoder_hash(),
            target_mode=target_mode,
        )
