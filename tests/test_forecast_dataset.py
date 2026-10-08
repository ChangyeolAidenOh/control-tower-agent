"""Tests for core_pipeline/forecast/dataset.py (Stage 3-A contract).

All tests run on ArrayPanelSource with a synthetic panel; no DB required.
Covers: D3 boundary cases, zero-sale-day retention, DemandHistory future
guard, future-mutation invariance, scenario isolation, advance/appended
immutability, encoder consistency, tuning split ranges, horizon clipping.
"""

import numpy as np
import pandas as pd
import pytest

from core_pipeline.forecast.dataset import (
    ArrayPanelSource,
    DatasetConfig,
    DemandHistory,
    FoldDatasetBuilder,
    FutureAccessError,
    MissingKnownCalendarError,
    TARGET_LT_CUM,
    retrain_fold_for,
)
from dataclasses import replace

N_DAYS = 200
TRAIN_END = 150
SERIES = ("itemA__ST1", "itemB__ST1", "itemC__ST2")


def _make_source(cal_end=N_DAYS + 28):
    rng = np.random.default_rng(7)
    days = np.arange(1, N_DAYS + 1)
    cal_days = np.arange(1, cal_end + 1)
    static = pd.DataFrame({
        "series_id": SERIES,
        "item_id": ["itemA", "itemB", "itemC"],
        "store_id": ["ST1", "ST1", "ST2"],
        "state_id": ["CA", "CA", "TX"],
        "dept_id": ["FOODS_3", "FOODS_3", "HOBBIES_1"],
        "cat_id": ["FOODS", "FOODS", "HOBBIES"],
        "group_id": ["B01", "B01", "B02"],
        "first_active_idx": [1, 5, 100],
    })
    rows = []
    for s, first_active, listed_from in zip(SERIES, [1, 5, 100], [1, 1, 10]):
        sales = rng.poisson(3.0, N_DAYS).astype(float)
        sales[: first_active - 1] = 0.0
        if s == SERIES[0]:
            sales[119:122] = 0.0  # zero-sale days inside the active period
        price = np.where(days >= listed_from, 2.5, np.nan)
        rows.append(pd.DataFrame({
            "series_id": s,
            "period_idx": days,
            "sales": sales,
            "price_ffill": price,
            "is_active": days >= first_active,
        }))
    dp = pd.concat(rows, ignore_index=True)

    wday = ((cal_days - 1) % 7) + 1
    month = (((cal_days - 1) // 28) % 12) + 1
    event_type = np.where(cal_days % 50 == 0, "Cultural", "none")
    is_event = (event_type != "none").astype(float)
    cal_rows = []
    for state, offset in (("CA", 0), ("TX", 3)):
        snap = ((cal_days + offset) % 10 < 3).astype(float)
        cal_rows.append(pd.DataFrame({
            "state_id": state, "period_idx": cal_days, "wday": wday,
            "month": month, "snap": snap, "event_type": event_type,
            "is_event": is_event,
        }))
    cal = pd.concat(cal_rows, ignore_index=True)

    folds = pd.DataFrame({
        "fold": [1], "train_end_idx": [TRAIN_END],
        "test_start_idx": [TRAIN_END + 1], "test_end_idx": [TRAIN_END + 28],
    })
    windows = {"validation": (101, 140)}
    return ArrayPanelSource(static, dp, cal, folds, windows), dp


@pytest.fixture()
def cfg():
    return replace(DatasetConfig(), tune_train_target_end=100,
                   tune_valid_target_start=101, tune_valid_target_end=140)


@pytest.fixture()
def built(cfg):
    source, dp = _make_source()
    return FoldDatasetBuilder(cfg, source), source, dp


def test_retrain_fold_mapping():
    assert [retrain_fold_for(f) for f in range(1, 13)] == \
        [1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10]
    with pytest.raises(ValueError):
        retrain_fold_for(0)


def test_d3_boundary_rows(built):
    builder, _, dp = built
    tf = builder.training_frame(1)
    m = tf.meta
    assert ((m["origin_idx"] == TRAIN_END - 1) & (m["horizon"] == 1)).any()
    assert not ((m["origin_idx"] == TRAIN_END - 1) & (m["horizon"] == 2)).any()
    assert not (m["origin_idx"] >= TRAIN_END).any()
    assert (m["target_idx"] == m["origin_idx"] + m["horizon"]).all()
    assert m["target_idx"].max() == TRAIN_END
    # y aligns with the panel value at target_idx
    row = m[(m["series_id"] == SERIES[0]) & (m["origin_idx"] == 140)
            & (m["horizon"] == 5)].index[0]
    truth = dp[(dp["series_id"] == SERIES[0]) & (dp["period_idx"] == 145)
               ]["sales"].iloc[0]
    assert tf.y[row] == np.float32(truth)


def test_min_history_and_activity_filter(built):
    builder, _, _ = built
    m = builder.training_frame(1).meta
    # series C first active at d_100: needs origin >= 156 > train_end - 1
    assert SERIES[2] not in set(m["series_id"])
    assert m["origin_idx"].min() >= 57  # idx_start + 56 trailing days


def test_zero_sale_days_kept(built):
    builder, _, _ = built
    tf = builder.training_frame(1)
    m = tf.meta
    # origins on zero-sale days inside the active period remain rows
    sel = (m["series_id"] == SERIES[0]) & (m["origin_idx"] == 120)
    assert sel.any()
    assert tf.X.loc[sel[sel].index[0], "sales_lag_1"] == 0.0


def test_tuning_split_target_ranges(built):
    builder, _, _ = built
    tr, va = builder.tuning_split()
    assert tr.meta["target_idx"].max() <= 100
    assert va.meta["target_idx"].min() >= 101
    assert va.meta["target_idx"].max() <= 140


def test_future_access_guard(built):
    _, source, _ = built
    hist = DemandHistory.from_source(source, max_period_idx=TRAIN_END)
    with pytest.raises(FutureAccessError):
        hist.observed_through(TRAIN_END + 1)
    obs = hist.observed_through(TRAIN_END)
    assert obs.shape[1] == TRAIN_END
    assert not obs.flags.writeable


def test_advance_and_appended_immutability(built):
    _, source, dp = built
    full = dp.pivot(index="series_id", columns="period_idx",
                    values="sales").sort_index().to_numpy(np.float32)
    series = tuple(sorted(SERIES))
    hist = DemandHistory(series, 1, full[:, :TRAIN_END + 7],
                         current_idx=TRAIN_END)
    h2 = hist.advance()
    assert hist.current_idx == TRAIN_END and h2.current_idx == TRAIN_END + 1
    tail = DemandHistory(series, 1, full[:, :TRAIN_END], current_idx=TRAIN_END)
    with pytest.raises(ValueError):
        tail.advance()
    h3 = tail.appended(np.ones(3))
    assert h3.current_idx == TRAIN_END + 1
    assert tail.observed_through(TRAIN_END).shape[1] == TRAIN_END
    with pytest.raises(ValueError):
        hist.appended(np.ones(3))  # preloaded days remain


def test_future_mutation_invariance_and_scenario_isolation(built, cfg):
    builder, source, dp = built
    base = dp.pivot(index="series_id", columns="period_idx",
                    values="sales").sort_index().to_numpy(np.float32)
    series = tuple(sorted(SERIES))
    future_a = np.zeros((3, 10), dtype=np.float32)
    future_b = np.full((3, 10), 99.0, dtype=np.float32)
    ha = DemandHistory(series, 1, np.concatenate(
        [base[:, :TRAIN_END], future_a], axis=1), current_idx=TRAIN_END)
    hb = DemandHistory(series, 1, np.concatenate(
        [base[:, :TRAIN_END], future_b], axis=1), current_idx=TRAIN_END)
    fa = builder.inference_frame(ha, TRAIN_END)
    fb = builder.inference_frame(hb, TRAIN_END)
    pd.testing.assert_frame_equal(fa.X, fb.X)
    pd.testing.assert_frame_equal(fa.meta, fb.meta)


def test_full_28_horizons_at_demand_end(built):
    # Fold-12 analogue: demand/price end at N_DAYS, known calendar extends
    # 28 more days -> every origin up to the last demand day still plans 28.
    builder, source, dp = built
    for origin in (N_DAYS - 10, N_DAYS - 1, N_DAYS):
        hist = DemandHistory.from_source(source, max_period_idx=origin)
        tf = builder.inference_frame(hist, origin)
        assert len(tf.X) == len(SERIES) * 28
        assert sorted(tf.meta["horizon"].unique()) == list(range(1, 29))
        assert tf.meta["target_idx"].max() == origin + 28
        assert len(tf.y) == 0


def test_missing_known_calendar_raises(cfg):
    source, _ = _make_source(cal_end=N_DAYS + 10)
    builder = FoldDatasetBuilder(cfg, source)
    hist = DemandHistory.from_source(source, max_period_idx=N_DAYS)
    with pytest.raises(MissingKnownCalendarError):
        builder.inference_frame(hist, N_DAYS)
    # a horizon subset that fits is still served
    tf = builder.inference_frame(hist, N_DAYS, horizons=range(1, 11))
    assert sorted(tf.meta["horizon"].unique()) == list(range(1, 11))


def test_encoder_consistency_train_vs_inference(built):
    builder, source, _ = built
    tr = builder.training_frame(1)
    hist = DemandHistory.from_source(source, max_period_idx=TRAIN_END)
    inf = builder.inference_frame(hist, TRAIN_END)
    code_tr = tr.X.loc[tr.meta["series_id"] == SERIES[0], "series_id"].iloc[0]
    code_inf = inf.X.loc[inf.meta["series_id"] == SERIES[0], "series_id"].iloc[0]
    assert code_tr == code_inf
    assert tr.encoder_hash == inf.encoder_hash
    assert tr.spec_hash == inf.spec_hash


def test_lt_cum_target_alignment(built, cfg):
    builder, _, dp = built
    tf = builder.training_frame(1, target_mode=TARGET_LT_CUM)
    m = tf.meta
    assert (m["horizon"] == cfg.lead_time_cum_horizon).all()
    assert m["target_idx"].max() <= TRAIN_END
    sel = m[(m["series_id"] == SERIES[0]) & (m["origin_idx"] == 140)]
    truth = dp[(dp["series_id"] == SERIES[0])
               & dp["period_idx"].between(141, 143)]["sales"].sum()
    assert tf.y[sel.index[0]] == np.float32(truth)
    assert "tgt_event_type" not in tf.X.columns


def test_windows_mismatch_rejected(cfg):
    source, _ = _make_source()
    bad = replace(cfg, tune_valid_target_end=139)
    with pytest.raises(ValueError):
        FoldDatasetBuilder(bad, source)


def test_training_frame_rejects_non_retrain_fold(built):
    builder, _, _ = built
    with pytest.raises(ValueError):
        builder.training_frame(2)
