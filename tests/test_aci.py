"""Boundary tests for the delayed adaptive CQR margin tracker (ST3-07).
Covers the reviewer-specified table: release timing, state continuity,
zero-width floor, interval validity, output-interval-based miss, future
tampering invariance, and trailing-origin update freeze."""

import numpy as np
import pytest

from core_pipeline.forecast import aci
from core_pipeline.forecast.dataset import DemandHistory, FutureAccessError

H = aci.HORIZON


def make_history(values, start_idx=1, current=None):
    n_s = values.shape[0]
    return DemandHistory([f"s{i}" for i in range(n_s)], start_idx,
                         values, current_idx=current)


def setup_replay(n_origins=35, n_series=2, demand=3.0,
                 lo_val=0.0, hi_val=10.0, extra_days=5):
    first = 100
    origins = np.arange(first, first + n_origins)
    last_needed = origins[-1] + extra_days
    values = np.full((n_series, last_needed), demand, dtype=np.float32)
    hist = make_history(values, start_idx=1, current=first)
    lo = np.full((n_origins, n_series, H), lo_val)
    hi = np.full((n_origins, n_series, H), hi_val)
    return origins, lo, hi, hist, values


def test_release_timing_is_origin_plus_28():
    # every bundle misses (demand 50 outside [0,10]); first release may
    # only land when the clock reaches origins[0] + 28
    origins, lo, hi, hist, _ = setup_replay(n_origins=32, demand=50.0)
    res = aci.replay(origins, lo, hi, hist)
    upd = res["updates_before"]           # (n_o, n_s, H)
    k_rel = 28                            # position of origins[0] + 28
    assert (upd[:k_rel] == 0).all()       # h=1 included: nothing early
    assert (upd[k_rel] == 1).all()        # exactly one bundle released
    assert (upd[-1] == 4).all()           # origins 100..103 released
    assert (res["m_at_origin"][:k_rel] == 0).all()
    assert (res["m_final"] > 0).all()     # misses pushed margins up


def test_state_accumulates_across_boundaries():
    origins, lo, hi, hist, _ = setup_replay(n_origins=35, demand=50.0)
    res = aci.replay(origins, lo, hi, hist)
    upd = res["updates_before"][:, 0, 0]
    assert (np.diff(upd) >= 0).all() and upd[-1] > 1   # no reset anywhere


def test_zero_width_interval_can_widen():
    origins, lo, hi, hist, _ = setup_replay(
        n_origins=35, demand=5.0, lo_val=0.0, hi_val=0.0)
    res = aci.replay(origins, lo, hi, hist)
    assert (res["m_final"] > 0).all()
    # eps floor (>=1.0) lets a grown margin open the [0,0] interval
    assert res["hi_adj"][-1].max() > 0.0


def test_intervals_always_valid():
    rng = np.random.default_rng(7)
    lo = rng.normal(-1, 2, (50, H))
    hi = lo + np.abs(rng.normal(0, 0.01, (50, H)))     # tiny widths
    w_eff = np.maximum(hi - lo, 1.0)
    for m_val in (-0.5, -0.3, 0.0, 0.7, 5.0):
        la, ha = aci.adjust_intervals(lo, hi, np.full_like(lo, m_val), w_eff)
        assert (la >= 0).all() and (ha >= la).all()


def test_miss_uses_issued_interval_not_raw_score():
    # base interval entirely negative -> issued interval clips to [0, 0];
    # demand 0 is INSIDE the issued interval, so margins must SHRINK
    # even though the raw base-interval score max(lo-y, y-hi) is positive
    origins, lo, hi, hist, _ = setup_replay(
        n_origins=32, demand=0.0, lo_val=-3.0, hi_val=-1.0)
    res = aci.replay(origins, lo, hi, hist)
    assert (res["m_final"] < 0).all()


def test_future_tampering_invariance():
    origins, lo, hi, hist, values = setup_replay(n_origins=32, demand=4.0)
    res_a = aci.replay(origins, lo, hi, hist)
    tampered = values.copy()
    tampered[:, origins[-1]:] = 999.0   # periods > last replay-read day
    res_b = aci.replay(origins, lo, hi,
                       make_history(tampered, 1, current=origins[0]))
    np.testing.assert_array_equal(res_a["lo_adj"], res_b["lo_adj"])
    np.testing.assert_array_equal(res_a["hi_adj"], res_b["hi_adj"])
    np.testing.assert_array_equal(res_a["m_final"], res_b["m_final"])


def test_trailing_origins_never_update():
    origins, lo, hi, hist, _ = setup_replay(n_origins=40, demand=50.0)
    res = aci.replay(origins, lo, hi, hist)
    assert res["origins_released"] == 40 - 28
    assert res["origins_unreleased"] == 28
    # total updates equal released bundles exactly
    assert res["updates_before"][-1, 0, 0] == 40 - 28


def test_guard_blocks_unobserved_days():
    values = np.zeros((2, 50), dtype=np.float32)
    hist = make_history(values, start_idx=1, current=30)
    with pytest.raises(FutureAccessError):
        hist.observed_through(31)


def test_replay_rejects_gapped_origins():
    origins, lo, hi, hist, _ = setup_replay(n_origins=30)
    bad = origins.copy()
    bad[-1] += 1
    with pytest.raises(ValueError):
        aci.replay(bad, lo, hi, hist)
