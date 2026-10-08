"""Tests for 2-F demand paths: T4 censoring designation (spec v0.1 section 11)
plus the fold-level shock contract.

T4: at p = 0.06 the designated-day count equals round_half_up(p x active days)
per SKU, designation happens only on zero-sales activity days, replacement is
the mean of non-designated days in the 28-day lookback, and SKUs with too few
zero-sales days designate all of them and report p_hat.
"""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.simulator.demand_paths import (
    _round_half_up,
    apply_censoring,
    apply_shock,
    build_fold_paths,
    draw_shock,
)
from core_pipeline.simulator.engine import load_sim_config, spawn_stream

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


def synth_history(rng, n_days=200, n_sku=8, zero_rate=0.35, start_max=40):
    mat = np.zeros((n_days, n_sku))
    for i in range(n_sku):
        start = int(rng.integers(0, start_max))
        vals = rng.poisson(3.0, size=n_days - start).astype(float)
        zeros = rng.random(n_days - start) < zero_rate
        vals[zeros] = 0.0
        if vals.size:
            vals[0] = max(vals[0], 1.0)
        mat[start:, i] = vals
    return mat


# --------------------------------------------------------------------------- #
# T4: censoring designation
# --------------------------------------------------------------------------- #

def test_round_half_up():
    assert _round_half_up(0.5) == 1
    assert _round_half_up(1.5) == 2
    assert _round_half_up(1.49) == 1


def test_t4_designation_counts(cfg):
    rng = spawn_stream(cfg.seeds.censor, 1, 0)
    mat = synth_history(spawn_stream(999, 0))
    window = (150, 177)  # 28 days
    res = apply_censoring(mat, 0.06, rng, window_rows=window,
                          lookback_days=cfg.censor_replacement_lookback_days)
    lo, hi = window
    for i in range(mat.shape[1]):
        positive = np.nonzero(mat[:, i] > 0)[0]
        act_start = positive[0]
        days = np.arange(max(lo, act_start), hi + 1)
        eligible = days[mat[days, i] == 0.0]
        target = _round_half_up(0.06 * days.size)
        assert res.n_active[i] == days.size
        assert res.n_target[i] == target
        assert res.n_designated[i] == min(target, eligible.size)
        # designated days are zero-sales activity days inside the window
        desig = np.nonzero(res.designated[:, i])[0]
        assert np.all((desig >= max(lo, act_start)) & (desig <= hi))
        assert np.all(mat[desig, i] == 0.0)
        if eligible.size < target:
            assert res.p_hat[i] == pytest.approx(eligible.size / days.size)
        else:
            assert res.p_hat[i] == pytest.approx(target / days.size)
    # only designated cells changed
    changed = np.nonzero(res.demand != mat)
    assert np.all(res.designated[changed])


def test_t4_replacement_is_lookback_mean(cfg):
    rng = spawn_stream(cfg.seeds.censor, 2, 3)
    mat = synth_history(spawn_stream(998, 0))
    res = apply_censoring(mat, 0.17, rng, window_rows=(150, 177),
                          lookback_days=28)
    found = 0
    for i in range(mat.shape[1]):
        act_start = int(np.nonzero(mat[:, i] > 0)[0][0])
        for day in np.nonzero(res.designated[:, i])[0]:
            lb = max(act_start, day - 28)
            span = np.arange(lb, day)
            span = span[~res.designated[span, i]]
            if span.size:
                assert res.demand[day, i] == pytest.approx(float(mat[span, i].mean()))
                found += 1
    assert found > 0


def test_t4_p_zero_noop(cfg):
    rng = spawn_stream(cfg.seeds.censor, 1, 0)
    mat = synth_history(spawn_stream(997, 0))
    res = apply_censoring(mat, 0.0, rng, window_rows=(150, 177), lookback_days=28)
    np.testing.assert_array_equal(res.demand, mat)
    assert not res.designated.any()


def test_t4_determinism_and_stream_separation(cfg):
    mat = synth_history(spawn_stream(996, 0))
    a = apply_censoring(mat, 0.06, spawn_stream(cfg.seeds.censor, 1, 0),
                        window_rows=(150, 177), lookback_days=28)
    b = apply_censoring(mat, 0.06, spawn_stream(cfg.seeds.censor, 1, 0),
                        window_rows=(150, 177), lookback_days=28)
    np.testing.assert_array_equal(a.designated, b.designated)
    np.testing.assert_array_equal(a.demand, b.demand)
    c = apply_censoring(mat, 0.06, spawn_stream(cfg.seeds.censor, 2, 0),
                        window_rows=(150, 177), lookback_days=28)
    assert not np.array_equal(a.designated, c.designated)


# --------------------------------------------------------------------------- #
# Shock: fold-level draw, one bundle, 7 consecutive days, x1.5
# --------------------------------------------------------------------------- #

def test_shock_draw_contract(cfg):
    gids = [f"B{k:02d}" for k in range(1, 11)]
    draws = {f: draw_shock(gids, cfg.test_window_days, cfg, f) for f in range(1, 13)}
    for f, d in draws.items():
        assert d.group_id in gids
        assert 0 <= d.start_offset <= cfg.test_window_days - d.duration_days
        assert d.duration_days == 7 and d.multiplier == 1.5
        again = draw_shock(gids, cfg.test_window_days, cfg, f)
        assert (again.group_id, again.start_offset) == (d.group_id, d.start_offset)
    assert len({(d.group_id, d.start_offset) for d in draws.values()}) > 1


def test_apply_shock_span(cfg):
    window = np.ones((28, 4))
    d = draw_shock(["A", "B"], 28, cfg, fold=3)
    out = apply_shock(window, d)
    sl = slice(d.start_offset, d.start_offset + 7)
    assert np.all(out[sl] == 1.5)
    mask = np.ones(28, dtype=bool)
    mask[sl] = False
    assert np.all(out[mask] == 1.0)


def test_build_fold_paths_one_bundle_shocked(cfg):
    rng = spawn_stream(995, 0)
    bundles = {f"B{k:02d}": (("s",), synth_history(rng, n_sku=3)) for k in range(1, 6)}
    paths = build_fold_paths(
        bundles, fold=4, warmup_start_row=136, test_start_row=150,
        test_end_row=177, p=0.06, cfg=cfg,
    )
    assert sum(1 for bp in paths.values() if bp.shocked) == 1
    for gid, bp in paths.items():
        assert bp.history.shape[0] == 136
        assert bp.warmup.shape[0] == 14
        assert bp.test.shape[0] == 28
        assert bp.censoring is not None
        # paths are shared objects per scenario: rebuilding is deterministic
    again = build_fold_paths(
        bundles, fold=4, warmup_start_row=136, test_start_row=150,
        test_end_row=177, p=0.06, cfg=cfg,
    )
    for gid in paths:
        np.testing.assert_array_equal(paths[gid].test, again[gid].test)
