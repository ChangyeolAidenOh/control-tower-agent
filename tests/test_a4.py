"""Unit tests for the A4 base-stock policy (2-D-1).

Covers: expanding-window target level (activity handling, interpolation),
order-up-to on inventory position, both pre-registered allocation rules, and
strict-mode engine integration (A4 output is feasible by construction).
"""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.policies.a4_basestock import (
    TAU_REFERENCE,
    A4BaseStock,
    A4Config,
    allocate_b_priority,
    make_reference_a4,
    target_level,
)
from core_pipeline.simulator.engine import (
    BundleState,
    Observation,
    SkuParams,
    load_sim_config,
    run_window,
    spawn_stream,
    weighted_sum,
)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


def make_obs(hist, inv, pipeline, k_g, b=None, n_sku=None):
    n_sku = n_sku or hist.shape[1]
    return Observation(
        t_idx=hist.shape[0] - 1,
        demand_hist=hist,
        inv=np.asarray(inv, dtype=float),
        pipeline=np.asarray(pipeline, dtype=float),
        k_g=k_g,
        h=np.ones(n_sku),
        b=np.ones(n_sku) * 3.0 if b is None else np.asarray(b, dtype=float),
        w=np.ones(n_sku),
    )


# --------------------------------------------------------------------------- #
# Target level: expanding empirical quantile of 3-day sums over activity
# --------------------------------------------------------------------------- #

def test_target_level_known_series():
    series = np.array([2.0, 1.0, 3.0, 0.0, 4.0])
    # 3-day sums: [6, 4, 7]; tau=0.5 linear -> 6.0
    assert target_level(series, 0.5) == 6.0


def test_target_level_excludes_pre_activity():
    base = np.array([2.0, 1.0, 3.0, 0.0, 4.0])
    padded = np.concatenate([np.zeros(10), base])
    assert target_level(padded, 0.5) == target_level(base, 0.5)


def test_target_level_keeps_zero_days_inside_activity():
    series = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    # activity spans all 5 days; 3-day sums = [1, 0, 1]
    assert target_level(series, 0.5) == 1.0
    assert target_level(series, 0.25) == pytest.approx(0.5)  # zeros kept in sample


def test_target_level_no_activity_or_short():
    assert target_level(np.zeros(20), 0.9) == 0.0
    # 2 active days only: fallback daily quantile x 3
    series = np.array([0.0, 0.0, 2.0, 4.0])
    assert target_level(series, 0.5) == pytest.approx(3.0 * 3.0)


# --------------------------------------------------------------------------- #
# Order-up-to on inventory position
# --------------------------------------------------------------------------- #

def test_order_up_to_inventory_position():
    hist = np.tile(np.array([[2.0, 5.0]]), (30, 1))  # constant demand 2 and 5
    policy = A4BaseStock(A4Config(tau=0.5, allocation="proportional"))
    # S = 3-day sums are constant: 6 and 15
    obs = make_obs(hist, inv=[1.0, 20.0], pipeline=[[2.0, 1.0], [0.0, 0.0]], k_g=100.0)
    q = policy.decide(obs)
    # SKU0: IP = 1+3 = 4 -> q = 6-4 = 2; SKU1: IP = 20 >= 15 -> q = 0
    np.testing.assert_allclose(q, np.array([2.0, 0.0]))


# --------------------------------------------------------------------------- #
# Allocation rules
# --------------------------------------------------------------------------- #

def test_proportional_allocation_hits_capacity(cfg):
    hist = np.tile(np.array([[4.0, 4.0]]), (30, 1))
    policy = A4BaseStock(A4Config(tau=0.5, allocation="proportional"))
    obs = make_obs(hist, inv=[0.0, 0.0], pipeline=np.zeros((2, 2)), k_g=6.0)
    q = policy.decide(obs)  # raw request 12 each -> scaled to sum 6
    assert weighted_sum(q, obs.w) == 6.0
    np.testing.assert_allclose(q, np.array([3.0, 3.0]))


def test_b_priority_allocation_order():
    q_raw = np.array([5.0, 5.0, 5.0])
    b = np.array([1.0, 9.0, 3.0])
    w = np.ones(3)
    q = allocate_b_priority(q_raw, b, w, k_g=8.0)
    # priority: SKU1 (b=9) full 5, SKU2 (b=3) partial 3, SKU0 gets 0
    np.testing.assert_allclose(q, np.array([0.0, 5.0, 3.0]))
    assert weighted_sum(q, w) <= 8.0


def test_b_priority_tie_break_by_index():
    q_raw = np.array([4.0, 4.0])
    b = np.array([2.0, 2.0])
    q = allocate_b_priority(q_raw, b, np.ones(2), k_g=5.0)
    np.testing.assert_allclose(q, np.array([4.0, 1.0]))


def test_b_priority_no_excess_capacity_left_unused():
    q_raw = np.array([1.0, 1.0])
    b = np.array([5.0, 1.0])
    q = allocate_b_priority(q_raw, b, np.ones(2), k_g=10.0)
    np.testing.assert_allclose(q, q_raw)


# --------------------------------------------------------------------------- #
# Engine integration: strict mode accepts A4 output; episodes are deterministic
# --------------------------------------------------------------------------- #

def run_a4_episode(cfg, allocation: str, seed_path=(1, 0)) -> float:
    n_sku, lead = 5, cfg.lead_time_core
    rng = spawn_stream(cfg.seeds.eval, *seed_path)
    hist = rng.poisson(3.0, size=(120, n_sku)).astype(float)
    demand = rng.poisson(3.0, size=(28, n_sku)).astype(float)
    params = SkuParams(
        series_ids=tuple(f"S{i}" for i in range(n_sku)),
        h=np.linspace(0.5, 1.5, n_sku),
        b=np.linspace(0.5, 1.5, n_sku) * 5.0,
        w=np.ones(n_sku),
    )
    state = BundleState(inv=np.full(n_sku, 6.0), pipeline=np.full((n_sku, lead), 3.0))
    policy = A4BaseStock(A4Config(tau=TAU_REFERENCE, allocation=allocation))
    final, days = run_window(
        state, demand, hist, policy, params, k_g=10.0, cfg=cfg, t0_idx=120,
    )
    assert all(not r.projection_applied for r in days)  # feasible by construction
    assert any(r.binding for r in days)                 # tight capacity binds
    return sum(r.holding_cost + r.shortage_cost for r in days)


def test_a4_strict_mode_both_rules(cfg):
    for allocation in ("proportional", "b_priority"):
        c1 = run_a4_episode(cfg, allocation)
        c2 = run_a4_episode(cfg, allocation)
        assert c1 == c2


def test_reference_a4_fixed_config():
    ref = make_reference_a4()
    assert ref.config.tau == pytest.approx(5.0 / 6.0)
    assert ref.config.allocation == "proportional"
    assert ref.config.protection_days == 3
    assert ref.config.quantile_method == "linear"


# --------------------------------------------------------------------------- #
# A4FromTargets equality with A4BaseStock (2-D-2 fast path, same code path)
# --------------------------------------------------------------------------- #

def test_from_targets_matches_basestock(cfg):
    from core_pipeline.policies.a4_basestock import A4FromTargets, target_level_grid

    n_sku, lead = 6, cfg.lead_time_core
    rng = spawn_stream(cfg.seeds.eval, 9, 9)
    full = rng.poisson(2.5, size=(160, n_sku)).astype(float)
    hist, window = full[:120], full[120:148]
    params = SkuParams(
        series_ids=tuple(f"S{i}" for i in range(n_sku)),
        h=np.linspace(0.4, 2.0, n_sku),
        b=np.linspace(0.4, 2.0, n_sku) * 5.0,
        w=np.ones(n_sku),
    )
    for allocation in ("proportional", "b_priority"):
        for tau in (0.6, 5.0 / 6.0, 0.99):
            targets = np.empty((28, n_sku))
            for j in range(28):
                for i in range(n_sku):
                    targets[j, i] = target_level_grid(full[: 120 + j + 1, i],
                                                     np.array([tau]))[0]
            state_a = BundleState(inv=np.full(n_sku, 5.0),
                                  pipeline=np.full((n_sku, lead), 2.0))
            state_b = BundleState(inv=np.full(n_sku, 5.0),
                                  pipeline=np.full((n_sku, lead), 2.0))
            pol_a = A4BaseStock(A4Config(tau=tau, allocation=allocation))
            pol_b = A4FromTargets(targets, t0_idx=120, allocation=allocation)
            _, days_a = run_window(state_a, window, hist, pol_a, params,
                                   k_g=9.0, cfg=cfg, t0_idx=120)
            _, days_b = run_window(state_b, window, hist, pol_b, params,
                                   k_g=9.0, cfg=cfg, t0_idx=120)
            for ra, rb in zip(days_a, days_b):
                np.testing.assert_array_equal(ra.order, rb.order)
