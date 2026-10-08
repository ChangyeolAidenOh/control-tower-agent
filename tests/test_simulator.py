"""Unit tests for the state-transition engine (simulator_spec.md v0.1 section 11).

Test timing split (review decision 2026-10-08, not a preregistration change):
  after 2-A: T1 lead time, T2 lost sales, T3 capacity, T7 determinism (this file)
  after 2-F: T4 censoring assignment
  after 2-C: T5 terminal treatment vs continuation
  after 2-G: T6 learning LP vs engine cost match (< 0.1%)
"""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.simulator.engine import (
    BundleState,
    EngineError,
    Observation,
    SkuParams,
    ViolationHandling,
    check_orders,
    gate_check,
    init_warmup_state,
    load_sim_config,
    midnight_transition,
    project_orders,
    run_window,
    spawn_stream,
    terminal_cost,
    weighted_sum,
)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


def make_params(n_sku: int, h: float = 1.0, r: float = 3.0) -> SkuParams:
    return SkuParams(
        series_ids=tuple(f"SKU_{i}" for i in range(n_sku)),
        h=np.full(n_sku, h),
        b=np.full(n_sku, h * r),
        w=np.ones(n_sku),
    )


def zero_state(n_sku: int, lead_time: int) -> BundleState:
    return BundleState(inv=np.zeros(n_sku), pipeline=np.zeros((n_sku, lead_time)))


class FixedSchedulePolicy:
    """Submit a pre-set order on scheduled days, zero otherwise."""

    name = "fixed_schedule"

    def __init__(self, schedule: dict[int, np.ndarray], n_sku: int):
        self.schedule = schedule
        self.n_sku = n_sku

    def decide(self, obs: Observation) -> np.ndarray:
        return self.schedule.get(obs.t_idx, np.zeros(self.n_sku))


class SeededNoisePolicy:
    """Stochastic policy drawing from a hierarchical stream; exercises T7."""

    name = "seeded_noise"

    def __init__(self, seed: int, fold: int, bundle_idx: int, scale: float):
        self.rng = spawn_stream(seed, fold, bundle_idx)
        self.scale = scale

    def decide(self, obs: Observation) -> np.ndarray:
        q = self.rng.uniform(0.0, self.scale, size=obs.inv.shape[0])
        return project_orders(q, np.ones_like(q), obs.k_g)


# --------------------------------------------------------------------------- #
# T1: zero demand, zero initial state, 1 unit ordered on day D
#     -> I_{D+1} = I_{D+2} = 0, I_{D+3} = 1 (order D evening, arrival D+3 morning)
# --------------------------------------------------------------------------- #

def test_t1_lead_time_tracking(cfg):
    n_sku, lead = 1, cfg.lead_time_core
    demand = np.zeros((5, n_sku))
    policy = FixedSchedulePolicy({0: np.array([1.0])}, n_sku)
    final, days = run_window(
        zero_state(n_sku, lead), demand, np.empty((0, n_sku)), policy,
        make_params(n_sku), k_g=10.0, cfg=cfg, t0_idx=0,
    )
    inv_start = {r.t_idx: float(r.inv_start[0]) for r in days}
    assert inv_start[1] == 0.0
    assert inv_start[2] == 0.0
    assert inv_start[3] == 1.0
    assert inv_start[4] == 1.0


def test_t1_stress_lead_times(cfg):
    # L = 1: order D -> arrival D+2 morning; L = 3: order D -> arrival D+4.
    for lead, arrival_day in [(1, 2), (3, 4)]:
        n_sku = 1
        demand = np.zeros((6, n_sku))
        policy = FixedSchedulePolicy({0: np.array([1.0])}, n_sku)
        _, days = run_window(
            zero_state(n_sku, lead), demand, np.empty((0, n_sku)), policy,
            make_params(n_sku), k_g=10.0, cfg=cfg, t0_idx=0,
        )
        inv_start = {r.t_idx: float(r.inv_start[0]) for r in days}
        for t in range(1, arrival_day):
            assert inv_start[t] == 0.0
        assert inv_start[arrival_day] == 1.0


# --------------------------------------------------------------------------- #
# T2: demand > inventory -> u_t = d_t - I_t, no carryover, I_{t+1} = 0 + arrival
# --------------------------------------------------------------------------- #

def test_t2_lost_sales(cfg):
    n_sku, lead = 1, cfg.lead_time_core
    state = BundleState(inv=np.array([5.0]), pipeline=np.array([[2.0, 0.0]]))
    demand = np.array([[8.0], [0.0]])
    policy = FixedSchedulePolicy({}, n_sku)
    _, days = run_window(
        state, demand, np.empty((0, n_sku)), policy,
        make_params(n_sku), k_g=10.0, cfg=cfg, t0_idx=0,
    )
    d0, d1 = days
    assert float(d0.sales[0]) == 5.0
    assert float(d0.unmet[0]) == 3.0
    assert float(d0.inv_eod[0]) == 0.0
    # next day: no backorder carryover; on-hand = 0 + pipeline arrival (2.0)
    assert float(d1.inv_start[0]) == 2.0
    assert float(d1.demand[0]) == 0.0 and float(d1.unmet[0]) == 0.0
    # costs on day 0: holding h*0, shortage b*3
    assert d0.holding_cost == 0.0
    assert d0.shortage_cost == pytest.approx(3.0 * 3.0)


# --------------------------------------------------------------------------- #
# T3: capacity -- projection yields sum == K_g exactly; gate rejects;
#     strict engine mode raises instead of silently repairing
# --------------------------------------------------------------------------- #

def test_t3_projection_exact_sum(cfg):
    w = np.ones(2)
    q = np.array([3.0, 4.0])
    k_g = 5.0
    q_proj = project_orders(q, w, k_g)
    assert weighted_sum(q_proj, w) == k_g
    np.testing.assert_allclose(q_proj, np.array([15.0 / 7.0, 20.0 / 7.0]))
    # negative components are clipped before scaling
    q2 = project_orders(np.array([-1.0, 4.0]), w, 2.0)
    assert weighted_sum(q2, w) == 2.0
    assert q2[0] == 0.0
    # feasible input passes through (after clipping) unchanged
    q3 = project_orders(np.array([1.0, 2.0]), w, 5.0)
    np.testing.assert_array_equal(q3, np.array([1.0, 2.0]))


def test_t3_projection_invariant_fuzz(cfg):
    # invariant: sum(w*q) <= K_g strictly; shortfall at most 2 ulp(K_g),
    # always far inside binding_epsilon so the day registers as binding
    rng = spawn_stream(999, 0, 0)
    for _ in range(2000):
        n = int(rng.integers(1, 17))
        q = rng.uniform(-2.0, 50.0, size=n)
        w = np.ones(n)
        k_g = float(rng.uniform(0.1, 200.0))
        q_proj = project_orders(q, w, k_g)
        s = weighted_sum(q_proj, w)
        assert s <= k_g
        if weighted_sum(np.maximum(q, 0.0), w) > k_g:
            assert k_g - s <= 2 * np.spacing(k_g)
            assert s >= k_g - cfg.binding_epsilon


def test_t3_gate_rejects(cfg):
    w = np.ones(2)
    res_bad = gate_check(np.array([3.0, 4.0]), w, 5.0, cfg.binding_epsilon)
    assert not res_bad.accepted
    assert res_bad.violation_amount == pytest.approx(2.0)
    res_ok = gate_check(np.array([2.0, 3.0]), w, 5.0, cfg.binding_epsilon)
    assert res_ok.accepted and res_ok.violation_amount == 0.0


def test_t3_strict_mode_raises_and_project_mode_records(cfg):
    n_sku, lead = 2, cfg.lead_time_core
    demand = np.zeros((1, n_sku))
    params = make_params(n_sku)
    policy = FixedSchedulePolicy({0: np.array([3.0, 4.0])}, n_sku)
    with pytest.raises(EngineError):
        run_window(
            zero_state(n_sku, lead), demand, np.empty((0, n_sku)), policy,
            params, k_g=5.0, cfg=cfg, t0_idx=0,
        )
    _, days = run_window(
        zero_state(n_sku, lead), demand, np.empty((0, n_sku)), policy,
        params, k_g=5.0, cfg=cfg, t0_idx=0,
        on_violation=ViolationHandling.PROJECT,
    )
    d0 = days[0]
    assert d0.projection_applied
    assert d0.violation_amount == pytest.approx(2.0)
    np.testing.assert_array_equal(d0.order_raw, np.array([3.0, 4.0]))
    assert weighted_sum(d0.order, params.w) == 5.0
    assert d0.binding


def test_t3_within_tolerance_not_repaired(cfg):
    # violations <= feasibility_tol_rel * K_g are accepted as submitted
    n_sku, lead = 1, cfg.lead_time_core
    k_g = 10.0
    q_val = k_g + 0.5 * cfg.feasibility_tol_rel * k_g
    policy = FixedSchedulePolicy({0: np.array([q_val])}, n_sku)
    _, days = run_window(
        zero_state(n_sku, lead), np.zeros((1, n_sku)), np.empty((0, n_sku)),
        policy, make_params(n_sku), k_g=k_g, cfg=cfg, t0_idx=0,
    )
    d0 = days[0]
    assert not d0.projection_applied
    assert float(d0.order[0]) == q_val
    assert d0.violation_amount > 0.0


# --------------------------------------------------------------------------- #
# T7: determinism -- identical seeds and settings reproduce episode costs exactly
# --------------------------------------------------------------------------- #

def run_noisy_episode(cfg, fold: int, bundle_idx: int) -> tuple[float, float]:
    n_sku, lead = 4, cfg.lead_time_core
    demand_rng = spawn_stream(cfg.seeds.eval, fold, bundle_idx)
    demand = demand_rng.poisson(3.0, size=(cfg.test_window_days, n_sku)).astype(np.float64)
    params = make_params(n_sku)
    recent_mean = demand[:7].mean(axis=0)
    state = init_warmup_state(recent_mean, lead, cfg)
    policy = SeededNoisePolicy(cfg.seeds.eval, fold, bundle_idx, scale=4.0)
    final, days = run_window(
        state, demand, np.empty((0, n_sku)), policy, params,
        k_g=12.0, cfg=cfg, t0_idx=0,
    )
    window_cost = sum(r.holding_cost + r.shortage_cost for r in days)
    return window_cost, terminal_cost(final, params.h)


def test_t7_determinism(cfg):
    a = run_noisy_episode(cfg, fold=1, bundle_idx=3)
    b = run_noisy_episode(cfg, fold=1, bundle_idx=3)
    assert a == b  # exact float equality, incl. terminal valuation


def test_t7_stream_independence_of_execution_order(cfg):
    # running bundle 5 before or after bundle 2 must not change either stream
    first = (run_noisy_episode(cfg, 1, 2), run_noisy_episode(cfg, 1, 5))
    second = (run_noisy_episode(cfg, 1, 5), run_noisy_episode(cfg, 1, 2))
    assert first[0] == second[1]
    assert first[1] == second[0]


# --------------------------------------------------------------------------- #
# Terminal valuation: post-transition form equals pre-transition form
# (review decision 4; supports T5 later)
# --------------------------------------------------------------------------- #

def test_terminal_equivalence(cfg):
    n_sku, lead = 2, cfg.lead_time_core
    params = make_params(n_sku, h=0.7)
    state = BundleState(
        inv=np.array([4.0, 1.0]),
        pipeline=np.array([[2.0, 1.0], [0.5, 3.0]]),
    )
    demand = np.array([[1.0, 2.0]])
    q_last = np.array([2.5, 0.5])
    policy = FixedSchedulePolicy({0: q_last}, n_sku)
    pipeline_before = state.pipeline.copy()
    final, days = run_window(
        state, demand, np.empty((0, n_sku)), policy, params,
        k_g=10.0, cfg=cfg, t0_idx=0,
    )
    inv_eod = days[0].inv_eod
    pre_transition_total = inv_eod + pipeline_before.sum(axis=1) + q_last
    expected = float(params.h @ pre_transition_total)
    assert terminal_cost(final, params.h) == pytest.approx(expected, rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Guard: engine rejects malformed policy output
# --------------------------------------------------------------------------- #

def test_negative_order_rejected(cfg):
    n_sku, lead = 1, cfg.lead_time_core
    policy = FixedSchedulePolicy({0: np.array([-0.1])}, n_sku)
    with pytest.raises(EngineError):
        run_window(
            zero_state(n_sku, lead), np.zeros((1, n_sku)), np.empty((0, n_sku)),
            policy, make_params(n_sku), k_g=5.0, cfg=cfg, t0_idx=0,
        )


def test_warmup_init_state(cfg):
    mean = np.array([2.0, 4.0])
    st = init_warmup_state(mean, cfg.lead_time_core, cfg)
    np.testing.assert_array_equal(st.inv, np.array([4.0, 8.0]))
    assert st.pipeline.shape == (2, cfg.lead_time_core)
    np.testing.assert_array_equal(st.pipeline[:, 0], mean)
    np.testing.assert_array_equal(st.pipeline[:, 1], mean)
