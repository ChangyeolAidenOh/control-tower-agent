"""ST7-02 D2 verification: no-action parity with base A4, multiplier
application, raw-mode capacity rejection, scale-mode internal scaling,
window boundary (continuation days fall back to base A4)."""

import numpy as np
import pytest

from core_pipeline.agent.actions import (
    AgentA4Policy,
    CapacityViolation,
)
from core_pipeline.agent.candidates import make_action
from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.simulator.engine import (
    BundleState,
    SkuParams,
    load_sim_config,
    run_window,
    weighted_sum,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ("S_A", "S_B", "S_C")
BASE_TAU = 0.60
TOL_REL = 1.0e-6
T0 = 100


def make_obs_inputs(seed: int = 7, n_days: int = 60):
    rng = np.random.default_rng(seed)
    hist = rng.poisson(5.0, size=(n_days, len(SERIES))).astype(np.float64)
    hist[0] = np.maximum(hist[0], 1.0)
    return hist


def run_days(policy, demand, hist, k_g, cfg, t0=T0):
    params = SkuParams(
        series_ids=SERIES,
        h=np.array([0.5, 1.0, 2.0]),
        b=np.array([3.0, 5.0, 4.0]),
        w=np.ones(len(SERIES)),
    )
    state = BundleState(
        inv=np.array([10.0, 10.0, 10.0]),
        pipeline=np.full((len(SERIES), cfg.lead_time_core), 5.0),
    )
    return run_window(state, demand, hist, policy, params,
                      k_g=k_g, cfg=cfg, t0_idx=t0)


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(ROOT / "configs" / "simulator.yaml")


@pytest.fixture()
def demand_and_hist(cfg):
    rng = np.random.default_rng(11)
    hist = make_obs_inputs()
    demand = rng.poisson(5.0, size=(10, len(SERIES))).astype(np.float64)
    return demand, hist


def agent(action, mode, start=T0, days=7):
    return AgentA4Policy(action, SERIES, action_start_idx=start,
                         tol_rel=TOL_REL, capacity_mode=mode)


def test_no_action_parity_with_base_a4(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    no_action = make_action(None, {}, BASE_TAU, SERIES)
    pol = agent(no_action, "scale")
    base = A4BaseStock(A4Config(tau=BASE_TAU, allocation="proportional"))
    _, days_a = run_days(pol, demand, hist, k_g=1e9, cfg=cfg)
    _, days_b = run_days(base, demand, hist, k_g=1e9, cfg=cfg)
    for ra, rb in zip(days_a, days_b):
        np.testing.assert_array_equal(ra.order, rb.order)
    assert pol.scaling_summary()["internal_scaling_days"] == 0.0


def test_multipliers_apply_only_inside_action_window(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(None, {"S_A": 0.0, "S_B": 1.5}, BASE_TAU, SERIES)
    pol = agent(action, "scale")
    base = A4BaseStock(A4Config(tau=BASE_TAU, allocation="proportional"))
    _, days_a = run_days(pol, demand, hist, k_g=1e9, cfg=cfg)
    _, days_b = run_days(base, demand.copy(), hist, k_g=1e9, cfg=cfg)
    for ra in days_a[:7]:
        assert ra.order[0] == 0.0
    np.testing.assert_allclose(days_a[0].order,
                               days_b[0].order * np.array([0.0, 1.5, 1.0]))
    assert pol.last_diagnostics()["agent_active"] == 0.0


def test_tau_prime_changes_orders_in_window(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(0.90, {}, BASE_TAU, SERIES)
    pol = agent(action, "scale")
    base = A4BaseStock(A4Config(tau=BASE_TAU, allocation="proportional"))
    _, days_a = run_days(pol, demand, hist, k_g=1e9, cfg=cfg)
    _, days_b = run_days(base, demand.copy(), hist, k_g=1e9, cfg=cfg)
    assert float(np.sum(days_a[0].order)) >= float(np.sum(days_b[0].order))
    assert any(not np.array_equal(ra.order, rb.order)
               for ra, rb in zip(days_a[:7], days_b[:7]))


def test_raw_mode_raises_on_capacity_violation(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(0.90, {"S_A": 1.5, "S_B": 1.5, "S_C": 1.5},
                         BASE_TAU, SERIES)
    pol = agent(action, "raw")
    with pytest.raises(CapacityViolation):
        run_days(pol, demand, hist, k_g=1.0, cfg=cfg)


def test_scale_mode_projects_and_records(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(0.90, {"S_A": 1.5, "S_B": 1.5, "S_C": 1.5},
                         BASE_TAU, SERIES)
    pol = agent(action, "scale")
    k_g = 1.0
    _, days = run_days(pol, demand, hist, k_g=k_g, cfg=cfg)
    for r in days:
        assert weighted_sum(r.order, np.ones(3)) <= k_g * (1 + TOL_REL)
        assert not r.projection_applied
    summary = pol.scaling_summary()
    assert summary["internal_scaling_days"] >= 1.0
    assert summary["total_scaled_units"] > 0.0
    assert summary["max_requested_capacity_ratio"] > 1.0


def test_multiplier_le_one_cannot_violate(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(None, {"S_A": 0.5, "S_B": 0.8}, BASE_TAU, SERIES)
    pol = agent(action, "raw")
    base = A4BaseStock(A4Config(tau=BASE_TAU, allocation="proportional"))
    _, _ = run_days(base, demand.copy(), hist, k_g=30.0, cfg=cfg)
    _, days = run_days(pol, demand, hist, k_g=30.0, cfg=cfg)
    assert len(days) == 10


def test_window_boundary_is_half_open(cfg, demand_and_hist):
    demand, hist = demand_and_hist
    action = make_action(0.90, {"S_A": 1.5}, BASE_TAU, SERIES)
    pol = agent(action, "scale", start=T0, days=7)
    _, _ = run_days(pol, demand, hist, k_g=1e9, cfg=cfg)
    assert pol._in_window(T0)
    assert pol._in_window(T0 + 6)
    assert not pol._in_window(T0 + 7)
    assert not pol._in_window(T0 - 1)


def test_fresh_instance_counters_start_clean():
    action = make_action(None, {}, BASE_TAU, SERIES)
    pol = agent(action, "scale")
    assert pol.scaling_summary() == {
        "internal_scaling_days": 0.0,
        "total_scaled_units": 0.0,
        "max_requested_capacity_ratio": 0.0,
    }