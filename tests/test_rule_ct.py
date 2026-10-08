"""Tests for the L0 rule CT policy and the A4-SL selection rule (2-H)."""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.policies.rule_ct import RuleControlTower
from core_pipeline.simulator.engine import (
    BundleState,
    Observation,
    SkuParams,
    load_sim_config,
    run_window,
    spawn_stream,
    weighted_sum,
)
from scripts.run_asis_diagnosis import select_a4_sl

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


def make_obs(hist, inv, pipeline, k_g):
    n_sku = hist.shape[1]
    return Observation(
        t_idx=hist.shape[0] - 1, demand_hist=hist,
        inv=np.asarray(inv, dtype=float), pipeline=np.asarray(pipeline, dtype=float),
        k_g=k_g, h=np.ones(n_sku), b=np.ones(n_sku) * 3.0, w=np.ones(n_sku),
    )


# --------------------------------------------------------------------------- #
# L0 (s, S) behaviour
# --------------------------------------------------------------------------- #

def test_l0_orders_only_below_reorder_point():
    hist = np.tile(np.array([[2.0, 2.0]]), (30, 1))      # MA7 = 2 -> s=6, S=14
    pol = RuleControlTower()
    # SKU0: IP = 4 < 6 -> order 14 - 4 = 10; SKU1: IP = 8 >= 6 -> no order
    obs = make_obs(hist, inv=[1.0, 5.0], pipeline=[[2.0, 1.0], [2.0, 1.0]], k_g=100.0)
    q = pol.decide(obs)
    np.testing.assert_allclose(q, np.array([10.0, 0.0]))


def test_l0_ma_includes_zero_days():
    hist = np.zeros((30, 1))
    hist[-7:, 0] = [7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # MA7 = 1 -> s=3, S=7
    pol = RuleControlTower()
    obs = make_obs(hist, inv=[0.0], pipeline=[[0.0, 0.0]], k_g=100.0)
    np.testing.assert_allclose(pol.decide(obs), np.array([7.0]))


def test_l0_capacity_internal_projection(cfg):
    hist = np.tile(np.array([[4.0, 4.0]]), (30, 1))      # s=12, S=28
    pol = RuleControlTower()
    obs = make_obs(hist, inv=[0.0, 0.0], pipeline=np.zeros((2, 2)), k_g=10.0)
    q = pol.decide(obs)
    assert weighted_sum(q, obs.w) == 10.0                # radial scaling to K_g


def test_l0_intermittent_vs_daily(cfg):
    # once restocked above s, L0 stays silent while demand drains IP
    n_sku = 1
    hist = np.tile(np.array([[2.0]]), (60, 1))
    demand = np.tile(np.array([[2.0]]), (10, 1))
    state = BundleState(inv=np.array([14.0]), pipeline=np.zeros((1, cfg.lead_time_core)))
    params = SkuParams(series_ids=("S0",), h=np.ones(1), b=np.ones(1) * 3.0,
                       w=np.ones(1))
    _, days = run_window(state, demand, hist, RuleControlTower(), params,
                         k_g=100.0, cfg=cfg, t0_idx=60)
    orders = np.array([float(r.order[0]) for r in days])
    assert np.sum(orders == 0.0) >= 3                    # silent days exist
    assert np.sum(orders > 0.0) >= 1                     # and reorders happen


def test_l0_feasible_in_strict_engine(cfg):
    rng = spawn_stream(cfg.seeds.eval, 11, 11)
    n_sku = 6
    hist = rng.poisson(3.0, size=(100, n_sku)).astype(float)
    demand = rng.poisson(3.0, size=(28, n_sku)).astype(float)
    state = BundleState(inv=np.full(n_sku, 4.0),
                        pipeline=np.full((n_sku, cfg.lead_time_core), 2.0))
    params = SkuParams(series_ids=tuple(f"S{i}" for i in range(n_sku)),
                       h=np.ones(n_sku), b=np.ones(n_sku) * 5.0, w=np.ones(n_sku))
    _, days = run_window(state, demand, hist, RuleControlTower(), params,
                         k_g=8.0, cfg=cfg, t0_idx=100)
    assert all(not r.projection_applied for r in days)
    assert any(r.binding for r in days)


# --------------------------------------------------------------------------- #
# A4-SL selection rule (Track B amendment)
# --------------------------------------------------------------------------- #

def cand(r, tau, alloc, cost, csl):
    return {"r": r, "tau": tau, "allocation": alloc,
            "validation_cost": cost, "csl_sku_day": csl}


def test_a4sl_min_cost_among_qualifying():
    cands = [
        cand(3, 0.50, "proportional", 100.0, 0.73),
        cand(3, 5 / 6, "proportional", 251.0, 0.903),
        cand(3, 0.85, "proportional", 263.0, 0.910),
        cand(3, 5 / 6, "b_priority", 259.0, 0.904),
    ]
    sel = select_a4_sl(cands, 3)
    assert sel["tau"] == 5 / 6 and sel["allocation"] == "proportional"


def test_a4sl_tie_break_proportional_then_smaller_tau():
    cands = [
        cand(5, 0.90, "b_priority", 300.0, 0.93),
        cand(5, 0.90, "proportional", 300.0, 0.93),
        cand(5, 0.85, "proportional", 300.0, 0.92),
    ]
    sel = select_a4_sl(cands, 5)
    assert sel["allocation"] == "proportional" and sel["tau"] == 0.85


def test_a4sl_none_qualifying():
    cands = [cand(3, 0.5, "proportional", 100.0, 0.73)]
    assert select_a4_sl(cands, 3) is None
