"""ST6-08 (4): A3 SAA LP contract tests (shared q, per-scenario recourse)."""

import math

import numpy as np
import pytest

from core_pipeline.dfl.lp_problem import A3SaaProblem, solve_a5

H, LEAD = 29, 2


def nondegenerate_instance():
    """h/b differentiated + asymmetric demand -> unique optimum by design
    (ST6-08 (4)); first-day orders are hard-asserted on this instance."""
    h = np.array([1.0, 1.7, 2.3])
    b = np.array([3.0, 8.5, 16.1])
    w = np.ones(3)
    d = np.zeros((H, 3))
    d[3] = [5.0, 2.0, 7.0]
    d[6] = [1.0, 4.0, 2.0]
    return h, b, w, d


def test_single_scenario_degenerate_matches_solve_a5():
    h, b, w, d = nondegenerate_instance()
    k_g = 100.0
    inv0 = np.array([1.0, 0.0, 2.0])
    pipe = np.array([[0.5, 0.0], [0.0, 1.0], [0.0, 0.0]])
    prob = A3SaaProblem(h, b, w, k_g, H, LEAD, n_scen=1)
    saa = prob.solve(d[None, :, :], inv0, pipe)
    ref = solve_a5(d, inv0, pipe, h, b, w, k_g)
    assert saa.status == "optimal"
    assert abs(ref.objective) > 1.0
    rel = abs(saa.objective - ref.objective) / max(abs(ref.objective), 1e-9)
    assert rel < 1e-6
    assert np.abs(saa.orders[0] - ref.orders[0]).max() < 1e-5


def test_nine_replicas_equal_single():
    h, b, w, d = nondegenerate_instance()
    k_g = 100.0
    inv0 = np.array([1.0, 0.0, 2.0])
    pipe = np.array([[0.5, 0.0], [0.0, 1.0], [0.0, 0.0]])
    single = A3SaaProblem(h, b, w, k_g, H, LEAD, 1).solve(
        d[None, :, :], inv0, pipe)
    nine = A3SaaProblem(h, b, w, k_g, H, LEAD, 9).solve(
        np.repeat(d[None, :, :], 9, axis=0), inv0, pipe)
    assert single.status == nine.status == "optimal"
    assert abs(single.objective) > 1.0
    rel = abs(nine.objective - single.objective) / max(
        abs(single.objective), 1e-9)
    assert rel < 1e-6


def scenario_set():
    h, b, w, base = nondegenerate_instance()
    rng = np.random.default_rng(20261010)
    scen = np.stack([np.clip(base * (0.4 + 0.15 * s)
                             + rng.gamma(2.0, 0.5, base.shape) * (s % 3),
                             0.0, None)
                     for s in range(9)])
    scen[:, 0, :] = 0.0
    return h, b, w, scen


def test_scenario_permutation_invariance():
    h, b, w, scen = scenario_set()
    k_g = 60.0
    inv0 = np.zeros(3)
    pipe = np.zeros((3, LEAD))
    prob = A3SaaProblem(h, b, w, k_g, H, LEAD, 9)
    a = prob.solve(scen, inv0, pipe)
    perm = np.random.default_rng(7).permutation(9)
    bsol = prob.solve(scen[perm], inv0, pipe)
    rel = abs(a.objective - bsol.objective) / max(abs(a.objective), 1e-9)
    assert rel < 1e-8


def test_state_change_reflected():
    h, b, w, scen = scenario_set()
    k_g = 60.0
    prob = A3SaaProblem(h, b, w, k_g, H, LEAD, 9)
    empty = prob.solve(scen, np.zeros(3), np.zeros((3, LEAD)))
    stocked = prob.solve(scen, np.full(3, 50.0), np.full((3, LEAD), 10.0))
    assert empty.status == stocked.status == "optimal"
    assert float(stocked.orders[0].sum()) < float(empty.orders[0].sum())
    assert abs(stocked.objective - empty.objective) > 1e-6


def test_capacity_binding_shared_q():
    h, b, w, scen = scenario_set()
    k_g = 4.0
    prob = A3SaaProblem(h, b, w, k_g, H, LEAD, 9)
    sol = prob.solve(scen, np.zeros(3), np.zeros((3, LEAD)))
    assert sol.status == "optimal"
    sums = [math.fsum((w * np.clip(sol.orders[j], 0.0, None)).tolist())
            for j in range(H)]
    assert max(sums) <= k_g * (1.0 + 1e-6)
    assert max(sums) > k_g - 1e-3


def test_tail_orders_zero():
    h, b, w, scen = scenario_set()
    k_g = 60.0
    prob = A3SaaProblem(h, b, w, k_g, H, LEAD, 9)
    sol = prob.solve(scen, np.zeros(3), np.zeros((3, LEAD)))
    assert np.abs(sol.orders[26:, :]).max() < 1e-6