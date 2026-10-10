"""ST5-04 boundary tests: decision-time 29-day mapping on solve_a5.

Torch-free by design -- runs inside the main suite and pins the shared
time axis at the code level (reviewer P0): first-order arrival D+3,
pipeline slot arrivals, tail orders, day-0 bookkeeping.
"""

import numpy as np
import pytest

from core_pipeline.dfl.lp_problem import solve_a5
from core_pipeline.dfl.lp_window import to_lp_window


def _solve(forecast_28, inv_eod, pipeline, h, b, w, k_g):
    demand_29, inv0, pipe0 = to_lp_window(forecast_28, inv_eod, pipeline)
    return solve_a5(demand_29, inv0, pipe0, h, b, w, k_g)


def _base(n_sku=1, lead=2):
    return (np.zeros(n_sku), np.zeros((n_sku, lead)), np.ones(n_sku),
            5.0 * np.ones(n_sku), np.ones(n_sku), 100.0)


def test_demand_at_t_plus_2_cannot_be_served_by_first_order():
    inv_eod, pipe, h, b, w, k_g = _base()
    f = np.zeros((28, 1))
    f[1, 0] = 1.0  # day t+2 = internal day 2; q[0] lands day 3
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g)
    assert sol.objective == pytest.approx(5.0, rel=1e-6)


def test_demand_at_t_plus_3_served_by_first_order():
    inv_eod, pipe, h, b, w, k_g = _base()
    f = np.zeros((28, 1))
    f[2, 0] = 1.0  # day t+3 = internal day 3
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g)
    assert sol.objective == pytest.approx(0.0, abs=1e-6)
    assert sol.orders[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_pipeline_slot_0_arrives_t_plus_1():
    inv_eod, pipe, h, b, w, k_g = _base()
    pipe = pipe.copy()
    pipe[0, 0] = 1.0
    f = np.zeros((28, 1))
    f[0, 0] = 1.0  # day t+1 = internal day 1
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g)
    assert sol.objective == pytest.approx(0.0, abs=1e-6)


def test_pipeline_slot_1_arrives_t_plus_2():
    inv_eod, pipe, h, b, w, k_g = _base()
    pipe = pipe.copy()
    pipe[0, 1] = 1.0
    f = np.zeros((28, 1))
    f[1, 0] = 1.0  # day t+2 = internal day 2
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g)
    assert sol.objective == pytest.approx(0.0, abs=1e-6)


def test_tail_orders_zero_at_optimum():
    rng = np.random.default_rng(0)
    n_sku = 4
    f = rng.gamma(4.0, 3.0, (28, n_sku))
    inv_eod = rng.uniform(0.0, 10.0, n_sku)
    pipe = rng.uniform(0.0, 10.0, (n_sku, 2))
    h = rng.uniform(0.5, 2.0, n_sku)
    b = 5.0 * h
    w = np.ones(n_sku)
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g=float(f.sum(axis=1).max() * 2))
    # q[26..28] arrive on days 29/30/31, outside the window: with positive h
    # and terminal valuation the optimum sets them to zero.
    assert np.abs(sol.orders[26:]).max() < 1e-6


def test_day0_constant_cost_included():
    n_sku = 1
    inv_eod = np.array([3.0])
    pipe = np.zeros((n_sku, 2))
    h = np.ones(n_sku)
    b = 5.0 * np.ones(n_sku)
    w = np.ones(n_sku)
    f = np.zeros((28, n_sku))
    sol = _solve(f, inv_eod, pipe, h, b, w, k_g=100.0)
    # Zero demand, zero orders optimal: inv c is held on each of the 29
    # internal days and valued once at terminal = 30 h c. Pins that day 0's
    # constant h x inv_eod charge is inside the LP (it cancels in regret).
    assert sol.objective == pytest.approx(30.0 * 3.0, rel=1e-8)


def test_forecast_shape_guard():
    inv_eod, pipe, _, _, _, _ = _base()
    with pytest.raises(ValueError):
        to_lp_window(np.zeros((27, 1)), inv_eod, pipe)
