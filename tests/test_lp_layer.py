"""ST5-03 (8): layer-level tests for the B training LP path.

Torch-importing file: run in its own process (same rule as
test_tsmixer_contract.py), excluded from the main suite command.
"""

import numpy as np
import pytest
import torch

from core_pipeline.dfl.lp_layer import (
    SOLVER_ARGS,
    LpLayerCache,
    build_bundle_layer,
    downstream_cost,
    oracle_cost,
)
from core_pipeline.dfl.lp_problem import solve_a5

torch.set_default_dtype(torch.float64)


def _instance(seed, n_sku=6, horizon=28, lead=2, r=5.0):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5.0, 25.0, n_sku)
    y = rng.gamma(4.0, np.tile(mean / 4.0, (horizon, 1)))
    d_hat = np.clip(y + rng.normal(0.0, 0.2 * mean, (horizon, n_sku)), 0.01, None)
    inv0 = mean * 1.5
    pipeline0 = np.tile((mean * 0.8)[:, None], (1, lead))
    h = rng.uniform(0.5, 2.0, n_sku)
    b = r * h
    w = np.ones(n_sku)
    k_g = float(mean.sum())
    return y, d_hat, inv0, pipeline0, h, b, w, k_g


def _tt(*arrays):
    return tuple(torch.from_numpy(np.asarray(a, dtype=np.float64)) for a in arrays)


def test_layer_matches_solve_a5_objective_and_first_day():
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(0)
    layer = build_bundle_layer(h, b, w, k_g, d_hat.shape[0], pipe.shape[1])
    d_t, inv_t, pipe_t, h_t, b_t = _tt(d_hat, inv0, pipe, h, b)
    (q,) = layer(d_t, inv_t, pipe_t, solver_args=SOLVER_ARGS)
    cost = downstream_cost(q, d_t, inv_t, pipe_t, h_t, b_t).item()
    sol = solve_a5(d_hat, inv0, pipe, h, b, w, k_g)
    rel = abs(cost - sol.objective) / max(abs(sol.objective), 1e-9)
    assert rel < 1e-6
    first_day = float(np.abs(q.detach().numpy()[0] - sol.orders[0]).max())
    if first_day > 1e-4:
        print(f"degenerate optimum suspected: first-day diff {first_day:.3e} "
              f"with objective rel {rel:.1e}")


@pytest.mark.parametrize("seed", range(4))
def test_regret_nonnegative_and_oracle_parity(seed):
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(seed)
    layer = build_bundle_layer(h, b, w, k_g, d_hat.shape[0], pipe.shape[1])
    d_t, y_t, inv_t, pipe_t, h_t, b_t = _tt(d_hat, y, inv0, pipe, h, b)
    (q,) = layer(d_t, inv_t, pipe_t, solver_args=SOLVER_ARGS)
    cost = downstream_cost(q, y_t, inv_t, pipe_t, h_t, b_t).item()
    o_cost, _, lp_obj = oracle_cost(y, inv0, pipe, h, b, w, k_g)
    assert cost >= o_cost - 1e-6 * max(o_cost, 1.0)
    assert abs(o_cost - lp_obj) / max(abs(lp_obj), 1e-9) < 1e-6


def test_full_path_gradient_finite_and_nonzero():
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(1)
    layer = build_bundle_layer(h, b, w, k_g, d_hat.shape[0], pipe.shape[1])
    _, y_t, inv_t, pipe_t, h_t, b_t = _tt(d_hat, y, inv0, pipe, h, b)
    d_var = torch.from_numpy(d_hat).clone().requires_grad_(True)
    (q,) = layer(d_var, inv_t, pipe_t, solver_args=SOLVER_ARGS)
    downstream_cost(q, y_t, inv_t, pipe_t, h_t, b_t).backward()
    g = d_var.grad
    assert torch.isfinite(g).all()
    assert float(g.norm()) > 0.0


def test_order_arrives_d_plus_3():
    horizon, lead = 6, 2
    y = np.zeros((horizon, 1))
    y[3, 0] = 4.0
    q = np.zeros((horizon, 1))
    q[0, 0] = 4.0
    inv0 = np.zeros(1)
    pipe = np.zeros((1, lead))
    h = np.ones(1)
    b = 5.0 * np.ones(1)
    q_t, y_t, inv_t, pipe_t, h_t, b_t = _tt(q, y, inv0, pipe, h, b)
    cost = downstream_cost(q_t, y_t, inv_t, pipe_t, h_t, b_t).item()
    # Order day 0 arrives the morning of day 3 and serves the demand exactly:
    # no holding, no shortage, nothing undelivered.
    assert cost == pytest.approx(0.0, abs=1e-12)
    # Demand one day earlier: day 2 is a full stockout (b x 4), the arrival is
    # then held days 3-5 (h x 4 x 3) and valued once at terminal (h x 4).
    y2 = np.zeros((horizon, 1))
    y2[2, 0] = 4.0
    (y2_t,) = _tt(y2)
    cost2 = downstream_cost(q_t, y2_t, inv_t, pipe_t, h_t, b_t).item()
    assert cost2 == pytest.approx(36.0, abs=1e-10)


def test_terminal_charges_last_day_order():
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(2)
    horizon = d_hat.shape[0]
    q0 = np.zeros_like(d_hat)
    q1 = q0.copy()
    q1[horizon - 1, 0] = 1.0
    q0_t, q1_t, y_t, inv_t, pipe_t, h_t, b_t = _tt(q0, q1, y, inv0, pipe, h, b)
    c0 = downstream_cost(q0_t, y_t, inv_t, pipe_t, h_t, b_t).item()
    c1 = downstream_cost(q1_t, y_t, inv_t, pipe_t, h_t, b_t).item()
    assert c1 - c0 == pytest.approx(float(h[0]), rel=1e-12)


def test_layer_cache_compiles_once_per_key():
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(3, horizon=14)
    cache = LpLayerCache()
    layer_a = cache.get(("B01", 3), h, 3.0 * h, w, k_g, 14, 2)
    layer_b = cache.get(("B01", 3), h, 3.0 * h, w, k_g, 14, 2)
    layer_c = cache.get(("B01", 5), h, 5.0 * h, w, k_g, 14, 2)
    assert layer_a is layer_b
    assert layer_a is not layer_c
    assert len(cache) == 2


def test_downstream_matches_engine_path():
    """ST5-03 (8): the differentiable cost reproduces run_window's daily
    arrivals, shortages, holding and terminal valuation on a replayed plan."""
    from core_pipeline.simulator.engine import (
        BundleState,
        SkuParams,
        load_sim_config,
        run_window,
        terminal_cost,
    )

    n_sku, horizon, lead = 5, 12, 2
    y, d_hat, inv0, pipe, h, b, w, k_g = _instance(4, n_sku=n_sku, horizon=horizon)
    sol = solve_a5(y, inv0, pipe, h, b, w, k_g)
    q_plan = np.asarray(sol.orders, dtype=np.float64)

    class PlanReplay:
        name = "plan_replay"

        def __init__(self, plan, t0_idx):
            self.plan = plan
            self.t0_idx = t0_idx

        def decide(self, obs):
            return self.plan[obs.t_idx - self.t0_idx]

    t0_idx = 100
    cfg = load_sim_config("configs/simulator.yaml")
    params = SkuParams(series_ids=tuple(f"s{i}" for i in range(n_sku)),
                       h=h, b=b, w=w)
    final_state, results = run_window(
        BundleState(inv=inv0.copy(), pipeline=pipe.copy()),
        y, np.zeros((0, n_sku)), PlanReplay(q_plan, t0_idx),
        params, k_g, cfg, t0_idx=t0_idx)
    engine_cost = (sum(r.holding_cost + r.shortage_cost for r in results)
                   + terminal_cost(final_state, h))

    q_t, y_t, inv_t, pipe_t, h_t, b_t = _tt(q_plan, y, inv0, pipe, h, b)
    cost = downstream_cost(q_t, y_t, inv_t, pipe_t, h_t, b_t).item()
    assert abs(cost - engine_cost) / max(abs(engine_cost), 1e-9) < 1e-9

    # Day-level path: same opening inventory (arrivals applied), end-of-day
    # inventory, unmet demand and executed orders as the engine.
    inv = inv0.copy()
    for j, day in enumerate(results):
        if j == 0:
            inv_start = inv
        elif j <= lead:
            inv_start = inv + pipe[:, j - 1]
        else:
            inv_start = inv + q_plan[j - lead - 1]
        unmet = np.maximum(y[j] - inv_start, 0.0)
        inv_eod = np.maximum(inv_start - y[j], 0.0)
        assert np.abs(inv_start - day.inv_start).max() < 1e-9
        assert np.abs(inv_eod - day.inv_eod).max() < 1e-9
        assert np.abs(unmet - day.unmet).max() < 1e-9
        assert np.abs(q_plan[j] - day.order).max() < 1e-9
        inv = inv_eod
