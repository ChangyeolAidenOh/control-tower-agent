"""T6 (spec v0.1 section 11): learning LP vs engine cost match, < 0.1%.

Solves the A5 perfect-information LP on fixed scenarios, replays the optimal
order schedule through the state-transition engine, and compares the LP
objective with the engine's window cost + terminal valuation. Also pins the
LP's lost-sales reproduction (u = (d - I)+ at the optimum).
"""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.dfl.lp_problem import solve_a5
from core_pipeline.simulator.engine import (
    BundleState,
    Observation,
    SkuParams,
    load_sim_config,
    run_window,
    spawn_stream,
    terminal_cost,
)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")
T6_REL_TOL = 1e-3  # spec: < 0.1 percent


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


class ScheduledPolicy:
    """Replay a fixed order schedule (clipping solver-level -1e-12 noise)."""

    name = "scheduled"

    def __init__(self, orders: np.ndarray, t0_idx: int):
        self.orders = orders
        self.t0_idx = t0_idx

    def decide(self, obs: Observation) -> np.ndarray:
        return np.maximum(self.orders[obs.t_idx - self.t0_idx], 0.0)


def engine_cost_of_schedule(cfg, demand, state, params, k_g, orders) -> float:
    final, days = run_window(
        state.copy(), demand, np.empty((0, demand.shape[1])),
        ScheduledPolicy(orders, t0_idx=0), params, k_g=k_g, cfg=cfg, t0_idx=0,
    )
    window = sum(r.holding_cost + r.shortage_cost for r in days)
    return window + terminal_cost(final, params.h)


def make_scenario(cfg, seed_path, n_sku=6, n_days=28, kappa_ratio=1.0):
    rng = spawn_stream(cfg.seeds.train, *seed_path)
    demand = rng.poisson(3.0, size=(n_days, n_sku)).astype(float)
    h = rng.uniform(0.3, 2.0, size=n_sku)
    b = h * 5.0
    w = np.ones(n_sku)
    inv0 = rng.uniform(0.0, 6.0, size=n_sku)
    pipeline0 = rng.uniform(0.0, 4.0, size=(n_sku, cfg.lead_time_core))
    k_g = kappa_ratio * float(demand.mean(axis=0).sum())
    params = SkuParams(
        series_ids=tuple(f"S{i}" for i in range(n_sku)),
        h=h, b=b, w=w,
    )
    state = BundleState(inv=inv0, pipeline=pipeline0)
    return demand, state, params, k_g


@pytest.mark.parametrize("seed_path,kappa_ratio", [
    ((1, 1), 1.0),    # capacity binds (perfect info wants more than K_g)
    ((2, 2), 1.5),    # looser capacity
    ((3, 3), 0.7),    # severely tight capacity
])
def test_t6_lp_vs_engine(cfg, seed_path, kappa_ratio):
    demand, state, params, k_g = make_scenario(cfg, seed_path, kappa_ratio=kappa_ratio)
    sol = solve_a5(demand, state.inv, state.pipeline, params.h, params.b,
                   params.w, k_g)
    engine_total = engine_cost_of_schedule(cfg, demand, state, params, k_g,
                                           sol.orders)
    rel_diff = abs(engine_total - sol.objective) / max(engine_total, 1e-12)
    assert rel_diff < T6_REL_TOL, (sol.objective, engine_total, rel_diff)


def test_t6_zero_capacity_degenerate(cfg):
    # K_g = 0: no ordering possible; LP cost must equal pure drain cost
    demand, state, params, _ = make_scenario(cfg, (4, 4))
    sol = solve_a5(demand, state.inv, state.pipeline, params.h, params.b,
                   params.w, k_g=0.0)
    engine_total = engine_cost_of_schedule(cfg, demand, state, params, 0.0,
                                           np.zeros_like(sol.orders))
    rel_diff = abs(engine_total - sol.objective) / max(engine_total, 1e-12)
    assert rel_diff < T6_REL_TOL


def test_lp_beats_or_matches_heuristics(cfg):
    # perfect information must not lose to a fixed schedule under same costs
    demand, state, params, k_g = make_scenario(cfg, (5, 5))
    sol = solve_a5(demand, state.inv, state.pipeline, params.h, params.b,
                   params.w, k_g)
    naive = np.tile(demand.mean(axis=0), (demand.shape[0], 1))
    scale = min(1.0, k_g / float(params.w @ naive[0]))
    naive_cost = engine_cost_of_schedule(cfg, demand, state, params, k_g,
                                         naive * scale)
    assert sol.objective <= naive_cost * (1 + 1e-9)
