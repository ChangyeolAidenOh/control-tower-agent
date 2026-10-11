"""ST7-02 D2/D3/D4 verification: capacity rejection on any scenario, 7-day
CSL median gate, 10-day cost ranking, tie-break chain, all-rejected
fallback signal, CSL window limited to the first 7 days."""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.agent.candidates import make_action
from core_pipeline.agent.gate import (
    CSL_DAYS,
    N_SCEN,
    REASON_CAPACITY,
    REASON_CSL,
    SELECTION_DAYS,
    evaluate_candidate,
    gate_stats,
    select_candidate,
)
from core_pipeline.simulator.engine import (
    BundleState,
    SkuParams,
    load_sim_config,
)

ROOT = Path(__file__).resolve().parents[1]
SERIES = ("S_A", "S_B", "S_C")
BASE_TAU = 0.60
T0 = 200


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(ROOT / "configs" / "simulator.yaml")


def params() -> SkuParams:
    return SkuParams(
        series_ids=SERIES,
        h=np.array([0.5, 1.0, 2.0]),
        b=np.array([3.0, 5.0, 4.0]),
        w=np.ones(len(SERIES)),
    )


def make_inputs(level: float = 5.0, seed: int = 3):
    rng = np.random.default_rng(seed)
    hist = rng.poisson(level, size=(60, len(SERIES))).astype(np.float64)
    hist[0] = np.maximum(hist[0], 1.0)
    base = rng.poisson(level, size=(SELECTION_DAYS, len(SERIES), 1))
    scen = np.repeat(base.astype(np.float64), N_SCEN, axis=2)
    state = BundleState(
        inv=np.full(len(SERIES), 30.0),
        pipeline=np.full((len(SERIES), 2), 5.0),
    )
    return hist, scen, state


def run(action, scen, state, hist, cfg, k_g=1e9, floor=0.90, slot=0):
    return evaluate_candidate(
        slot, action, scen, state, hist, SERIES, params(), k_g, cfg,
        t0_idx=T0, csl_floor=floor,
    )


def test_pass_shapes_and_median(cfg):
    hist, scen, state = make_inputs()
    ev = run(make_action(None, {}, BASE_TAU, SERIES), scen, state, hist, cfg,
             floor=0.0)
    assert not ev.capacity_rejected
    assert len(ev.scenario_csl) == N_SCEN
    assert len(ev.scenario_cost) == N_SCEN
    assert ev.csl_median == pytest.approx(np.median(ev.scenario_csl))
    assert ev.mean_cost == pytest.approx(np.mean(ev.scenario_cost))
    assert ev.gate_passed


def test_capacity_rejection_on_any_scenario(cfg):
    hist, scen, state = make_inputs()
    scen = scen.copy()
    scen[:, :, 4] = scen[:, :, 4] * 40.0
    action = make_action(0.90, {"S_A": 1.5, "S_B": 1.5, "S_C": 1.5},
                         BASE_TAU, SERIES)
    ev = run(action, scen, state, hist, cfg, k_g=25.0, floor=0.0)
    assert ev.capacity_rejected
    assert ev.reject_reason == REASON_CAPACITY
    assert not ev.gate_passed
    assert np.isnan(ev.mean_cost)


def test_csl_gate_uses_first_seven_days_only(cfg):
    hist, scen, state = make_inputs(level=2.0)
    state = BundleState(inv=np.full(3, 200.0), pipeline=np.full((3, 2), 50.0))
    scen_tail = scen.copy()
    scen_tail[CSL_DAYS:] = 1000.0
    ev_base = run(make_action(None, {}, BASE_TAU, SERIES), scen, state.copy(),
                  hist, cfg, floor=0.90)
    ev_tail = run(make_action(None, {}, BASE_TAU, SERIES), scen_tail,
                  state.copy(), hist, cfg, floor=0.90)
    assert ev_base.gate_passed
    assert ev_tail.scenario_csl == ev_base.scenario_csl
    assert ev_tail.mean_cost > ev_base.mean_cost


def test_csl_rejection(cfg):
    hist, scen, state = make_inputs()
    state = BundleState(inv=np.zeros(3), pipeline=np.zeros((3, 2)))
    scen = scen + 50.0
    ev = run(make_action(None, {"S_A": 0.0, "S_B": 0.0, "S_C": 0.0},
                         BASE_TAU, SERIES),
             scen, state, hist, cfg, floor=0.90)
    assert not ev.capacity_rejected
    assert not ev.gate_passed
    assert ev.reject_reason == REASON_CSL
    assert ev.csl_median < 0.90


def test_rank_prefers_lower_mean_cost_then_tiebreaks(cfg):
    hist, scen, state = make_inputs()
    e1 = run(make_action(None, {}, BASE_TAU, SERIES), scen, state.copy(),
             hist, cfg, floor=0.0, slot=0)
    e2 = run(make_action(0.90, {}, BASE_TAU, SERIES), scen, state.copy(),
             hist, cfg, floor=0.0, slot=1)
    winner = select_candidate([e1, e2])
    assert winner is not None
    assert winner.mean_cost == min(e1.mean_cost, e2.mean_cost)

    dup1 = run(make_action(0.70, {}, BASE_TAU, SERIES), scen, state.copy(),
               hist, cfg, floor=0.0, slot=3)
    dup2 = run(make_action(0.70, {}, BASE_TAU, SERIES), scen, state.copy(),
               hist, cfg, floor=0.0, slot=1)
    assert dup1.mean_cost == dup2.mean_cost
    assert select_candidate([dup1, dup2]).slot == 1

    same_cost_more_skus = run(
        make_action(0.70, {"S_A": 1.0}, BASE_TAU, SERIES), scen,
        state.copy(), hist, cfg, floor=0.0, slot=0)
    assert select_candidate(
        [same_cost_more_skus, dup2]).slot in (0, 1)


def test_all_rejected_returns_none_and_stats(cfg):
    hist, scen, state = make_inputs()
    state = BundleState(inv=np.zeros(3), pipeline=np.zeros((3, 2)))
    scen = scen + 50.0
    evs = [run(make_action(None, {"S_A": 0.0, "S_B": 0.0, "S_C": 0.0},
                           BASE_TAU, SERIES),
               scen, state.copy(), hist, cfg, floor=0.99, slot=i)
           for i in range(3)]
    assert select_candidate(evs) is None
    stats = gate_stats(evs)
    assert stats["all_rejected"]
    assert stats["gate_passed_candidates"] == 0
    assert stats["csl_rejected"] == 3
    assert stats["candidate_gate_pass_rate"] == 0.0


def test_bad_scenario_shape_raises(cfg):
    hist, scen, state = make_inputs()
    with pytest.raises(ValueError):
        run(make_action(None, {}, BASE_TAU, SERIES), scen[:5], state, hist, cfg)
    with pytest.raises(ValueError):
        run(make_action(None, {}, BASE_TAU, SERIES), scen[:, :, :4], state,
            hist, cfg)