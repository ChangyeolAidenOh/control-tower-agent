"""ST7-02 D6 verification: no-action parity with the single 28-day A4 run
(state continuity + chaining identity, bit-level), weekly selection origins
and provider weeks, SelectionContext integrity (information blocking, state
isolation, prev-week summary), scenario-only selection, fallback
accounting, selected action changes realized orders."""

from pathlib import Path

import numpy as np
import pytest

from core_pipeline.agent.candidates import make_action
from core_pipeline.agent.weekly_runner import (
    N_WEEKS,
    WINDOW_DAYS,
    run_agent_window,
)
from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.simulator.engine import (
    BundleState,
    SkuParams,
    load_sim_config,
)
from core_pipeline.simulator.episode import run_policy_window

ROOT = Path(__file__).resolve().parents[1]
SERIES = ("S_A", "S_B", "S_C")
BASE_TAU = 0.60
T_START = 500


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


def make_world(seed: int = 9, level: float = 5.0):
    rng = np.random.default_rng(seed)
    base_hist = rng.poisson(level, size=(60, len(SERIES))).astype(np.float64)
    base_hist[0] = np.maximum(base_hist[0], 1.0)
    test = rng.poisson(level, size=(WINDOW_DAYS, len(SERIES))).astype(np.float64)
    scen = np.repeat(
        rng.poisson(level, size=(28, len(SERIES), 1)).astype(np.float64),
        9, axis=2)
    state = BundleState(
        inv=np.full(len(SERIES), 30.0),
        pipeline=np.full((len(SERIES), 2), 5.0),
    )
    return base_hist, test, scen, state


def forecast_recorder(scen, calls: list):
    def get(origin_idx: int):
        calls.append(origin_idx)
        return scen
    return get


def run(provider, scen, state, hist, test, cfg, k_g=40.0, floor=None,
        calls=None):
    calls = [] if calls is None else calls
    return run_agent_window(
        state, test, hist, provider, forecast_recorder(scen, calls),
        SERIES, params(), k_g, cfg,
        test_start_idx=T_START, fold=1, group_id="B99",
        base_tau=BASE_TAU, policy_name="agent_numeric",
        csl_floor=floor,
    ), calls


def test_no_action_parity_with_single_28day_a4_run(cfg):
    hist, test, scen, state = make_world()
    res, _ = run(lambda w, ctx: [], scen, state.copy(), hist, test, cfg)
    base = A4BaseStock(A4Config(tau=BASE_TAU, allocation="proportional"))
    ref = run_policy_window(
        state.copy(), test, hist, base, params(), 40.0, cfg,
        t0_idx=T_START, fold=1, group_id="B99",
    )
    assert res.n_days == ref.n_days == WINDOW_DAYS
    assert res.window_cost == pytest.approx(ref.window_cost, abs=1e-12)
    assert res.terminal == pytest.approx(ref.terminal, abs=1e-12)
    for ra, rb in zip(res.records, ref.records):
        assert ra.cost == pytest.approx(rb.cost, abs=1e-12)
        assert ra.csl == rb.csl
        assert ra.week == rb.week
    assert all(wk.fallback for wk in res.weeks)
    assert all(wk.gate["evaluated_candidates"] == 0 for wk in res.weeks)


def test_selection_origins_and_provider_weeks(cfg):
    hist, test, scen, state = make_world()
    weeks_seen = []

    def provider(w, ctx):
        weeks_seen.append(w)
        return []

    res, calls = run(provider, scen, state, hist, test, cfg)
    assert weeks_seen == [0, 1, 2, 3]
    assert calls == [T_START - 1 + 7 * w for w in range(N_WEEKS)]
    assert [wk.origin_idx for wk in res.weeks] == calls


def test_selection_context_integrity(cfg):
    hist, test, scen, state = make_world()
    contexts = []

    def provider(w, ctx):
        contexts.append(ctx)
        ctx.state.inv[:] = -999.0
        return []

    res, _ = run(provider, scen, state.copy(), hist, test, cfg)
    for w, ctx in enumerate(contexts):
        assert ctx.week_idx == w
        assert ctx.origin_idx == T_START - 1 + 7 * w
        assert ctx.hist.shape[0] == hist.shape[0] + 7 * w
        if w > 0:
            np.testing.assert_array_equal(ctx.hist[-7:],
                                          test[7 * (w - 1): 7 * w])
            assert set(ctx.prev) == {"stockout_sku_days", "binding_days",
                                     "selected_tau", "n_modified_skus"}
            assert ctx.prev["selected_tau"] == BASE_TAU
        else:
            np.testing.assert_array_equal(ctx.hist, hist)
            assert ctx.prev is None
    ref, _ = run(lambda w, ctx: [], scen, state.copy(), hist, test, cfg)
    assert res.window_cost == pytest.approx(ref.window_cost, abs=1e-12)


def test_scenario_rejection_is_counterfactual_only(cfg):
    hist, test, scen, state = make_world()
    bad_scen = scen + 1000.0
    action = make_action(0.90, {}, BASE_TAU, SERIES)
    res_bad, _ = run(lambda w, ctx: [action], bad_scen, state.copy(), hist,
                     test, cfg, floor=0.99)
    res_none, _ = run(lambda w, ctx: [], scen, state.copy(), hist, test, cfg)
    assert all(wk.fallback for wk in res_bad.weeks)
    assert res_bad.window_cost == pytest.approx(res_none.window_cost,
                                                abs=1e-12)
    for wk in res_bad.weeks:
        assert wk.gate["csl_rejected"] == 1


def test_selected_action_changes_realized_orders_and_logs(cfg):
    hist, test, scen, state = make_world()
    action = make_action(0.90, {"S_A": 1.5}, BASE_TAU, SERIES)
    res_act, _ = run(lambda w, ctx: [action], scen, state.copy(), hist, test,
                     cfg, k_g=1e9, floor=0.0)
    res_base, _ = run(lambda w, ctx: [], scen, state.copy(), hist, test, cfg,
                      k_g=1e9)
    assert all(not wk.fallback for wk in res_act.weeks)
    assert all(wk.selected_slot == 0 for wk in res_act.weeks)
    assert all(wk.selected_tau == 0.90 for wk in res_act.weeks)
    assert res_act.window_cost != pytest.approx(res_base.window_cost)
    assert all(np.isfinite(wk.selected_mean_cost) for wk in res_act.weeks)


def test_fallback_then_selection_mix_keeps_state_chain(cfg):
    hist, test, scen, state = make_world()
    action = make_action(0.70, {}, BASE_TAU, SERIES)

    def provider(w, ctx):
        return [action] if w in (1, 3) else []

    res, _ = run(provider, scen, state.copy(), hist, test, cfg, k_g=1e9,
                 floor=0.0)
    assert [wk.fallback for wk in res.weeks] == [True, False, True, False]
    assert res.n_days == WINDOW_DAYS
    assert len(res.records) == N_WEEKS
    assert res.records[-1].terminal_cost == pytest.approx(res.terminal)
    assert sum(r.cost for r in res.records) == pytest.approx(res.window_cost)


def test_bad_shapes_raise(cfg):
    hist, test, scen, state = make_world()
    with pytest.raises(ValueError):
        run(lambda w, ctx: [], scen, state, hist, test[:21], cfg)
    with pytest.raises(ValueError):
        run(lambda w, ctx: [], scen[:, :, :4], state, hist, test, cfg)