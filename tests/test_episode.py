"""Tests for 2-C episode assembly and T5 (spec v0.1 section 11).

T5: the core terminal treatment and the 7-day continuation variant are both
computed for every policy; the per-policy cost differences and the pairwise
ranking under both treatments are recorded. Fold-12-style runs (no post-window
demand) simply omit the continuation variant.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.simulator.engine import (
    SkuParams,
    load_sim_config,
    spawn_stream,
)
from core_pipeline.simulator.episode import (
    DiagnosticsRecorder,
    compute_common_warmup,
    continuation_report,
    run_policy_window,
)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(CONFIG_PATH)


def make_inputs(cfg, n_sku=5, hist_days=120, cont_days=7, seed_path=(7, 7)):
    rng = spawn_stream(cfg.seeds.eval, *seed_path)
    total = hist_days + cfg.warmup_days + cfg.test_window_days + cont_days
    demand = rng.poisson(3.0, size=(total, n_sku)).astype(float)
    pre = demand[:hist_days]
    warm = demand[hist_days: hist_days + cfg.warmup_days]
    test28 = demand[hist_days + cfg.warmup_days:
                    hist_days + cfg.warmup_days + cfg.test_window_days]
    test35 = demand[hist_days + cfg.warmup_days:]
    params = SkuParams(
        series_ids=tuple(f"S{i}" for i in range(n_sku)),
        h=np.linspace(0.5, 1.5, n_sku),
        b=np.linspace(0.5, 1.5, n_sku) * 5.0,
        w=np.ones(n_sku),
    )
    k_g = float(demand.mean(axis=0).sum())
    return pre, warm, test28, test35, params, k_g


def policies():
    return {
        "a4_lo": A4BaseStock(A4Config(tau=0.55, allocation="proportional")),
        "a4_hi": A4BaseStock(A4Config(tau=0.90, allocation="proportional")),
    }


def run_both_treatments(cfg, seed_path=(7, 7)):
    pre, warm, test28, test35, params, k_g = make_inputs(cfg, seed_path=seed_path)
    warm_policy = A4BaseStock(A4Config(tau=0.60, allocation="proportional"))
    state = compute_common_warmup(warm, pre, warm_policy, params, k_g, cfg,
                                  warmup_start_idx=120)
    hist = np.concatenate([pre, warm], axis=0)
    t0 = 120 + cfg.warmup_days
    core, cont = {}, {}
    for name, pol in policies().items():
        pol.name = name
        core[name] = run_policy_window(state, test28, hist, pol, params, k_g,
                                       cfg, t0_idx=t0, fold=1, group_id="B")
        cont[name] = run_policy_window(state, test35, hist, pol, params, k_g,
                                       cfg, t0_idx=t0, fold=1, group_id="B")
    return core, cont, state


# --------------------------------------------------------------------------- #
# Aggregation consistency
# --------------------------------------------------------------------------- #

def test_weekly_records_sum_to_window_cost(cfg):
    core, _, _ = run_both_treatments(cfg)
    for res in core.values():
        assert len(res.records) == 4
        assert sum(r.cost for r in res.records) == pytest.approx(res.window_cost)
        terms = [r.terminal_cost for r in res.records]
        assert terms[:3] == [0.0, 0.0, 0.0] and terms[3] == pytest.approx(res.terminal)
        for r in res.records:
            assert r.n_days == 7
            assert 0.0 <= r.csl <= 1.0
            assert 0.0 <= r.fill_rate <= 1.0
            assert r.guardrail_blocked == (r.csl < cfg.episode_csl_floor)


def test_common_warmup_state_shared(cfg):
    # same warm-up inputs -> identical inherited state across repeated calls
    pre, warm, _, _, params, k_g = make_inputs(cfg)
    wp = A4BaseStock(A4Config(tau=0.60, allocation="proportional"))
    s1 = compute_common_warmup(warm, pre, wp, params, k_g, cfg, warmup_start_idx=120)
    s2 = compute_common_warmup(warm, pre, wp, params, k_g, cfg, warmup_start_idx=120)
    np.testing.assert_array_equal(s1.inv, s2.inv)
    np.testing.assert_array_equal(s1.pipeline, s2.pipeline)


def test_diagnostics_collected_for_a4(cfg):
    core, _, _ = run_both_treatments(cfg)
    for res in core.values():
        assert res.raw_exceed_days is not None
        assert 0 <= res.raw_exceed_days <= res.n_days


def test_no_diagnostics_for_plain_policy(cfg):
    pre, warm, test28, _, params, k_g = make_inputs(cfg)

    class ZeroPolicy:
        name = "zero"

        def decide(self, obs):
            return np.zeros(obs.inv.shape[0])

    hist = np.concatenate([pre, warm], axis=0)
    res = run_policy_window(
        compute_common_warmup(warm, pre,
                              A4BaseStock(A4Config(tau=0.6, allocation="proportional")),
                              params, k_g, cfg, warmup_start_idx=120),
        test28, hist, ZeroPolicy(), params, k_g, cfg,
        t0_idx=120 + cfg.warmup_days, fold=1, group_id="B",
    )
    assert res.raw_exceed_days is None


# --------------------------------------------------------------------------- #
# T5: terminal treatment vs continuation, recorded for all policies
# --------------------------------------------------------------------------- #

def test_t5_terminal_variants(cfg):
    core, cont, _ = run_both_treatments(cfg)
    report = continuation_report(core, cont)
    assert set(report["cost_diff"]) == {"a4_lo", "a4_hi"}
    for p, diff in report["cost_diff"].items():
        assert cont[p].n_days == 35 and core[p].n_days == 28
        assert math.isfinite(diff)
    assert set(report["rank_core"]) == {"a4_lo", "a4_hi"}
    assert report["rank_preserved"] in (True, False)
    # continuation variant also closes with a terminal valuation at day 35
    for p in cont:
        assert cont[p].terminal > 0.0


def test_t5_fold12_no_continuation(cfg):
    # fold-12-style: continuation omitted entirely, core still complete
    core, _, _ = run_both_treatments(cfg)
    report = continuation_report(core, {})
    assert report["cost_diff"] == {}
    assert report["rank_preserved"] is None
    assert len(report["rank_core"]) == 2


def test_t5_determinism(cfg):
    a_core, a_cont, _ = run_both_treatments(cfg)
    b_core, b_cont, _ = run_both_treatments(cfg)
    for p in a_core:
        assert a_core[p].window_cost == b_core[p].window_cost
        assert a_cont[p].window_cost == b_cont[p].window_cost
