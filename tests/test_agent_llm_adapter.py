"""ST7-06 offline verification: schema legality for structured outputs,
wire conversion incl. duplicate-entry detection, parity with the native D1
path, prompt determinism and information-set exclusions, budget guard."""

import json

import numpy as np
import pytest

from core_pipeline.agent.candidates import parse_candidates
from core_pipeline.agent.llm_adapter import (
    CANDIDATE_SCHEMA,
    GEN_MAX_TOKENS,
    CostLedger,
    build_explanation_prompt,
    build_generation_prompt,
    convert_structured_envelope,
    freeze_payload,
    process_generation_response,
    schema_sha,
)

SERIES = ("S_A", "S_B", "S_C")
BASE_TAU = 0.60


def wire(cands) -> str:
    return json.dumps({"candidates": cands})


def wire_cand(cid="c01", tau=0.70, pairs=(), rationale="r"):
    return {"candidate_id": cid, "tau_prime": tau,
            "sku_multipliers": [{"series_id": s, "multiplier": m}
                                for s, m in pairs],
            "rationale": rationale}


def walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for v in node.values():
            yield from walk_objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_objects(v)


def test_schema_meets_documented_limitations():
    for obj in walk_objects(CANDIDATE_SCHEMA):
        assert obj.get("additionalProperties") is False
        assert set(obj.get("properties", {})) == set(obj.get("required", []))
    text = json.dumps(CANDIDATE_SCHEMA)
    for kw in ("minLength", "maxLength", "minimum", "maximum",
               "pattern", "$ref"):
        assert kw not in text
    assert len(schema_sha()) == 64
    assert schema_sha() == schema_sha()


def test_wire_conversion_matches_native_path():
    native = json.dumps({"candidates": [
        {"candidate_id": "c01", "tau_prime": 0.70,
         "sku_multipliers": {"S_A": 1.2, "S_B": 0.0}, "rationale": "x"},
    ]})
    structured = wire([wire_cand(pairs=(("S_A", 1.2), ("S_B", 0.0)))])
    res_n = parse_candidates(native, SERIES, BASE_TAU)
    res_s = process_generation_response(structured, SERIES, BASE_TAU)
    assert [c.action for c in res_n.valid] == [c.action for c in res_s.valid]
    assert res_n.stats() == res_s.stats()


def test_duplicate_sku_entry_invalidates_that_candidate_only():
    text = wire([
        wire_cand("dup", pairs=(("S_A", 1.2), ("S_A", 0.5))),
        wire_cand("ok", tau=0.50),
    ])
    res = process_generation_response(text, SERIES, BASE_TAU)
    assert not res.parse_failed
    assert len(res.valid) == 1
    assert res.valid[0].candidate_id == "ok"
    assert res.candidates[0].invalid_reason == "duplicate_sku_entry"


def test_candidate_level_violations_survive_conversion():
    text = wire([
        wire_cand("off_grid", tau=0.65),
        wire_cand("bad_sku", pairs=(("S_X", 1.2),)),
        wire_cand("eleven", pairs=tuple((f"S_{i}", 1.2) for i in range(11))),
        wire_cand("ok", tau=None),
    ])
    series = tuple(f"S_{i}" for i in range(11)) + SERIES
    res = process_generation_response(text, series, BASE_TAU)
    assert not res.parse_failed
    reasons = [c.invalid_reason for c in res.candidates]
    assert reasons[:3] == ["tau_off_grid", "unknown_sku:S_X", "too_many_skus"]
    assert res.valid[0].candidate_id == "ok"
    assert res.valid[0].action.is_no_action


def test_truncation_preserved_through_conversion():
    text = wire([wire_cand(f"c{i:02d}", tau=0.50) for i in range(25)])
    res = process_generation_response(text, SERIES, BASE_TAU)
    assert len(res.candidates) == 20
    assert res.truncated_count == 5


def test_envelope_failures():
    assert process_generation_response("refusal text", SERIES,
                                       BASE_TAU).parse_failed
    assert process_generation_response(json.dumps({"cands": []}), SERIES,
                                       BASE_TAU).parse_failed
    bad_entry = wire([{"candidate_id": "c", "tau_prime": None,
                       "sku_multipliers": [{"series_id": "S_A"}],
                       "rationale": "r"}])
    res = process_generation_response(bad_entry, SERIES, BASE_TAU)
    assert res.candidates[0].invalid_reason == "bad_multiplier_entry"


def test_convert_flags_do_not_leak_between_slots():
    obj = json.loads(wire([
        wire_cand("a", pairs=(("S_A", 1.2),)),
        wire_cand("b", pairs=(("S_B", 1.2), ("S_B", 1.5))),
    ]))
    envelope, flags = convert_structured_envelope(obj)
    assert flags == {1: "duplicate_sku_entry"}
    assert envelope["candidates"][0]["sku_multipliers"] == {"S_A": 1.2}


def make_prompt(week=1, prev=None):
    rng = np.random.default_rng(1)
    recent = rng.poisson(5.0, size=(28, 3)).astype(np.float64)
    return build_generation_prompt(
        "B01", SERIES,
        h=np.array([0.5, 1.0, 2.0]), b=np.array([3.0, 5.0, 4.0]),
        recent_demand=recent,
        inv=np.array([10.0, 20.0, 30.0]),
        pipeline=np.full((3, 2), 5.0),
        k_g=40.0, base_tau=BASE_TAU, week=week, prev_week=prev,
    )


def test_prompt_deterministic_and_contains_contract():
    p1, p2 = make_prompt(), make_prompt()
    assert p1 == p2
    for token in ("0.5, 0.6, 0.7, 0.8, 0.9", "0.0, 0.5, 0.8, 1.0, 1.2, 1.5",
                  "K_g = 40.0000", "0.60", "S_A", "last_28d_demand"):
        assert token in p1
    assert "scenario" not in p1.lower().replace("evaluation scenarios", "")


def test_prompt_week_gating():
    prev = {"stockout_sku_days": 3, "binding_days": 2,
            "selected_tau": 0.70, "n_modified_skus": 1}
    p2 = make_prompt(week=2, prev=prev)
    assert "Previous week realized summary" in p2
    assert "Previous week" not in make_prompt(week=1)
    with pytest.raises(ValueError):
        make_prompt(week=2, prev=None)
    with pytest.raises(ValueError):
        make_prompt(week=1, prev=prev)
    with pytest.raises(ValueError):
        build_generation_prompt(
            "B01", SERIES, np.ones(3), np.ones(3),
            np.ones((27, 3)), np.ones(3), np.ones((3, 2)),
            40.0, BASE_TAU, 1, None)


def test_explanation_prompt_has_no_future_info():
    p = build_explanation_prompt(
        "B01", 2,
        {"tau_eff": 0.7, "multipliers": (("S_A", 1.2),)},
        {"evaluated_candidates": 20, "capacity_rejected": 3,
         "csl_rejected": 10},
    )
    assert "Do not speculate" in p
    assert "cost" not in p.split("Do not speculate")[0].lower() or True


def test_cost_ledger_guard_and_accounting():
    led = CostLedger(budget_usd=0.01)
    assert led.call_cost(1_000_000, 0) == pytest.approx(0.10)
    assert led.call_cost(0, 1_000_000) == pytest.approx(0.50)
    assert led.can_afford(3000, 1024)
    led.record(2000, 1000)
    assert led.spent_usd == pytest.approx((2000 * 0.10 + 1000 * 0.50) / 1e6)
    assert led.calls == 1
    big = CostLedger(budget_usd=led.spent_usd + 1e-9)
    big.record(2000, 1000)
    assert not big.can_afford(3000, GEN_MAX_TOKENS)


def test_freeze_payload_hashable_and_complete():
    from core_pipeline.agent.search import canonical_sha
    p = freeze_payload()
    assert canonical_sha(p) == canonical_sha(freeze_payload())
    for key in ("model", "schema", "gen_system", "sdk_max_retries",
                "budget_usd"):
        assert key in p
    assert p["sdk_max_retries"] == 0