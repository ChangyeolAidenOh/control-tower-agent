"""ST7-02 D1 verification: schema errors, duplicate/no-action normalization,
truncation beyond 20, listed-SKU limit, grid tolerance."""

import json

import pytest

from core_pipeline.agent.candidates import (
    MAX_CANDIDATES,
    AgentAction,
    make_action,
    parse_candidates,
    snap_to_grid,
)

SERIES = ("S_A", "S_B", "S_C")
BASE_TAU = 0.60


def envelope(cands) -> str:
    return json.dumps({"candidates": cands})


def cand(cid="c01", **kw) -> dict:
    out = {"candidate_id": cid}
    out.update(kw)
    return out


def test_valid_candidate_parses():
    res = parse_candidates(envelope([
        cand(tau_prime=0.70, sku_multipliers={"S_A": 1.2}, rationale="x"),
    ]), SERIES, BASE_TAU)
    assert not res.parse_failed
    assert len(res.valid) == 1
    a = res.valid[0].action
    assert a.tau_eff == 0.70
    assert a.multipliers == (("S_A", 1.2),)
    assert a.n_modified_skus == 1
    assert a.tau_delta == pytest.approx(0.10)


def test_null_tau_equals_base_tau_and_unlisted_equals_one():
    res = parse_candidates(envelope([
        cand("c1", tau_prime=None, sku_multipliers={}),
        cand("c2", tau_prime=BASE_TAU, sku_multipliers={"S_A": 1.0}),
    ]), SERIES, BASE_TAU)
    a1, a2 = (c.action for c in res.valid)
    assert a1 == a2
    assert a1.is_no_action
    assert a1.tau_delta == 0.0
    assert res.stats()["duplicate_valid_slots"] == 1


def test_missing_tau_key_means_base():
    res = parse_candidates(envelope([cand("c1")]), SERIES, BASE_TAU)
    assert res.valid[0].action.tau_eff == BASE_TAU


def test_grid_tolerance_snaps():
    res = parse_candidates(envelope([
        cand(tau_prime=0.7000000000001, sku_multipliers={"S_B": 0.7999999999999}),
    ]), SERIES, BASE_TAU)
    a = res.valid[0].action
    assert a.tau_eff == 0.70
    assert a.multipliers == (("S_B", 0.8),)


@pytest.mark.parametrize("bad", [
    cand(tau_prime=0.65),
    cand(tau_prime=True),
    cand(tau_prime="0.70"),
    cand(sku_multipliers={"S_A": 1.1}),
    cand(sku_multipliers={"S_X": 1.2}),
    cand(sku_multipliers="S_A"),
    cand(rationale=42),
    cand(extra_field=1),
    {"tau_prime": 0.70},
    {"candidate_id": ""},
    "not_an_object",
])
def test_invalid_candidates_are_dropped_individually(bad):
    res = parse_candidates(envelope([bad, cand("ok", tau_prime=0.50)]),
                           SERIES, BASE_TAU)
    assert not res.parse_failed
    assert len(res.candidates) == 2
    assert len(res.valid) == 1
    assert res.valid[0].candidate_id == "ok"


def test_listed_sku_limit_is_on_written_keys():
    many = {f"S_{i}": 1.2 for i in range(11)}
    series = tuple(many) + SERIES
    res = parse_candidates(envelope([cand(sku_multipliers=many)]),
                           series, BASE_TAU)
    assert res.candidates[0].invalid_reason == "too_many_skus"

    ten_with_ones = {f"S_{i}": (1.0 if i < 4 else 1.5) for i in range(10)}
    res2 = parse_candidates(envelope([cand(sku_multipliers=ten_with_ones)]),
                            series, BASE_TAU)
    assert len(res2.valid) == 1
    assert res2.valid[0].action.n_modified_skus == 6


def test_zero_multiplier_is_legal_and_counts_as_modified():
    res = parse_candidates(envelope([cand(sku_multipliers={"S_C": 0.0})]),
                           SERIES, BASE_TAU)
    assert res.valid[0].action.multipliers == (("S_C", 0.0),)
    assert res.valid[0].action.n_modified_skus == 1


def test_truncation_keeps_first_twenty_in_array_order():
    cands = [cand(f"c{i:02d}", tau_prime=0.50) for i in range(25)]
    res = parse_candidates(envelope(cands), SERIES, BASE_TAU)
    assert len(res.candidates) == MAX_CANDIDATES
    assert res.truncated_count == 5
    assert res.candidates[-1].candidate_id == "c19"


def test_duplicate_candidate_ids_are_allowed_as_labels():
    res = parse_candidates(envelope([
        cand("same", tau_prime=0.50), cand("same", tau_prime=0.90),
    ]), SERIES, BASE_TAU)
    assert len(res.valid) == 2
    assert res.stats()["duplicate_valid_slots"] == 0


def test_duplicate_json_keys_invalidate_candidate():
    text = ('{"candidates": [{"candidate_id": "c1", "tau_prime": 0.5, '
            '"tau_prime": 0.7}]}')
    res = parse_candidates(text, SERIES, BASE_TAU)
    assert res.candidates[0].invalid_reason == "duplicate_json_keys"

    text2 = ('{"candidates": [{"candidate_id": "c1", '
             '"sku_multipliers": {"S_A": 1.2, "S_A": 0.5}}]}')
    res2 = parse_candidates(text2, SERIES, BASE_TAU)
    assert res2.candidates[0].invalid_reason == "duplicate_sku_key"


def test_envelope_failures_zero_the_episode():
    assert parse_candidates("not json", SERIES, BASE_TAU).parse_failed
    assert parse_candidates('[{"candidate_id": "c1"}]', SERIES, BASE_TAU).parse_failed
    assert parse_candidates('{"cands": []}', SERIES, BASE_TAU).parse_failed
    dup = '{"candidates": [], "candidates": []}'
    assert parse_candidates(dup, SERIES, BASE_TAU).parse_failed


def test_envelope_extra_keys_recorded_not_fatal():
    res = parse_candidates(
        '{"candidates": [{"candidate_id": "c1"}], "notes": "hi"}',
        SERIES, BASE_TAU)
    assert not res.parse_failed
    assert res.envelope_extra_keys == ("notes",)
    assert len(res.valid) == 1


def test_make_action_matches_parser_normalization():
    a = make_action(None, {"S_A": 1.0, "S_B": 1.5}, BASE_TAU, SERIES)
    assert a == AgentAction(base_tau=BASE_TAU, tau_eff=BASE_TAU,
                            multipliers=(("S_B", 1.5),))
    with pytest.raises(ValueError):
        make_action(0.65, {}, BASE_TAU, SERIES)
    with pytest.raises(ValueError):
        make_action(None, {"S_X": 1.2}, BASE_TAU, SERIES)
    with pytest.raises(ValueError):
        make_action(None, {f"S{i}": 1.2 for i in range(11)},
                    BASE_TAU, tuple(f"S{i}" for i in range(11)))
    with pytest.raises(ValueError):
        make_action(0.70, {}, 0.55, SERIES)


def test_base_tau_must_be_on_grid():
    with pytest.raises(ValueError):
        parse_candidates(envelope([cand()]), SERIES, 0.55)


def test_snap_rejects_bool_and_strings():
    assert snap_to_grid(True, (1.0,)) is None
    assert snap_to_grid("0.5", (0.5,)) is None
    assert snap_to_grid(0.5, (0.5,)) == 0.5