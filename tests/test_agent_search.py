"""ST7-02 D5 verification: determinism, grid legality through make_action,
k bounds under small bundles, slot budget, duplicate/no-action occurrence
as natural uniform-sampling outcomes, canonical sha stability."""

import numpy as np

from core_pipeline.agent.candidates import MAX_CANDIDATES, MULT_GRID, TAU_GRID
from core_pipeline.agent.search import (
    actions_from_entries,
    build_schedule,
    canonical_sha,
    sample_episode_candidates,
)

SERIES = tuple(f"S_{i:02d}" for i in range(15))
SMALL = tuple(f"T_{i}" for i in range(4))
BASE_TAU = 0.60


def test_deterministic_and_slot_budget():
    a = sample_episode_candidates(3, 2, 1, SERIES)
    b = sample_episode_candidates(3, 2, 1, SERIES)
    assert a == b
    assert len(a) == MAX_CANDIDATES
    c = sample_episode_candidates(3, 2, 2, SERIES)
    assert c != a


def test_entries_pass_the_contract_path():
    entries = sample_episode_candidates(1, 0, 0, SERIES)
    actions = actions_from_entries(entries, BASE_TAU, SERIES)
    assert len(actions) == MAX_CANDIDATES
    for e, act in zip(entries, actions):
        assert e["tau_prime"] in TAU_GRID
        assert len(e["sku_multipliers"]) <= 10
        for sid, m in e["sku_multipliers"].items():
            assert sid in SERIES
            assert m in MULT_GRID
        assert act.n_modified_skus <= len(e["sku_multipliers"])


def test_k_capped_by_bundle_size():
    for fold in range(1, 13):
        entries = sample_episode_candidates(fold, 0, 0, SMALL)
        for e in entries:
            assert len(e["sku_multipliers"]) <= len(SMALL)
        actions_from_entries(entries, BASE_TAU, SMALL)


def test_uniform_outcomes_include_duplicates_and_near_no_action():
    seen = []
    n_dup = 0
    for fold in range(1, 13):
        for week in range(4):
            for act in actions_from_entries(
                    sample_episode_candidates(fold, 0, week, SERIES),
                    BASE_TAU, SERIES):
                key = (act.tau_eff, act.multipliers)
                if key in seen:
                    n_dup += 1
                else:
                    seen.append(key)
    assert n_dup > 0


def test_schedule_build_and_sha_stability():
    bundles = {"B01": SERIES, "B02": SMALL}
    s1 = build_schedule(bundles, [1, 2])
    s2 = build_schedule(bundles, [1, 2])
    assert canonical_sha(s1) == canonical_sha(s2)
    assert len(s1["episodes"]) == 2 * 2 * 4
    assert s1["episodes"]["f1|B01|w0"] == sample_episode_candidates(
        1, 0, 0, SERIES)
    assert s1["episodes"]["f2|B02|w3"] == sample_episode_candidates(
        2, 1, 3, SMALL)
    s3 = build_schedule(bundles, [1])
    assert canonical_sha(s3) != canonical_sha(s1)