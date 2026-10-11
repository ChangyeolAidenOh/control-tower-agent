"""ST7-02 D5: numeric-search comparison arm -- uniform candidate sampling
over the SAME discrete action grids, with a pre-generated, sha-recorded
sample schedule.

Sampling per episode (fold, bundle, week), seed rng([20261011, fold,
bundle_rank, week_idx]), 20 iid candidates in a FIXED draw order per
candidate: tau' ~ U(TAU_GRID), k ~ U{0..k_max}, SKU subset uniform without
replacement (size k), each m_i ~ U(MULT_GRID). k_max = min(10, n_sku)
(the <= 10 listed-SKU contract capped by the bundle size; recorded in the
schedule meta). Stated limitation (ST7-02 D5): uniform over THIS generation
procedure, not over the full action-combination space (the number of SKU
combinations varies with k).

Consequences kept as-generated: a drawn m_i = 1.0 normalizes away (the
effective modified-SKU count can be below k), and duplicate actions can
occur -- both consume their budget slot and are evaluated separately in the
official H5 runs (no caching, ST7-02 D5).

The schedule is generated ONCE (scripts/build_agent_search_schedule.py),
written with a canonical-JSON sha256, and the runner later verifies both
the file sha and bitwise regeneration equality before use.
"""

import hashlib
import json

import numpy as np

from core_pipeline.agent.candidates import (
    MAX_CANDIDATES,
    MAX_LISTED_SKUS,
    MULT_GRID,
    TAU_GRID,
    AgentAction,
    make_action,
)

SEED_ROOT = 20261011
N_WEEKS = 4


def sample_episode_candidates(
    fold: int,
    bundle_rank: int,
    week_idx: int,
    series_ids,
    n_candidates: int = MAX_CANDIDATES,
) -> list[dict]:
    """Deterministic raw candidate list for one episode. Returns the
    as-generated form ({tau_prime, sku_multipliers}); normalization to
    AgentAction happens downstream via make_action."""
    series = [str(s) for s in series_ids]
    n_sku = len(series)
    k_max = min(MAX_LISTED_SKUS, n_sku)
    rng = np.random.default_rng([SEED_ROOT, fold, bundle_rank, week_idx])
    out = []
    for _ in range(n_candidates):
        tau = float(TAU_GRID[int(rng.integers(0, len(TAU_GRID)))])
        k = int(rng.integers(0, k_max + 1))
        idx = rng.choice(n_sku, size=k, replace=False)
        mults = {series[int(i)]: float(MULT_GRID[int(j)])
                 for i, j in zip(idx, rng.integers(0, len(MULT_GRID), size=k))}
        out.append({"tau_prime": tau, "sku_multipliers": mults})
    return out


def actions_from_entries(entries: list[dict], base_tau: float,
                         series_ids) -> list[AgentAction]:
    """Normalize schedule entries through the SAME contract path as the LLM
    arm (make_action raises on any breach -- the sampler cannot emit an
    illegal action silently)."""
    return [make_action(e["tau_prime"], e["sku_multipliers"], base_tau,
                        series_ids)
            for e in entries]


def build_schedule(bundle_series: dict[str, tuple[str, ...]],
                   folds: list[int]) -> dict:
    """Full pre-generated schedule: folds x sorted bundles x weeks 0..3."""
    group_ids = sorted(bundle_series)
    episodes = {}
    for fold in folds:
        for rank, gid in enumerate(group_ids):
            for week in range(N_WEEKS):
                key = f"f{fold}|{gid}|w{week}"
                episodes[key] = sample_episode_candidates(
                    fold, rank, week, bundle_series[gid])
    return {
        "seed_root": SEED_ROOT,
        "tau_grid": list(TAU_GRID),
        "mult_grid": list(MULT_GRID),
        "n_candidates": MAX_CANDIDATES,
        "k_rule": f"uniform 0..min({MAX_LISTED_SKUS}, n_sku)",
        "draw_order": "tau, k, sku_indices(no-replacement), multipliers",
        "folds": folds,
        "group_ids": group_ids,
        "n_sku": {gid: len(bundle_series[gid]) for gid in group_ids},
        "episodes": episodes,
    }


def canonical_sha(obj: dict) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()