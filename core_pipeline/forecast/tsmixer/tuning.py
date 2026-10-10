"""ST4-07 tuning protocol -- pure logic (torch-free, main-suite tested).

Grid (frozen by verdict): d_model x n_blocks x dropout x lr = 3*2*2*2 = 24,
with ff_dim = d_model, d_feat = d_model / 2, d_static = 16, batch 1024,
L = 56, max_epochs = 40, patience = 5. trial_id is the canonical
enumeration order; execution runs in a deterministically shuffled order
(fixed seed) so an interrupted search is not biased toward one model size
or learning rate.

Selection: argmin over evaluated trials of the A3 validation metric
computed on rearranged, non-negative, raw-unit quantiles. Tie rule
(verdict): (L_j - L_min) / L_min <= 1e-3 -> fewest A3 parameters ->
lowest canonical trial_id. The common HP is A3-favoring by design and is
reported as such; it does not guarantee per-model optimality for
A1/A2/B.
"""

from itertools import product

import numpy as np

D_MODEL_GRID = (32, 64, 128)
N_BLOCKS_GRID = (2, 4)
DROPOUT_GRID = (0.1, 0.2)
LR_GRID = (1e-3, 3e-4)
MAX_EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 1024
EXECUTION_SHUFFLE_SEED = 20261010
TRIAL_SEED_BASE = 90000
REL_TIE = 1e-3


def build_trials() -> list[dict]:
    trials = []
    for tid, (dm, nb, do, lr) in enumerate(
            product(D_MODEL_GRID, N_BLOCKS_GRID, DROPOUT_GRID, LR_GRID)):
        trials.append({
            "trial_id": tid,
            "d_model": dm,
            "n_blocks": nb,
            "dropout": do,
            "lr": lr,
            "ff_dim": dm,
            "d_feat": dm // 2,
            "seed": TRIAL_SEED_BASE + tid,
        })
    return trials


def execution_order(n_trials: int = 24) -> list[int]:
    rng = np.random.default_rng(EXECUTION_SHUFFLE_SEED)
    return [int(i) for i in rng.permutation(n_trials)]


def select_trial(rows: list[dict]) -> dict:
    """rows: evaluated trials with best_metric, params, trial_id. Returns
    the winner plus selection bookkeeping (tie membership is recorded)."""
    if not rows:
        raise ValueError("no evaluated trials")
    finite = [r for r in rows if np.isfinite(r["best_metric"])]
    if not finite:
        raise ValueError("no finite trial metrics")
    l_min = min(r["best_metric"] for r in finite)
    tied = [r for r in finite
            if (r["best_metric"] - l_min) / l_min <= REL_TIE]
    winner = sorted(tied, key=lambda r: (r["params"], r["trial_id"]))[0]
    return {
        "trial_id": winner["trial_id"],
        "l_min": float(l_min),
        "tie_group": sorted(r["trial_id"] for r in tied),
        "tie_rule_applied": len(tied) > 1,
    }
