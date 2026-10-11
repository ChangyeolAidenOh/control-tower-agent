"""ST7-02 D2/D3/D4: candidate gate (capacity raw rejection -> 7-day CSL
scenario median) and 10-day mean-cost ranking.

Per candidate, per scenario s in 1..9 (A3 comonotone quantile paths at the
week's selection origin t_w): simulate 10 days from the inherited realized
state -- 7 action days (tau', multipliers) + 3 continuation days (base A4,
multipliers 1.0) -- with capacity_mode="raw", so any pre-scaling order
beyond K_g raises CapacityViolation and the candidate is rejected
(ST7-02 D2; the realized path uses "scale" elsewhere). Gate statistics:

  capacity   violation on ANY scenario day -> reject, reason "capacity"
  csl        median over the 9 scenarios of the FIRST-7-day bundle-week
             CSL(SKU-day) = mean 1[u == 0] (episode definition reused)
             < floor -> reject, reason "csl"
  rank       among passing candidates: minimal mean over 9 scenarios of
             the 10-day cost (holding + shortage + terminal at day 10);
             ties -> fewer modified SKUs -> smaller |tau' - base| -> slot.

The 10-day cost is a counterfactual selection quantity only; realized
execution runs the first 7 days and H3/H5 costs come from the realized
28-day window (ST7-02 D4/D6). All-candidates-rejected -> the caller runs
the no-action A4 fallback; that is an operational fallback, not a safety
certification (ST7-02 D3).
"""

from dataclasses import dataclass

import numpy as np

from core_pipeline.agent.actions import ACTION_DAYS, AgentA4Policy, CapacityViolation
from core_pipeline.agent.candidates import AgentAction
from core_pipeline.simulator.engine import (
    BundleState,
    SimConfig,
    SkuParams,
    run_window,
    terminal_cost,
)

SELECTION_DAYS = 10
CSL_DAYS = 7
N_SCEN = 9
REASON_CAPACITY = "capacity"
REASON_CSL = "csl"


@dataclass(frozen=True)
class CandidateEvaluation:
    slot: int
    action: AgentAction
    capacity_rejected: bool
    capacity_violation_t: int | None
    scenario_csl: tuple[float, ...]      # () when capacity-rejected
    csl_median: float                    # nan when capacity-rejected
    scenario_cost: tuple[float, ...]     # () when capacity-rejected
    mean_cost: float                     # nan when capacity-rejected
    gate_passed: bool
    reject_reason: str | None


def evaluate_candidate(
    slot: int,
    action: AgentAction,
    scenario_demand: np.ndarray,         # (SELECTION_DAYS, n_sku, N_SCEN)
    start_state: BundleState,
    demand_hist: np.ndarray,             # all latent demand up to t_w incl.
    series_ids,
    params: SkuParams,
    k_g: float,
    cfg: SimConfig,
    *,
    t0_idx: int,                         # first action day = t_w + 1
    csl_floor: float,
) -> CandidateEvaluation:
    if scenario_demand.shape[0] != SELECTION_DAYS:
        raise ValueError(
            f"scenario_demand has {scenario_demand.shape[0]} days, "
            f"expected {SELECTION_DAYS}")
    if scenario_demand.shape[2] != N_SCEN:
        raise ValueError(
            f"scenario_demand has {scenario_demand.shape[2]} scenarios, "
            f"expected {N_SCEN}")

    csls: list[float] = []
    costs: list[float] = []
    for s in range(N_SCEN):
        policy = AgentA4Policy(
            action, series_ids,
            action_start_idx=t0_idx,
            tol_rel=cfg.feasibility_tol_rel,
            capacity_mode="raw",
            action_days=ACTION_DAYS,
        )
        try:
            final, days = run_window(
                start_state.copy(), scenario_demand[:, :, s], demand_hist,
                policy, params, k_g=k_g, cfg=cfg, t0_idx=t0_idx,
            )
        except CapacityViolation as exc:
            return CandidateEvaluation(
                slot=slot, action=action,
                capacity_rejected=True, capacity_violation_t=exc.t_idx,
                scenario_csl=(), csl_median=float("nan"),
                scenario_cost=(), mean_cost=float("nan"),
                gate_passed=False, reject_reason=REASON_CAPACITY,
            )
        unmet = np.stack([r.unmet for r in days[:CSL_DAYS]])
        csls.append(float(np.mean(unmet == 0.0)))
        cost = sum(r.holding_cost + r.shortage_cost for r in days)
        costs.append(float(cost + terminal_cost(final, params.h)))

    csl_median = float(np.median(csls))
    passed = csl_median >= csl_floor
    return CandidateEvaluation(
        slot=slot, action=action,
        capacity_rejected=False, capacity_violation_t=None,
        scenario_csl=tuple(csls), csl_median=csl_median,
        scenario_cost=tuple(costs), mean_cost=float(np.mean(costs)),
        gate_passed=passed,
        reject_reason=None if passed else REASON_CSL,
    )


def select_candidate(
    evaluations: list[CandidateEvaluation],
) -> CandidateEvaluation | None:
    """Rank the gate-passing candidates (ST7-02 D4 tie-break chain); None
    means all rejected -> the caller executes the no-action A4 fallback."""
    passed = [e for e in evaluations if e.gate_passed]
    if not passed:
        return None
    return min(passed, key=lambda e: (
        e.mean_cost,
        e.action.n_modified_skus,
        e.action.tau_delta,
        e.slot,
    ))


def gate_stats(evaluations: list[CandidateEvaluation]) -> dict:
    """Per-episode gate accounting (ST7-02 D3 reporting split)."""
    n = len(evaluations)
    n_cap = sum(1 for e in evaluations if e.reject_reason == REASON_CAPACITY)
    n_csl = sum(1 for e in evaluations if e.reject_reason == REASON_CSL)
    n_pass = sum(1 for e in evaluations if e.gate_passed)
    return {
        "evaluated_candidates": n,
        "capacity_rejected": n_cap,
        "csl_rejected": n_csl,
        "gate_passed_candidates": n_pass,
        "candidate_gate_pass_rate": n_pass / n if n else float("nan"),
        "all_rejected": n_pass == 0,
    }