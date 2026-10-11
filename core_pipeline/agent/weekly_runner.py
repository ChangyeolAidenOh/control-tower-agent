"""ST7-02 D6: weekly sequential agent runner (one arm, one fold x bundle).

Structure: the 28-day window runs as FOUR sequential 7-day run_window calls.
Candidate selection for week w happens at the close of t_w = T - 1 + 7w,
strictly between t_w's midnight transition and t_w+1's demand realization:
  - selection state  = the realized state inherited from week w-1 (week 0:
    the common warm-up state), BEFORE any week-w demand is observed;
  - selection history = base history + realized test days [T, t_w] only
    (structural information blocking, P0-1);
  - selection demand  = the A3 agent-selection replay at origin t_w,
    first SELECTION_DAYS days (counterfactual, never advances real state).
The candidate provider receives a SelectionContext assembled at exactly
this point (state COPY, history view, previous-week realized summary), so
an LLM provider's information set coincides with the gate's by
construction (L2); the numeric arm ignores the context.

Selection timing and budget (ST7-01/L6): selection_time_s covers the
provider call (generation + parsing for the LLM arm) plus candidate
evaluation, gating and ranking -- never the realized execution or the
explanation call. With selection_time_budget_s set (numeric arm official
runs: the LLM arm's measured per-episode median), the evaluation loop
stops once the elapsed selection time exceeds the budget and ONLY the
candidates already evaluated are used; the cut is recorded
(budget_exceeded, candidates_provided vs evaluated_candidates).

The winner (or the no-action fallback when all candidates are rejected)
executes the realized week in capacity_mode="scale" (ST7-02 D2); the final
engine state carries into week w+1 (states are arm-private, D6). After week
4 the terminal valuation and the weekly aggregation reuse the FROZEN episode
machinery (_aggregate_weeks import), so H3/H5 realized costs stay on the
Stage 6 28-day contract. The 10-day selection costs are never added to
realized costs (D4).

Chaining identity: run_window ends every day with midnight_transition, so
four chained 7-day calls reproduce one 28-day call day-by-day when the
policy coincides (pinned by the no-action parity test).

ST8-05 observability extension: AgentWindowResult carries the executed
28-day DayResult sequence (days, default ()) for instrumentation-only
consumers (Stage 8 inventory diagnostics). The field never feeds back into
candidate selection, gating, or week-to-week state -- decision logic is
byte-identical to the frozen official runs.
"""

import time
from dataclasses import dataclass

import numpy as np

from core_pipeline.agent.actions import AgentA4Policy
from core_pipeline.agent.candidates import AgentAction, make_action
from core_pipeline.agent.gate import (
    SELECTION_DAYS,
    CandidateEvaluation,
    evaluate_candidate,
    gate_stats,
    select_candidate,
)
from core_pipeline.simulator.engine import (
    BundleState,
    DayResult,
    SimConfig,
    SkuParams,
    run_window,
    terminal_cost,
)
from core_pipeline.simulator.episode import (
    WEEK_DAYS,
    EpisodeRecord,
    _aggregate_weeks,
)

N_WEEKS = 4
WINDOW_DAYS = N_WEEKS * WEEK_DAYS
FORECAST_DAYS = 28
N_SCEN = 9


@dataclass(frozen=True)
class SelectionContext:
    """Close-of-t_w information set handed to the candidate provider (L2).
    state is a COPY -- mutating it never touches the realized rollout."""
    week_idx: int                    # 0-based
    origin_idx: int                  # t_w
    state: BundleState
    hist: np.ndarray                 # all latent demand up to and incl. t_w
    prev: dict | None                # realized summary of week w-1; None at w=0


@dataclass(frozen=True)
class WeekLog:
    week: int                        # 1-based
    origin_idx: int                  # t_w
    gate: dict
    fallback: bool                   # all candidates rejected -> no-action A4
    selected_slot: int | None
    selected_tau: float
    selected_multipliers: tuple[tuple[str, float], ...]
    selected_mean_cost: float        # nan on fallback
    scaling: dict                    # realized-path D2 record
    evaluations: tuple[CandidateEvaluation, ...]
    selection_time_s: float          # provider + evaluation + rank (L6)
    candidates_provided: int         # slots handed over by the provider
    budget_exceeded: bool            # evaluation cut by the time budget


@dataclass(frozen=True)
class AgentWindowResult:
    fold: int
    group_id: str
    policy: str
    records: tuple[EpisodeRecord, ...]
    window_cost: float
    terminal: float
    n_days: int
    weeks: tuple[WeekLog, ...]
    days: tuple[DayResult, ...] = ()  # ST8-05 instrumentation only


def run_agent_window(
    start_state: BundleState,
    test_demand: np.ndarray,         # (28, n_sku) realized path (bp.test)
    base_hist: np.ndarray,           # concat(history, warmup)
    candidate_provider,              # callable(week_idx, SelectionContext) -> list[AgentAction]
    selection_forecast,              # callable(origin_idx) -> (28, n_sku, 9)
    series_ids,
    params: SkuParams,
    k_g: float,
    cfg: SimConfig,
    *,
    test_start_idx: int,
    fold: int,
    group_id: str,
    base_tau: float,
    policy_name: str,
    csl_floor: float | None = None,  # official runs: None -> cfg.episode_csl_floor
    selection_time_budget_s: float | None = None,
) -> AgentWindowResult:
    n_days, n_sku = test_demand.shape
    if n_days != WINDOW_DAYS:
        raise ValueError(f"test_demand has {n_days} days, expected {WINDOW_DAYS}")
    if len(series_ids) != n_sku:
        raise ValueError("series/demand sku dimension mismatch")
    floor = cfg.episode_csl_floor if csl_floor is None else float(csl_floor)
    no_action = make_action(None, {}, base_tau, series_ids)

    days_all = []
    week_logs = []
    state = start_state.copy()
    for w in range(N_WEEKS):
        t_w = test_start_idx - 1 + WEEK_DAYS * w
        hist_sel = np.concatenate(
            [base_hist, test_demand[: WEEK_DAYS * w]], axis=0)

        prev = None
        if w > 0:
            chunk = days_all[-WEEK_DAYS:]
            last = week_logs[-1]
            prev = {
                "stockout_sku_days": float(
                    sum(int((r.unmet > 0).sum()) for r in chunk)),
                "binding_days": float(sum(1 for r in chunk if r.binding)),
                "selected_tau": last.selected_tau,
                "n_modified_skus": float(len(last.selected_multipliers)),
            }
        ctx = SelectionContext(week_idx=w, origin_idx=t_w,
                               state=state.copy(), hist=hist_sel, prev=prev)

        scen = np.asarray(selection_forecast(t_w), dtype=np.float64)
        if scen.shape != (FORECAST_DAYS, n_sku, N_SCEN):
            raise ValueError(
                f"selection forecast shape {scen.shape} != "
                f"({FORECAST_DAYS}, {n_sku}, {N_SCEN})")
        scen_sel = scen[:SELECTION_DAYS]

        sel_t0 = time.perf_counter()
        actions = list(candidate_provider(w, ctx))
        evals: list[CandidateEvaluation] = []
        budget_exceeded = False
        for slot, action in enumerate(actions):
            if (selection_time_budget_s is not None
                    and time.perf_counter() - sel_t0
                    > selection_time_budget_s):
                budget_exceeded = True
                break
            evals.append(evaluate_candidate(
                slot, action, scen_sel, state, hist_sel, series_ids,
                params, k_g, cfg, t0_idx=t_w + 1, csl_floor=floor,
            ))
        winner = select_candidate(evals)
        selection_time_s = time.perf_counter() - sel_t0
        fallback = winner is None
        action = no_action if fallback else winner.action

        policy = AgentA4Policy(
            action, series_ids,
            action_start_idx=t_w + 1,
            tol_rel=cfg.feasibility_tol_rel,
            capacity_mode="scale",
        )
        state, days = run_window(
            state, test_demand[WEEK_DAYS * w: WEEK_DAYS * (w + 1)],
            hist_sel, policy, params, k_g=k_g, cfg=cfg, t0_idx=t_w + 1,
        )
        days_all.extend(days)
        week_logs.append(WeekLog(
            week=w + 1,
            origin_idx=t_w,
            gate=gate_stats(evals),
            fallback=fallback,
            selected_slot=None if fallback else winner.slot,
            selected_tau=action.tau_eff,
            selected_multipliers=action.multipliers,
            selected_mean_cost=float("nan") if fallback else winner.mean_cost,
            scaling=policy.scaling_summary(),
            evaluations=tuple(evals),
            selection_time_s=selection_time_s,
            candidates_provided=len(actions),
            budget_exceeded=budget_exceeded,
        ))

    terminal = terminal_cost(state, params.h)
    records = _aggregate_weeks(days_all, params, cfg, terminal, fold,
                               group_id, policy_name)
    return AgentWindowResult(
        fold=fold, group_id=group_id, policy=policy_name, records=records,
        window_cost=sum(r.holding_cost + r.shortage_cost
                        for r in days_all) + terminal,
        terminal=terminal,
        n_days=len(days_all),
        weeks=tuple(week_logs),
        days=tuple(days_all),
    )
