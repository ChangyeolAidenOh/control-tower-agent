"""2-C: episode assembly -- warm-up inheritance, bundle x week aggregation,
terminal treatment variants (spec v0.1 sections 7-8).

Episode = bundle x week. Per fold x bundle, a policy run is:
  common warm-up state (14 days, the r-environment's tuned Track A A4;
  identical state inherited by EVERY compared policy, A4-SL included)
  -> 28-day test window with the policy under evaluation
  -> terminal valuation after the final midnight transition.

Aggregation (decision log 2-C):
- Weekly episode cost = sum over the week's days of h.I + b.u; the terminal
  valuation is assigned to the LAST week's episode so that the sum of episode
  costs equals the full window accounting (the bootstrap in preregistration
  section 6 operates on episode series, which must carry the whole cost).
- CSL(episode) = mean of SKU-day indicators 1[u == 0] over the bundle-week.
- fill rate = sum(s) / sum(d) over the episode (1.0 when the episode has zero
  demand); days of inventory = mean daily ending stock / mean daily demand
  (NaN when demand is zero) -- diagnostic only.
- guardrail_blocked = episode CSL < cfg.episode_csl_floor (L2 gate statistic,
  reported in 2-H; criteria unchanged).

Continuation sensitivity (T5, decision log 2-C): folds 1-11 only. The variant
runs the SAME policy for 7 more days of real demand and aggregates 35 days of
daily cost plus the terminal valuation at day 35 (same boundary rule, pushed
out); the report is the per-policy difference vs the core treatment and the
pairwise cost ranking under both treatments. Fold 12 has no post-window
demand and is excluded; no synthetic demand is fabricated.

Diagnostics: policies may expose last_diagnostics() (A4 family: pre-allocation
desired_sum). The runner wraps the policy in a recorder; the engine stays
untouched. raw_exceed_days counts window days with desired_sum > K_g.
"""

from dataclasses import dataclass

import numpy as np

from core_pipeline.simulator.engine import (
    BundleState,
    DayResult,
    Observation,
    Policy,
    SimConfig,
    SkuParams,
    init_warmup_state,
    run_window,
    terminal_cost,
)

WEEK_DAYS = 7


class DiagnosticsRecorder:
    """Transparent policy wrapper collecting optional per-day diagnostics."""

    def __init__(self, policy: Policy):
        self._policy = policy
        self.name = policy.name
        self.records: list[tuple[int, dict[str, float]]] = []

    def decide(self, obs: Observation) -> np.ndarray:
        q = self._policy.decide(obs)
        probe = getattr(self._policy, "last_diagnostics", None)
        if probe is not None:
            self.records.append((obs.t_idx, dict(probe())))
        return q


@dataclass(frozen=True)
class EpisodeRecord:
    fold: int
    group_id: str
    week: int                    # 1-based within the window
    policy: str
    cost: float                  # holding + shortage (+ terminal on last week)
    holding_cost: float
    shortage_cost: float
    terminal_cost: float         # 0.0 except on the last week
    csl: float                   # mean SKU-day 1[u == 0]
    fill_rate: float
    days_of_inventory: float     # NaN when the episode has zero demand
    guardrail_blocked: bool      # csl < cfg.episode_csl_floor
    binding_days: int
    n_days: int


@dataclass(frozen=True)
class WindowResult:
    fold: int
    group_id: str
    policy: str
    records: tuple[EpisodeRecord, ...]
    window_cost: float           # sum of daily costs + terminal
    terminal: float
    binding_days: int
    raw_exceed_days: int | None  # None when the policy exposes no diagnostics
    n_days: int


def compute_common_warmup(
    warmup_demand: np.ndarray,       # (warmup_days, n_sku), from v_fold_warmup
    pre_warmup_demand: np.ndarray,   # history before warm-up start (for A4 + init)
    warmup_policy: Policy,           # the r-environment's tuned Track A A4
    params: SkuParams,
    k_g: float,
    cfg: SimConfig,
    *,
    warmup_start_idx: int,
) -> BundleState:
    """Run the 14-day common warm-up once per (fold, bundle, r, scenario);
    every compared policy inherits the returned state (spec section 7)."""
    lookback = cfg.warmup_init_lookback_days
    recent_mean = pre_warmup_demand[-lookback:].mean(axis=0)
    state = init_warmup_state(recent_mean, cfg.lead_time_core, cfg)
    final, _ = run_window(
        state, warmup_demand, pre_warmup_demand, warmup_policy, params,
        k_g=k_g, cfg=cfg, t0_idx=warmup_start_idx,
    )
    return final


def _aggregate_weeks(
    days: list[DayResult], params: SkuParams, cfg: SimConfig, terminal: float,
    fold: int, group_id: str, policy_name: str,
) -> tuple[EpisodeRecord, ...]:
    n_days = len(days)
    n_weeks = (n_days + WEEK_DAYS - 1) // WEEK_DAYS
    records = []
    for wk in range(n_weeks):
        chunk = days[wk * WEEK_DAYS: (wk + 1) * WEEK_DAYS]
        holding = sum(r.holding_cost for r in chunk)
        shortage = sum(r.shortage_cost for r in chunk)
        term = terminal if wk == n_weeks - 1 else 0.0
        unmet = np.stack([r.unmet for r in chunk])
        sales = sum(float(r.sales.sum()) for r in chunk)
        demand = sum(float(r.demand.sum()) for r in chunk)
        inv_eod_mean = float(np.mean([r.inv_eod.sum() for r in chunk]))
        demand_mean = demand / len(chunk)
        csl = float(np.mean(unmet == 0.0))
        records.append(EpisodeRecord(
            fold=fold, group_id=group_id, week=wk + 1, policy=policy_name,
            cost=holding + shortage + term,
            holding_cost=holding, shortage_cost=shortage, terminal_cost=term,
            csl=csl,
            fill_rate=sales / demand if demand > 0 else 1.0,
            days_of_inventory=inv_eod_mean / demand_mean if demand_mean > 0 else float("nan"),
            guardrail_blocked=csl < cfg.episode_csl_floor,
            binding_days=sum(1 for r in chunk if r.binding),
            n_days=len(chunk),
        ))
    return tuple(records)


def run_policy_window(
    start_state: BundleState,
    test_demand: np.ndarray,         # (28, n_sku) or (35, n_sku) for continuation
    demand_hist: np.ndarray,         # everything before the test window (warm-up incl.)
    policy: Policy,
    params: SkuParams,
    k_g: float,
    cfg: SimConfig,
    *,
    t0_idx: int,
    fold: int,
    group_id: str,
) -> WindowResult:
    """One policy's evaluation run from the inherited warm-up state, including
    the terminal valuation after the final midnight transition."""
    recorder = DiagnosticsRecorder(policy)
    final, days = run_window(
        start_state.copy(), test_demand, demand_hist, recorder, params,
        k_g=k_g, cfg=cfg, t0_idx=t0_idx,
    )
    terminal = terminal_cost(final, params.h)
    records = _aggregate_weeks(days, params, cfg, terminal, fold, group_id,
                               policy.name)
    raw_exceed = None
    if recorder.records:
        raw_exceed = sum(1 for _, d in recorder.records
                         if d.get("desired_sum", 0.0) > k_g)
    return WindowResult(
        fold=fold, group_id=group_id, policy=policy.name, records=records,
        window_cost=sum(r.holding_cost + r.shortage_cost for r in days) + terminal,
        terminal=terminal,
        binding_days=sum(1 for r in days if r.binding),
        raw_exceed_days=raw_exceed,
        n_days=len(days),
    )


def continuation_report(
    core: dict[str, WindowResult], continuation: dict[str, WindowResult],
) -> dict:
    """T5 statistic: per-policy cost difference (continuation - core) and the
    pairwise cost ranking under both treatments, for one fold x bundle."""
    policies = sorted(core)
    diffs = {p: continuation[p].window_cost - core[p].window_cost
             for p in policies if p in continuation}
    rank_core = sorted(policies, key=lambda p: core[p].window_cost)
    rank_cont = sorted(diffs, key=lambda p: continuation[p].window_cost)
    return {
        "cost_diff": diffs,
        "rank_core": rank_core,
        "rank_continuation": rank_cont,
        "rank_preserved": rank_core == rank_cont if diffs else None,
    }
