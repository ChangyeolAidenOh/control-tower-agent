"""A4: tuned base-stock policy (preregistration section 4).

Decision rule at day t (after close), per SKU i in bundle g:
  S_{i,t} = Q_tau( 3-day cumulative demand | F_t )        protection interval
  q_raw_i = max(0, S_{i,t} - (I_eod_i + sum_k P_i[k]))    order-up-to on
                                                          inventory position
then, if sum_i(w_i * q_raw_i) > K_g, the pre-registered allocation rule
(proportional radial scaling, or shortage-cost-weighted shortfall priority)
reduces the orders to feasibility INSIDE the policy, so the engine's strict
constraint contract holds (the engine never repairs policy output).

Review decisions (2026-10-08):
- Protection interval = L + 1 = 3 days: the order placed on day D arrives on
  the morning of D+3 and the next order on D+4, so D's order must carry the
  inventory position over days D+1..D+3. This is a base-stock heuristic on the
  inventory position, not a claim that q_t itself serves D+1/D+2 demand, and
  under lost sales it is practical, not provably optimal.
- Target-level estimate: expanding-window empirical quantile of 3-day rolling
  sums over the ACTIVITY period observed up to t. Activity starts at the first
  positive-sale day; zero-sale days after that are included; rolling windows
  that reach before the activity start are excluded; no future data. Quantile
  interpolation fixed to numpy method "linear". Overlapping 3-day sums are not
  independent samples -- the empirical distribution is a forecasting heuristic
  and no independence claim is made.
- Fallback (fewer than 3 observed active days, not expected on subset v2 in
  the validation/test windows): tau-quantile of daily active demand x 3.
- kappa-selection reference A4 (2-E, fixed BEFORE any binding-share results):
  tau_ref = 5/6 (= b/(h+b) at r = 5), proportional allocation, the quantile
  machinery above, no tuning. Final A4 is tuned once per r in {3, 5} at the
  frozen kappa (2-D-2); kappa is never re-selected afterwards (Step D).

A4FromTargets evaluates pre-computed target levels through the SAME order-up-to
and allocation code path; it exists so the 2-D-2 tuning grid can compute all
tau quantiles in one pass. test_a4.py pins its equality with A4BaseStock.
"""

from dataclasses import dataclass

import numpy as np

from core_pipeline.simulator.engine import Observation, project_orders, weighted_sum

TAU_REFERENCE = 5.0 / 6.0
PROTECTION_DAYS = 3
QUANTILE_METHOD = "linear"

ALLOCATION_PROPORTIONAL = "proportional"
ALLOCATION_B_PRIORITY = "b_priority"
ALLOCATIONS = (ALLOCATION_PROPORTIONAL, ALLOCATION_B_PRIORITY)


def _activity_series(series: np.ndarray) -> np.ndarray | None:
    positive = np.nonzero(series > 0)[0]
    if positive.size == 0:
        return None
    return series[positive[0]:]


def target_level_grid(series: np.ndarray, taus: np.ndarray,
                      protection_days: int = PROTECTION_DAYS,
                      quantile_method: str = QUANTILE_METHOD) -> np.ndarray:
    """Expanding-window empirical quantiles of protection-interval demand for
    a vector of tau values at once (single np.quantile call). Returns an array
    of shape (len(taus),). Single source of truth for the target level."""
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    active = _activity_series(series)
    if active is None:
        return np.zeros(taus.shape)
    if active.size >= protection_days:
        csum = np.concatenate(([0.0], np.cumsum(active)))
        sums = csum[protection_days:] - csum[:-protection_days]
        return np.quantile(sums, taus, method=quantile_method)
    return np.quantile(active, taus, method=quantile_method) * protection_days


def target_level(series: np.ndarray, tau: float, protection_days: int = PROTECTION_DAYS,
                 quantile_method: str = QUANTILE_METHOD) -> float:
    """Scalar wrapper over target_level_grid (same code path)."""
    return float(target_level_grid(series, np.array([tau]), protection_days,
                                   quantile_method)[0])


def allocate_b_priority(q_raw: np.ndarray, b: np.ndarray, w: np.ndarray,
                        k_g: float) -> np.ndarray:
    """Shortage-cost-weighted shortfall-priority allocation: SKUs receive their
    requested shortfall q_raw in descending b_i order until capacity K_g is
    exhausted; the marginal SKU receives the remainder, later SKUs receive 0.
    Deterministic tie-break on equal b_i: ascending SKU index. Operational
    definition recorded in the decision log (the preregistration fixes only
    the rule's name)."""
    order = np.lexsort((np.arange(b.size), -b))
    q = np.zeros_like(q_raw)
    remaining = k_g
    for i in order:
        if remaining <= 0.0:
            break
        give = min(q_raw[i] * w[i], remaining)
        q[i] = give / w[i]
        remaining -= give
    if weighted_sum(q, w) > k_g:
        q = project_orders(q, w, k_g)
    return q


def apply_allocation(q_raw: np.ndarray, allocation: str, obs: Observation) -> np.ndarray:
    """Shared feasibility step for every A4 variant."""
    if weighted_sum(q_raw, obs.w) <= obs.k_g:
        return q_raw
    if allocation == ALLOCATION_PROPORTIONAL:
        return project_orders(q_raw, obs.w, obs.k_g)
    return allocate_b_priority(q_raw, obs.b, obs.w, obs.k_g)


@dataclass(frozen=True)
class A4Config:
    tau: float
    allocation: str              # "proportional" | "b_priority"
    protection_days: int = PROTECTION_DAYS
    quantile_method: str = QUANTILE_METHOD


class A4BaseStock:
    """Policy protocol implementation; submits feasible orders by construction."""

    def __init__(self, config: A4Config):
        if config.allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation rule: {config.allocation}")
        self.config = config
        self.name = f"a4_tau{config.tau:.4f}_{config.allocation}"

    def decide(self, obs: Observation) -> np.ndarray:
        n_sku = obs.demand_hist.shape[1]
        s_target = np.empty(n_sku)
        for i in range(n_sku):
            s_target[i] = target_level(
                obs.demand_hist[:, i],
                self.config.tau,
                self.config.protection_days,
                self.config.quantile_method,
            )
        inventory_position = obs.inv + obs.pipeline.sum(axis=1)
        q_raw = np.maximum(0.0, s_target - inventory_position)
        self._last_diag = {"desired_sum": weighted_sum(q_raw, obs.w)}
        return apply_allocation(q_raw, self.config.allocation, obs)

    def last_diagnostics(self) -> dict[str, float]:
        """Pre-allocation raw order demand of the LAST decide call (2-H metric:
        share of days where the un-allocated request exceeds K_g). Optional
        protocol -- only A4-family policies expose it (decision log 2-C)."""
        return self._last_diag


class A4FromTargets:
    """A4 with pre-computed target levels S_{i,t} (tuning fast path, 2-D-2).

    targets: (n_days, n_sku), row j = targets for absolute day t0_idx + j,
    computed with target_level_grid on the same expanding histories. Order-up-to
    and allocation are shared with A4BaseStock via apply_allocation."""

    def __init__(self, targets: np.ndarray, t0_idx: int, allocation: str):
        if allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation rule: {allocation}")
        self.targets = targets
        self.t0_idx = t0_idx
        self.allocation = allocation
        self.name = f"a4_from_targets_{allocation}"

    def decide(self, obs: Observation) -> np.ndarray:
        s_target = self.targets[obs.t_idx - self.t0_idx]
        inventory_position = obs.inv + obs.pipeline.sum(axis=1)
        q_raw = np.maximum(0.0, s_target - inventory_position)
        self._last_diag = {"desired_sum": weighted_sum(q_raw, obs.w)}
        return apply_allocation(q_raw, self.allocation, obs)

    def last_diagnostics(self) -> dict[str, float]:
        return self._last_diag


def make_reference_a4() -> A4BaseStock:
    """Fixed kappa-selection reference (2-E): tau = 5/6, proportional, untuned."""
    return A4BaseStock(A4Config(tau=TAU_REFERENCE, allocation=ALLOCATION_PROPORTIONAL))
