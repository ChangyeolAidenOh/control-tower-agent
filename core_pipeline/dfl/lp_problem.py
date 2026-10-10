"""2-G: learning LP (A5 perfect-information policy), spec v0.1 section 12.

The DFL training loss (Stage 4) and the SimPy-replacement engine share ONE
economic structure; this LP is that structure written as an optimization
problem: 28-day horizon, bundle-shared ordering capacity, SKU-specific h/b,
epsilon = 0 (pure LP), Clarabel. Solved with the realized demand path it is
the perfect-information policy A5, which T6 uses to pin LP == engine < 0.1%.

Formulation (one bundle, window days j = 0..T-1, lead time L):
  variables   q[j] >= 0 (orders), u[j] >= 0 (lost sales), I_eod[j] >= 0
  arrivals    arr[0] = 0 (already inside I0); arr[j] = P0[j-1] for j <= L;
              arr[j] = q[j-L-1] for j > L            (order day j' -> j'+L+1)
  balance     I_eod[j] = I_start[j] - d[j] + u[j],  I_start[j+1] = I_eod[j] + arr[j+1]
  capacity    sum_i(w_i q[i, j]) <= K_g             for every day j
  objective   sum_j (h . I_eod[j] + b . u[j]) + terminal
  terminal    h . (I_eod[T-1] + every undelivered unit)   -- identical to the
              engine's post-final-transition valuation (q_T included), so the
              two implementations price the boundary the same way.

With b > 0 the LP split I_eod - u = I_start - d reproduces lost sales
s = min(d, I) exactly at the optimum (raising u and I_eod together costs
h + b > 0), so no integer or complementarity machinery is needed.

Determinism note: Clarabel is deterministic for a fixed problem; A5 ties
(degenerate optima) are possible in principle but T6 compares costs, which
are unique at the optimum.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

SOLVER = cp.CLARABEL


@dataclass(frozen=True)
class A5Solution:
    orders: np.ndarray        # (T, n_sku) optimal order schedule
    objective: float          # LP optimal cost incl. terminal valuation
    status: str


def solve_a5(
    demand: np.ndarray,       # (T, n_sku) realized demand path (perfect info)
    inv0: np.ndarray,         # (n_sku,) on-hand at start of window day 0
    pipeline0: np.ndarray,    # (n_sku, L) in-transit; [:, k] arrives day k+1
    h: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    k_g: float,
) -> A5Solution:
    n_days, n_sku = demand.shape
    lead = pipeline0.shape[1]

    q = cp.Variable((n_days, n_sku), nonneg=True)
    u = cp.Variable((n_days, n_sku), nonneg=True)
    inv_eod = cp.Variable((n_days, n_sku), nonneg=True)

    constraints = []
    inv_start = inv0
    for j in range(n_days):
        constraints.append(inv_eod[j] == inv_start - demand[j] + u[j])
        constraints.append(w @ q[j] <= k_g)
        if j + 1 < n_days:
            if j + 1 <= lead:
                arrival = pipeline0[:, j]
            else:
                arrival = q[j - lead]
            inv_start = inv_eod[j] + arrival

    daily_cost = cp.sum(inv_eod @ h) + cp.sum(u @ b)

    # Terminal valuation: everything on hand or still undelivered after the
    # final midnight transition, charged h x 1 day (engine terminal_cost).
    undelivered = 0
    for k in range(lead):
        if n_days - 1 < k + 1:                      # initial slot arrives past T
            undelivered = undelivered + pipeline0[:, k]
    for j in range(n_days):
        if j + lead + 1 > n_days - 1:               # order arrives past day T-1
            undelivered = undelivered + q[j]
    terminal = (inv_eod[n_days - 1] + undelivered) @ h

    problem = cp.Problem(cp.Minimize(daily_cost + terminal), constraints)
    problem.solve(solver=SOLVER)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"A5 LP not solved: {problem.status}")
    return A5Solution(
        orders=np.asarray(q.value), objective=float(problem.value),
        status=problem.status,
    )


# --------------------------------------------------------------------------- #
# ST6-08: A3 open-loop SAA -- shared q, per-scenario inventory/lost-sales.
# Parameters = demand scenarios + inv0 + pipeline0; h/b/w/k_g baked, so one
# compiled problem per (bundle, r): 10 x {3, 5, 10} = 30 objects, cached.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class A3SaaSolution:
    orders: np.ndarray        # (T, n_sku) shared order schedule
    objective: float          # (1/S) sum_s [daily + terminal_s]
    status: str


class A3SaaProblem:
    """Compiled SAA LP (ST6-08 (2)-(3)): q shared across scenarios,
    u/inv_eod per scenario, transition j -> j+L+1 and terminal valuation
    identical to solve_a5, replicated per scenario."""

    def __init__(self, h: np.ndarray, b: np.ndarray, w: np.ndarray,
                 k_g: float, horizon: int, lead: int, n_scen: int):
        h = np.asarray(h, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        n = int(h.shape[0])
        self.n_scen = int(n_scen)
        self.horizon = int(horizon)
        self.d = [cp.Parameter((horizon, n), nonneg=True)
                  for _ in range(n_scen)]
        self.inv0 = cp.Parameter(n, nonneg=True)
        self.pipeline0 = cp.Parameter((n, lead), nonneg=True)
        q = cp.Variable((horizon, n), nonneg=True)

        constraints = []
        undelivered = 0
        for k in range(lead):
            if horizon - 1 < k + 1:
                undelivered = undelivered + self.pipeline0[:, k]
        for j in range(horizon):
            constraints.append(w @ q[j] <= k_g)
            if j + lead + 1 > horizon - 1:
                undelivered = undelivered + q[j]

        total = 0
        for s in range(n_scen):
            u = cp.Variable((horizon, n), nonneg=True)
            inv_eod = cp.Variable((horizon, n), nonneg=True)
            inv_start = self.inv0
            for j in range(horizon):
                constraints.append(
                    inv_eod[j] == inv_start - self.d[s][j] + u[j])
                if j + 1 < horizon:
                    arrival = (self.pipeline0[:, j] if j + 1 <= lead
                               else q[j - lead])
                    inv_start = inv_eod[j] + arrival
            total = total + cp.sum(inv_eod @ h) + cp.sum(u @ b)
            total = total + (inv_eod[horizon - 1] + undelivered) @ h

        self._q = q
        self.problem = cp.Problem(cp.Minimize(total / n_scen), constraints)

    def solve(self, demand_scen: np.ndarray, inv0: np.ndarray,
              pipeline0: np.ndarray) -> A3SaaSolution:
        demand_scen = np.asarray(demand_scen, dtype=np.float64)
        if demand_scen.shape[0] != self.n_scen:
            raise ValueError(
                f"expected {self.n_scen} scenarios, got {demand_scen.shape}")
        for s in range(self.n_scen):
            self.d[s].value = demand_scen[s]
        self.inv0.value = np.asarray(inv0, dtype=np.float64)
        self.pipeline0.value = np.asarray(pipeline0, dtype=np.float64)
        self.problem.solve(solver=SOLVER)
        status = str(self.problem.status)
        orders = (np.asarray(self._q.value, dtype=np.float64)
                  if self._q.value is not None
                  else np.full((self.horizon, self.inv0.shape[0]), np.nan))
        obj = (float(self.problem.value)
               if self.problem.value is not None else float("nan"))
        return A3SaaSolution(orders=orders, objective=obj, status=status)


def get_a3_saa_problem(cache: dict, key, h, b, w, k_g: float,
                       horizon: int, lead: int,
                       n_scen: int) -> A3SaaProblem:
    """Compile-once cache, intended key = (bundle_id, r)."""
    if key not in cache:
        cache[key] = A3SaaProblem(h, b, w, k_g, horizon, lead, n_scen)
    return cache[key]