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
