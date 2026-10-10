"""Stage 5 (B/DFL): differentiable bundle LP layer and downstream cost.

Contract source: core_pipeline/dfl/lp_problem.py (solve_a5) and the NumPy
engine v1 -- an order placed on day j arrives the morning of day j+L+1
(D -> D+3 for L=2), and the terminal valuation charges h_i x 1 day on the
final end-of-day inventory plus every undelivered unit (initial pipeline
slots landing past the window and the last L+1 days' orders). The 4-B
spike's j -> j+L convention is NOT reused (ST5-01/ST5-03 P0).

Layers are compiled once per (bundle, r): SKU costs h/b, unit weights w and
the capacity K_g are baked as constants; cvxpy Parameters are the demand
path, the initial on-hand and the in-transit pipeline. eps = 0 (pure LP),
Clarabel via diffcp.
"""

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from core_pipeline.dfl.lp_problem import solve_a5

SOLVER_ARGS = {"solve_method": "Clarabel"}


def build_bundle_layer(h: np.ndarray, b: np.ndarray, w: np.ndarray,
                       k_g: float, horizon: int, lead: int) -> CvxpyLayer:
    """Compile the solve_a5 LP as a CvxpyLayer for one bundle and one r.

    Parameters (in call order): d_hat (horizon, n_sku), inv0 (n_sku,),
    pipeline0 (n_sku, lead). Returns the order schedule q (horizon, n_sku).
    """
    n_sku = int(h.shape[0])
    d_hat = cp.Parameter((horizon, n_sku), nonneg=True)
    inv0 = cp.Parameter(n_sku, nonneg=True)
    pipeline0 = cp.Parameter((n_sku, lead), nonneg=True)
    q = cp.Variable((horizon, n_sku), nonneg=True)
    u = cp.Variable((horizon, n_sku), nonneg=True)
    inv_eod = cp.Variable((horizon, n_sku), nonneg=True)

    constraints = []
    inv_start = inv0
    for j in range(horizon):
        constraints.append(inv_eod[j] == inv_start - d_hat[j] + u[j])
        constraints.append(w @ q[j] <= k_g)
        if j + 1 < horizon:
            arrival = pipeline0[:, j] if j + 1 <= lead else q[j - lead]
            inv_start = inv_eod[j] + arrival

    daily_cost = cp.sum(inv_eod @ h) + cp.sum(u @ b)

    undelivered = 0
    for k in range(lead):
        if horizon - 1 < k + 1:
            undelivered = undelivered + pipeline0[:, k]
    for j in range(horizon):
        if j + lead + 1 > horizon - 1:
            undelivered = undelivered + q[j]
    terminal = (inv_eod[horizon - 1] + undelivered) @ h

    problem = cp.Problem(cp.Minimize(daily_cost + terminal), constraints)
    assert problem.is_dpp(), "bundle LP is not DPP"
    return CvxpyLayer(problem, parameters=[d_hat, inv0, pipeline0], variables=[q])


class LpLayerCache:
    """One compiled layer per key, intended key = (bundle_id, r): 10 x 2 = 20."""

    def __init__(self):
        self._layers: dict = {}

    def get(self, key, h: np.ndarray, b: np.ndarray, w: np.ndarray,
            k_g: float, horizon: int, lead: int) -> CvxpyLayer:
        if key not in self._layers:
            self._layers[key] = build_bundle_layer(h, b, w, k_g, horizon, lead)
        return self._layers[key]

    def __len__(self) -> int:
        return len(self._layers)


def downstream_cost(q: torch.Tensor, y: torch.Tensor, inv0: torch.Tensor,
                    pipeline0: torch.Tensor, h: torch.Tensor,
                    b: torch.Tensor) -> torch.Tensor:
    """Engine-convention realized cost of an order plan q under demand path y.

    All tensors float64. q, y: (horizon, n_sku); inv0: (n_sku,);
    pipeline0: (n_sku, lead). Day j's opening inventory already includes
    that morning's arrival (pipeline0[:, j-1] for 1 <= j <= lead, then
    q[j-lead-1]), matching run_window where the previous midnight
    transition folded the arrival in. Terminal charges h x 1 day on the
    final end-of-day inventory plus every undelivered unit -- the same set
    as solve_a5's terminal. Capacity feasibility of q is the LP's job, not
    this function's. Piecewise linear (relu kinks), differentiable a.e.
    """
    horizon, n_sku = q.shape
    lead = pipeline0.shape[1]
    inv = inv0
    total = torch.zeros((), dtype=torch.float64)
    for j in range(horizon):
        if j == 0:
            inv_start = inv
        elif j <= lead:
            inv_start = inv + pipeline0[:, j - 1]
        else:
            inv_start = inv + q[j - lead - 1]
        unmet = torch.relu(y[j] - inv_start)
        inv_eod = torch.relu(inv_start - y[j])
        total = total + inv_eod @ h + unmet @ b
        inv = inv_eod
    undelivered = torch.zeros(n_sku, dtype=torch.float64)
    for k in range(lead):
        if horizon - 1 < k + 1:
            undelivered = undelivered + pipeline0[:, k]
    for j in range(horizon):
        if j + lead + 1 > horizon - 1:
            undelivered = undelivered + q[j]
    return total + (inv + undelivered) @ h


def oracle_cost(y: np.ndarray, inv0: np.ndarray, pipeline0: np.ndarray,
                h: np.ndarray, b: np.ndarray, w: np.ndarray,
                k_g: float) -> tuple[float, np.ndarray, float]:
    """Perfect-information reference for the regret surrogate.

    solve_a5 on the realized path, then the oracle plan is priced through
    downstream_cost so both regret terms share ONE cost functional (the
    regret >= 0 invariant holds only under this construction). Returns
    (cost, orders, lp_objective); |cost - lp_objective| is a parity
    diagnostic -- the lost-sales split is exact at the LP optimum.
    """
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    y, inv0, pipeline0, h, b, w = map(f64, (y, inv0, pipeline0, h, b, w))
    sol = solve_a5(y, inv0, pipeline0, h, b, w, float(k_g))
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
    cost = float(downstream_cost(t(sol.orders), t(y), t(inv0), t(pipeline0),
                                 t(h), t(b)).item())
    return cost, sol.orders, sol.objective
