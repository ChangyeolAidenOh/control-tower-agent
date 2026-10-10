"""ST5-04: shared decision-time LP window mapping (29-day zero-padding).

Every LP-based policy that decides q_t at the end of day t builds its LP
window identically: internal day 0 = the decision day with zero demand
(day t's demand is already realized into inv_eod), internal days 1..28 =
the 28-day forecast for t+1..t+28; inv0 = end-of-day on-hand, pipeline0 =
P_t before the midnight shift. Under lp_problem's transition the first
decision q[0] arrives the morning of internal day 3 = t+3 (D -> D+3),
pipeline slots 0/1 arrive t+1/t+2, and the last three orders q[26..28]
cannot land inside the window (they arrive days 29/30/31) so a positive-h
optimum sets them to zero, priced only by the terminal valuation.

This mapping does NOT change the pre-registered 28-day planning horizon;
it aligns the current decision with the LP's first decision variable.
A5's window-level perfect-information use (solve_a5 on a test window with
day-0 demand realized inside the LP, the T6 anchor) shares the same
transition and order time axis; only the day-0 bookkeeping differs. The
two uses must not be mixed: decision-time callers (B training layer, B
inference solve_a5, later A1/A2/A3 evaluation policies) go through this
function only.
"""

import numpy as np

FORECAST_DAYS = 28
LP_WINDOW_DAYS = FORECAST_DAYS + 1


def to_lp_window(forecast_28: np.ndarray, inv_eod: np.ndarray,
                 pipeline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a decision-time state to solve_a5 / layer inputs.

    forecast_28: (28, n_sku) demand forecast for days t+1..t+28
    inv_eod:     (n_sku,) on-hand after day t's realization
    pipeline:    (n_sku, L) in-transit before day t's midnight shift

    Returns (demand_29, inv0, pipeline0) with demand_29[0] = 0.
    """
    forecast_28 = np.asarray(forecast_28, dtype=np.float64)
    if forecast_28.ndim != 2 or forecast_28.shape[0] != FORECAST_DAYS:
        raise ValueError(f"forecast must be ({FORECAST_DAYS}, n_sku), "
                         f"got {forecast_28.shape}")
    zero = np.zeros((1, forecast_28.shape[1]), dtype=np.float64)
    demand_29 = np.concatenate([zero, forecast_28], axis=0)
    return (demand_29,
            np.asarray(inv_eod, dtype=np.float64),
            np.asarray(pipeline, dtype=np.float64))
