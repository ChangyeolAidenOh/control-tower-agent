"""L0: rule-based Control Tower baseline (2-H; plan section 4.7 "threshold
alert + fixed playbook", operationalized as an (s, S) reorder-point policy).

IMPORTANT CONTRACT (decision log, Stage 2-H): L0 is NOT a reproduction of any
real client's measured policy. It is the FICTIONAL client's explicitly assumed
fixed ordering playbook -- "a fixed reorder playbook using a 7-day
order-up-to level", not a claim about common industry practice (no public
evidence is cited for MA7). All constants are frozen before any AS-IS
diagnosis result is seen and are never tuned.

Rule (per SKU i, decision at the end of day t):
  d_hat_{i,t} = mean of the last 7 observed demand days (zero days included,
                no activity awareness -- deliberate contrast with A4)
  s_{i,t} = 3 x d_hat   (alert threshold: lead-time-plus-one cover)
  S_{i,t} = 7 x d_hat   (order-up-to: 7-day cover playbook)
  q_raw   = (S - IP)+ if IP < s else 0,  IP = end-of-day on-hand + pipeline
Orders are intermittent (only below the alert threshold), unlike A4's daily
base-stock. Capacity: proportional radial scaling INSIDE the policy -- the
engine never repairs L0 output. No last_diagnostics(): the pre-allocation
raw-demand metric is registered for the A4 family only.
"""

import numpy as np

from core_pipeline.simulator.engine import Observation, project_orders, weighted_sum

MA_DAYS = 7
REORDER_COVER_DAYS = 3
ORDER_UP_TO_COVER_DAYS = 7


class RuleControlTower:
    """Policy protocol implementation; submits feasible orders by construction."""

    name = "l0_rule_ct"

    def decide(self, obs: Observation) -> np.ndarray:
        hist = obs.demand_hist
        window = hist[-MA_DAYS:] if hist.shape[0] >= MA_DAYS else hist
        d_hat = window.mean(axis=0)
        ip = obs.inv + obs.pipeline.sum(axis=1)
        s = REORDER_COVER_DAYS * d_hat
        s_level = ORDER_UP_TO_COVER_DAYS * d_hat
        q_raw = np.where(ip < s, np.maximum(0.0, s_level - ip), 0.0)
        if weighted_sum(q_raw, obs.w) <= obs.k_g:
            return q_raw
        return project_orders(q_raw, obs.w, obs.k_g)
