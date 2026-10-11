"""ST7-02 D2: agent action applicator (tau' -> A4 -> SKU multipliers).

Execution order fixed by ST7-01: apply tau' (A4Config swap on the existing
implementation, no new formula) -> A4 produces its order with its internal
proportional feasibility INCLUDED (Track A code path, a4_basestock.py) ->
multiply per-SKU m_i -> capacity handling by mode:

  raw   (gate use): q_raw := post-allocation A4 order x multipliers
        (ST7-02 D2 definition: the pre-scaling quantity). Any weighted sum
        beyond K_g * (1 + tol_rel) raises CapacityViolation; the gate catches
        it and rejects the candidate. No projection here, so capacity
        violations are actually rejectable.
  scale (realized path): the approved action meets unexpected violations
        (realized demand differs from the selection scenarios) with
        policy-internal proportional scaling (ST6-11 A-PTO precedent),
        recorded per day. The engine's strict contract stays intact
        (ST6-01): the submitted order is feasible by construction.

Outside the action window the policy IS the base A4 (base tau, all
multipliers 1.0): used for the 3-day continuation segment (ST7-02 D4) and
as the no-action parity reference. One instance covers one candidate
evaluation or one realized week; build a fresh instance per run (the
scaling counters are per-instance).
"""

import numpy as np

from core_pipeline.policies.a4_basestock import (
    ALLOCATION_PROPORTIONAL,
    A4BaseStock,
    A4Config,
)
from core_pipeline.simulator.engine import Observation, project_orders, weighted_sum

from core_pipeline.agent.candidates import AgentAction

ACTION_DAYS = 7
CAPACITY_RAW = "raw"
CAPACITY_SCALE = "scale"
CAPACITY_MODES = (CAPACITY_RAW, CAPACITY_SCALE)


class CapacityViolation(RuntimeError):
    """Raised in raw mode when the multiplied order exceeds shared capacity;
    the gate translates this into candidate rejection (ST7-02 D2)."""

    def __init__(self, t_idx: int, requested: float, k_g: float):
        super().__init__(
            f"agent order exceeds capacity at t={t_idx}: "
            f"sum(w*q_raw)={requested:.6f} > K_g={k_g:.6f}")
        self.t_idx = t_idx
        self.requested = requested
        self.k_g = k_g


class AgentA4Policy:
    """Policy-protocol implementation wrapping the frozen A4 code path."""

    def __init__(
        self,
        action: AgentAction,
        series_ids,
        *,
        action_start_idx: int,
        tol_rel: float,
        capacity_mode: str,
        action_days: int = ACTION_DAYS,
        allocation: str = ALLOCATION_PROPORTIONAL,
    ):
        if capacity_mode not in CAPACITY_MODES:
            raise ValueError(f"unknown capacity_mode: {capacity_mode}")
        self.action = action
        self.series_ids = tuple(str(s) for s in series_ids)
        self.action_start_idx = int(action_start_idx)
        self.action_days = int(action_days)
        self.tol_rel = float(tol_rel)
        self.capacity_mode = capacity_mode
        self._a4_action = A4BaseStock(
            A4Config(tau=action.tau_eff, allocation=allocation))
        self._a4_base = A4BaseStock(
            A4Config(tau=action.base_tau, allocation=allocation))
        mult_map = action.multiplier_map()
        self._mult = np.array(
            [mult_map.get(sid, 1.0) for sid in self.series_ids],
            dtype=np.float64)
        unknown = set(mult_map) - set(self.series_ids)
        if unknown:
            raise ValueError(f"multiplier for unknown series: {sorted(unknown)}")
        self.name = (f"agent_tau{action.tau_eff:.2f}"
                     f"_m{action.n_modified_skus}_{capacity_mode}")
        self.internal_scaling_days = 0
        self.total_scaled_units = 0.0
        self.max_requested_capacity_ratio = 0.0
        self._diag: dict[str, float] = {}

    def _in_window(self, t_idx: int) -> bool:
        return (self.action_start_idx <= t_idx
                < self.action_start_idx + self.action_days)

    def decide(self, obs: Observation) -> np.ndarray:
        if obs.demand_hist.shape[1] != len(self.series_ids):
            raise ValueError("observation sku dimension mismatch")
        if not self._in_window(obs.t_idx):
            q = self._a4_base.decide(obs)
            self._diag = {**self._a4_base.last_diagnostics(),
                          "agent_active": 0.0}
            return q

        q_a4 = self._a4_action.decide(obs)
        q_raw_agent = q_a4 * self._mult
        requested = weighted_sum(q_raw_agent, obs.w)
        ratio = requested / obs.k_g if obs.k_g > 0 else float("inf")
        self.max_requested_capacity_ratio = max(
            self.max_requested_capacity_ratio, ratio)
        tol = self.tol_rel * obs.k_g

        scaled = False
        if requested > obs.k_g + tol:
            if self.capacity_mode == CAPACITY_RAW:
                raise CapacityViolation(obs.t_idx, requested, obs.k_g)
            q_final = project_orders(q_raw_agent, obs.w, obs.k_g)
            self.internal_scaling_days += 1
            self.total_scaled_units += float(np.sum(q_raw_agent - q_final))
            scaled = True
        else:
            q_final = q_raw_agent

        self._diag = {
            **self._a4_action.last_diagnostics(),
            "agent_active": 1.0,
            "requested_capacity_ratio": ratio,
            "internal_scaling_applied": float(scaled),
        }
        return q_final

    def last_diagnostics(self) -> dict[str, float]:
        return dict(self._diag)

    def scaling_summary(self) -> dict[str, float]:
        """ST7-02 D2 realized-path record (scale mode; zeros in raw mode)."""
        return {
            "internal_scaling_days": float(self.internal_scaling_days),
            "total_scaled_units": float(self.total_scaled_units),
            "max_requested_capacity_ratio": float(
                self.max_requested_capacity_ratio),
        }