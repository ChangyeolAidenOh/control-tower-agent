"""Stage 6 (ST6-11): A-PTO base-stock policy over CatBoost replayed
forecasts.

Contract (ST6-11 (3)-(4)):
  target   S_i = max(0, sum_{h=1..3} max(0, dhat_ih) + Qhat_tau(i)) --
           3-day protection window matching the A4 base-stock contract
           (order placed day t arrives morning t+3), per-SKU empirical
           residual quantile from outputs/apto/safety_quantiles.json;
  position IP_i = inv_eod_i + pipeline_i[0] + pipeline_i[1];
  request  q~_i = (S_i - IP_i)+;
  capacity if fsum(w q~) > K_g the whole vector is scaled by
           K_g / fsum(w q~) INSIDE the policy -- a deliberately different
           contract from the LP policies' hard stop (ST6-01 (3)); the
           pre-scale request, the ratio and an active flag are recorded
           in the diagnostics so the two contracts stay distinguishable;
  tau      keyed by the ENVIRONMENT r (r3/r5/r10): r=10 re-queries
           tau=10/11 on the same frozen residual distribution.

Separate module from lp_policy because the capacity contract differs;
the replayer, numeric hygiene and error types are shared imports.
"""

import json
import math
from pathlib import Path

import numpy as np

from core_pipeline.data.db import PROJECT_ROOT
from core_pipeline.policies.lp_policy import (
    FORECAST_DAYS, PolicyError, _finalize_orders,
)
from core_pipeline.simulator.engine import Observation

SAFETY_PATH = PROJECT_ROOT / "outputs" / "apto" / "safety_quantiles.json"
PROTECT_DAYS = 3


def load_safety_quantiles(r: int, series_ids,
                          path: Path | None = None) -> dict[str, float]:
    """Per-SKU Qhat_tau for environment r in {3, 5, 10}; every requested
    series must be present (missing series is a hard error, no default)."""
    p = Path(path) if path is not None else SAFETY_PATH
    payload = json.loads(p.read_text())
    if int(payload.get("protection_days", -1)) != PROTECT_DAYS:
        raise ValueError(
            f"{p.name}: protection_days "
            f"{payload.get('protection_days')!r} != {PROTECT_DAYS} "
            "(ST6-11 (3) D+3 contract)")
    key = f"r{int(r)}"
    table = payload.get("quantiles", {}).get(key)
    if table is None:
        raise ValueError(f"{p.name}: no quantile table for {key!r}")
    out: dict[str, float] = {}
    missing = []
    for s in series_ids:
        v = table.get(str(s))
        if v is None:
            missing.append(str(s))
        else:
            out[str(s)] = float(v)
    if missing:
        raise ValueError(
            f"{p.name}[{key}]: missing quantiles for "
            f"{len(missing)} series (first: {missing[:3]})")
    return out


class APTOPolicy:
    """Rolling 3-day base-stock policy (engine Policy protocol)."""

    def __init__(self, name: str, replayer, series_ids,
                 quantiles: dict[str, float]):
        self.name = name
        self._replayer = replayer
        self._series_ids = tuple(str(s) for s in series_ids)
        try:
            self._q_tau = np.asarray(
                [float(quantiles[s]) for s in self._series_ids],
                dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                f"{name}: no safety quantile for series "
                f"{exc.args[0]!r}") from None
        if not np.isfinite(self._q_tau).all():
            raise ValueError(f"{name}: non-finite safety quantiles")
        self._diag: dict[str, float] = {}
        self.n_solves = 0

    def decide(self, obs: Observation) -> np.ndarray:
        n_sku = len(self._series_ids)
        forecast = np.asarray(
            self._replayer.get(obs.t_idx, self._series_ids),
            dtype=np.float64)
        if forecast.shape != (FORECAST_DAYS, n_sku):
            raise PolicyError(
                f"{self.name}: forecast shape {forecast.shape} != "
                f"({FORECAST_DAYS}, {n_sku})")

        d3 = np.maximum(forecast[:PROTECT_DAYS, :], 0.0).sum(axis=0)
        s_target = np.maximum(0.0, d3 + self._q_tau)
        pipeline = np.asarray(obs.pipeline, dtype=np.float64)
        ip = (np.asarray(obs.inv, dtype=np.float64)
              + pipeline[:, 0] + pipeline[:, 1])
        q_req = np.maximum(0.0, s_target - ip)

        w = np.asarray(obs.w, dtype=np.float64)
        cap = float(obs.k_g)
        req_sum = math.fsum((w * q_req).tolist())
        if req_sum > cap:
            scale = cap / req_sum
            q_raw = q_req * scale
            scaling_active = 1.0
        else:
            scale = 1.0
            q_raw = q_req
            scaling_active = 0.0

        q, diag = _finalize_orders(
            self.name, q_raw, w, cap, obs.t_idx, fhash="apto")
        self.n_solves += 1
        self._diag = {
            "requested_weighted_sum": float(req_sum),
            "scale_ratio": float(scale),
            "scaling_active": scaling_active,
            "n_pos_requests": float(int((q_req > 0.0).sum())),
            **diag,
        }
        return q

    def last_diagnostics(self) -> dict[str, float]:
        return dict(self._diag)