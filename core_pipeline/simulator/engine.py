"""Daily state-transition engine. Contract: docs/simulator_spec.md v0.1 section 2.

Day t order of events (spec section 2):
  1. demand realization: s_t = min(d_t, I_t), u_t = d_t - s_t (lost sales)
  2. end of day:        I_t <- I_t - s_t; costs h_i * I_t + b_i * u_t
  3. decision:          policy plans a rolling horizon on F_t, submits q_t only;
                        bundle constraint sum_i(w_i * q_i) <= K_g
  4. midnight:          I_{t+1} = I_t + P_t[0]; P_{t+1} = (P_t[1], ..., q_t)

P_t[k] arrives on the morning of day t+k+1; an order on day D arrives on the
morning of day D+3 under L = 2. All arrays are float64 of shape (n_sku,) within
one bundle; the bundle is the simulation unit (shared constraint), so folds x
bundles are independent runs.

Implementation decisions (Stage 2 decision log, 2026-10-08):
- NumPy synchronous loop replaces SimPy (development-stage change; the spec
  section 2 table is the only contract). Determinism requires identical runtime,
  operation order, and random streams -- the engine holds no global RNG and all
  stochastic demand paths are built outside and passed in as fixed arrays.
- The engine never silently repairs policy output. Core policies (A2/A3/B/A5)
  must submit feasible orders; violations beyond tolerance raise EngineError.
  project_orders() exists as a library function for policies that explicitly
  use radial scaling internally (A4) and for tests. q_raw, q_final,
  projection_applied, violation_amount are recorded per day.
- Terminal valuation happens once, after the final midnight transition:
  h . (I_{T+1} + sum(P_{T+1})), which equals h . (I_T_eod + P[0] + P[1] + q_T).
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import yaml


class EngineError(RuntimeError):
    """Raised when a policy submits an infeasible order beyond tolerance."""


class ViolationHandling(str, Enum):
    ERROR = "error"      # default for core policies: validate, fail on violation
    PROJECT = "project"  # explicit opt-in: radial scaling applied and recorded


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Seeds:
    train: int
    eval: int
    censor: int
    shock: int


@dataclass(frozen=True)
class SimConfig:
    seeds: Seeds
    test_window_days: int
    warmup_days: int
    planning_horizon_days: int
    lead_time_core: int
    lead_time_stress: tuple[int, ...]
    observation_mode: str
    censor_core_p: float
    censor_stress_grid: tuple[float, ...]
    censor_replacement_lookback_days: int
    shock_eval_only: bool
    shock_bundles_per_fold: int
    shock_duration_days: int
    shock_multiplier: float
    r_train: tuple[int, ...]
    r_eval: tuple[int, ...]
    fixed_order_cost_core: float
    fixed_order_cost_stress: float | None
    kappa_candidates: tuple[float, ...]
    kappa_selected: float | None
    binding_epsilon: float
    weight_wi: float
    feasibility_tol_rel: float
    warmup_policy: str
    warmup_init_lookback_days: int
    warmup_init_inventory_multiplier: float
    warmup_init_pipeline_slot_multiplier: float
    terminal_core: str
    continuation_sensitivity_days: int
    episode_csl_floor: float


def load_sim_config(path: str) -> SimConfig:
    """Parse configs/simulator.yaml into a frozen SimConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SimConfig(
        seeds=Seeds(**raw["seeds"]),
        test_window_days=raw["time"]["test_window_days"],
        warmup_days=raw["time"]["warmup_days"],
        planning_horizon_days=raw["time"]["planning_horizon_days"],
        lead_time_core=raw["lead_time"]["core"],
        lead_time_stress=tuple(raw["lead_time"]["stress"]),
        observation_mode=raw["observation"]["mode"],
        censor_core_p=raw["demand_censoring"]["core_p"],
        censor_stress_grid=tuple(raw["demand_censoring"]["stress_grid"]),
        censor_replacement_lookback_days=raw["demand_censoring"]["replacement_lookback_days"],
        shock_eval_only=raw["demand_shock"]["eval_only"],
        shock_bundles_per_fold=raw["demand_shock"]["bundles_per_fold"],
        shock_duration_days=raw["demand_shock"]["duration_days"],
        shock_multiplier=raw["demand_shock"]["multiplier"],
        r_train=tuple(raw["cost"]["r_train"]),
        r_eval=tuple(raw["cost"]["r_eval"]),
        fixed_order_cost_core=raw["cost"]["fixed_order_cost_core"],
        fixed_order_cost_stress=raw["cost"]["fixed_order_cost_stress"],
        kappa_candidates=tuple(raw["capacity"]["kappa_candidates"]),
        kappa_selected=raw["capacity"]["kappa_selected"],
        binding_epsilon=raw["capacity"]["binding_epsilon"],
        weight_wi=raw["capacity"]["weight_wi"],
        feasibility_tol_rel=raw["constraint_handling"]["feasibility_tol_rel"],
        warmup_policy=raw["warmup"]["policy"],
        warmup_init_lookback_days=raw["warmup"]["init_lookback_days"],
        warmup_init_inventory_multiplier=raw["warmup"]["init_inventory_multiplier"],
        warmup_init_pipeline_slot_multiplier=raw["warmup"]["init_pipeline_slot_multiplier"],
        terminal_core=raw["terminal"]["core"],
        continuation_sensitivity_days=raw["terminal"]["continuation_sensitivity_days"],
        episode_csl_floor=raw["guardrail"]["episode_csl_floor"],
    )


def spawn_stream(seed: int, *path: int) -> np.random.Generator:
    """Hierarchical random-stream construction: one Generator per identifier
    path, independent of execution order. Example: spawn_stream(cfg.seeds.shock,
    fold) for the fold-level shock-bundle draw; spawn_stream(cfg.seeds.censor,
    fold, bundle_idx) for per-bundle censor assignment."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *path])))


# --------------------------------------------------------------------------- #
# State and results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SkuParams:
    """Per-bundle SKU parameters, read from mart.sku_cost / mart.bundles."""
    series_ids: tuple[str, ...]
    h: np.ndarray            # (n_sku,) holding cost per unit-day
    b: np.ndarray            # (n_sku,) shortage cost at the chosen r
    w: np.ndarray            # (n_sku,) constraint weights, all 1.0 in core


@dataclass
class BundleState:
    inv: np.ndarray          # (n_sku,) on-hand at start of day, arrivals applied
    pipeline: np.ndarray     # (n_sku, L); [:, k] arrives on the morning of t+k+1

    def copy(self) -> "BundleState":
        return BundleState(inv=self.inv.copy(), pipeline=self.pipeline.copy())


@dataclass(frozen=True)
class Observation:
    """Information set F_t (spec section 6), handed to the policy at step 3.
    demand_hist covers all latent demand up to AND INCLUDING day t. The policy
    does not know where the evaluation window ends (spec section 7)."""
    t_idx: int
    demand_hist: np.ndarray            # (n_hist, n_sku)
    inv: np.ndarray                    # (n_sku,) end-of-day on-hand, after sales
    pipeline: np.ndarray               # (n_sku, L)
    k_g: float
    h: np.ndarray
    b: np.ndarray
    features: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class DayResult:
    t_idx: int
    demand: np.ndarray
    sales: np.ndarray
    unmet: np.ndarray                  # u_t; CSL(SKU-day) = 1[u_t == 0] downstream
    inv_start: np.ndarray              # on-hand at day start (arrivals applied)
    inv_eod: np.ndarray                # after sales
    order_raw: np.ndarray              # q as submitted by the policy
    order: np.ndarray                  # q as executed
    projection_applied: bool           # True only under ViolationHandling.PROJECT
    violation_amount: float            # max(0, sum(w*q_raw) - K_g)
    holding_cost: float
    shortage_cost: float
    binding: bool                      # sum(w*order) >= K_g - binding_epsilon


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    violation_amount: float


class Policy(Protocol):
    name: str

    def decide(self, obs: Observation) -> np.ndarray:
        """Return q_t >= 0 of shape (n_sku,): the first day of a fresh rolling
        plan. Core policies must submit feasible orders (sum(w*q) <= K_g up to
        tolerance); radial-scaling policies apply project_orders internally."""
        ...


# --------------------------------------------------------------------------- #
# Constraint handling (review decision: validation and projection are separate)
# --------------------------------------------------------------------------- #

def weighted_sum(q: np.ndarray, w: np.ndarray) -> float:
    """Correctly rounded weighted sum (math.fsum): the single summation
    semantics used for every constraint comparison in the engine, so that
    feasibility and binding judgements are machine- and order-independent.
    n_sku <= 16 per bundle, so the Python-level loop cost is negligible."""
    return math.fsum(float(wi) * float(qi) for wi, qi in zip(w, q))


def check_orders(q: np.ndarray, w: np.ndarray, k_g: float) -> float:
    """Return the constraint violation amount max(0, sum(w*q) - K_g)."""
    return max(0.0, weighted_sum(q, w) - k_g)


def project_orders(q: np.ndarray, w: np.ndarray, k_g: float) -> np.ndarray:
    """Radial proportional scaling onto the shared capacity constraint:
    q+ = max(q, 0); if sum(w*q+) > K_g, scale so sum(w*q+) == K_g exactly.
    This is a feasibility operator (not a Euclidean projection and not a policy
    allocation rule). Library function: policies that use it do so explicitly."""
    q_pos = np.maximum(q, 0.0)
    total = weighted_sum(q_pos, w)
    if total <= k_g or total == 0.0:
        return q_pos
    scaled = q_pos * (k_g / total)
    # Deterministic residual correction into the largest weighted component.
    # Guaranteed invariant (fuzzed, 200k cases): sum(w*q) <= K_g strictly, and
    # K_g - sum(w*q) <= 2 ulp(K_g) (exact K_g where representable). The at-most
    # 1-ulp shortfall is 10 orders of magnitude below binding_epsilon, so a
    # projected order always registers as binding.
    j = int(np.argmax(scaled * w))
    for _ in range(16):
        s = weighted_sum(scaled, w)
        if s == k_g:
            return scaled
        scaled[j] = max(0.0, scaled[j] + (k_g - s) / w[j])
    while weighted_sum(scaled, w) > k_g:
        scaled[j] = np.nextafter(scaled[j], 0.0)
    return scaled


def gate_check(q: np.ndarray, w: np.ndarray, k_g: float, eps: float) -> GateResult:
    """Recommendation gate: capacity-violating orders are rejected, never
    approved or repaired (preregistration section 5.3, spec section 4)."""
    violation = check_orders(q, w, k_g)
    return GateResult(accepted=violation <= eps, violation_amount=violation)


# --------------------------------------------------------------------------- #
# Core transition primitives (shared with the learning LP check, T6)
# --------------------------------------------------------------------------- #

def realize_day(
    state: BundleState, d_t: np.ndarray, params: SkuParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Steps 1-2: returns (sales, unmet, inv_eod, holding_cost, shortage_cost)."""
    sales = np.minimum(d_t, state.inv)
    unmet = d_t - sales
    inv_eod = state.inv - sales
    holding_cost = float(params.h @ inv_eod)
    shortage_cost = float(params.b @ unmet)
    return sales, unmet, inv_eod, holding_cost, shortage_cost


def midnight_transition(
    inv_eod: np.ndarray, pipeline: np.ndarray, q_t: np.ndarray
) -> BundleState:
    """Step 4: I_{t+1} = inv_eod + P_t[0]; the pipeline shifts left and q_t
    enters the last slot. FIFO slots make overtaking impossible by construction."""
    inv_next = inv_eod + pipeline[:, 0]
    pipeline_next = np.empty_like(pipeline)
    if pipeline.shape[1] > 1:
        pipeline_next[:, :-1] = pipeline[:, 1:]
    pipeline_next[:, -1] = q_t
    return BundleState(inv=inv_next, pipeline=pipeline_next)


def init_warmup_state(
    recent_mean_7d: np.ndarray, lead_time: int, cfg: SimConfig
) -> BundleState:
    """Warm-up start state (spec section 7): I = 7-day mean demand x 2, each
    pipeline slot = 7-day mean x 1, unmet 0. recent_mean_7d is the per-SKU mean
    over the warmup_init_lookback_days days preceding warmup_start_idx."""
    inv = recent_mean_7d * cfg.warmup_init_inventory_multiplier
    pipeline = np.tile(
        (recent_mean_7d * cfg.warmup_init_pipeline_slot_multiplier)[:, None],
        (1, lead_time),
    )
    return BundleState(inv=inv.astype(np.float64), pipeline=pipeline.astype(np.float64))


def terminal_cost(final_state: BundleState, h: np.ndarray) -> float:
    """Core terminal treatment (spec section 7): after the final midnight
    transition, charge h_i x 1 day once on on-hand and all in-transit units,
    salvage 0. Equals h . (I_T_eod + P_T[0] + P_T[1] + q_T) in pre-transition
    notation, so the last-day order is charged (no free-final-order artifact).
    The regular day-T holding cost h . I_T_eod is a separate, earlier charge."""
    return float(h @ (final_state.inv + final_state.pipeline.sum(axis=1)))


# --------------------------------------------------------------------------- #
# Window runner
# --------------------------------------------------------------------------- #

def run_window(
    state: BundleState,
    demand: np.ndarray,
    demand_hist: np.ndarray,
    policy: Policy,
    params: SkuParams,
    k_g: float,
    cfg: SimConfig,
    *,
    t0_idx: int,
    features: dict[str, np.ndarray] | None = None,
    on_violation: ViolationHandling = ViolationHandling.ERROR,
) -> tuple[BundleState, list[DayResult]]:
    """Run steps 1-4 for every day in `demand`, including the final midnight
    transition. Used twice per episode: warm-up (policy = A4, costs excluded
    from aggregation downstream) and test window (costs kept).

    demand:      (n_days, n_sku) latent demand path for this window (fixed,
                 pre-built; identical realization for all compared policies)
    demand_hist: (n_hist, n_sku) latent demand before the window
    features:    optional dict of arrays aligned to concat(demand_hist, demand)
                 along axis 0; Observation receives views up to and incl. day t
    on_violation: ERROR (default) validates and raises EngineError beyond
                 feasibility_tol_rel * K_g; PROJECT applies project_orders and
                 records it. The engine never repairs silently.
    """
    n_days, n_sku = demand.shape
    if state.inv.shape != (n_sku,):
        raise ValueError("state/demand sku dimension mismatch")
    hist = np.concatenate([demand_hist, demand], axis=0) if demand_hist.size else demand
    n_hist0 = hist.shape[0] - n_days
    tol = cfg.feasibility_tol_rel * k_g
    results: list[DayResult] = []
    cur = state.copy()

    for day in range(n_days):
        t_idx = t0_idx + day
        d_t = demand[day]
        inv_start = cur.inv.copy()
        sales, unmet, inv_eod, holding_cost, shortage_cost = realize_day(cur, d_t, params)

        obs_features = None
        if features is not None:
            obs_features = {k: v[: n_hist0 + day + 1] for k, v in features.items()}
        obs = Observation(
            t_idx=t_idx,
            demand_hist=hist[: n_hist0 + day + 1],
            inv=inv_eod,
            pipeline=cur.pipeline,
            k_g=k_g,
            h=params.h,
            b=params.b,
            features=obs_features,
        )
        q_raw = np.asarray(policy.decide(obs), dtype=np.float64)
        if q_raw.shape != (n_sku,):
            raise EngineError(
                f"policy {policy.name} returned shape {q_raw.shape}, expected ({n_sku},)"
            )
        if np.any(q_raw < 0.0):
            raise EngineError(f"policy {policy.name} returned negative orders at t={t_idx}")

        violation = check_orders(q_raw, params.w, k_g)
        projection_applied = False
        if violation > tol:
            if on_violation is ViolationHandling.ERROR:
                raise EngineError(
                    f"policy {policy.name} violated capacity at t={t_idx}: "
                    f"sum(w*q)={float(params.w @ q_raw):.6f} > K_g={k_g:.6f} "
                    f"(violation={violation:.3e}, tol={tol:.3e})"
                )
            q_final = project_orders(q_raw, params.w, k_g)
            projection_applied = True
        else:
            q_final = q_raw

        order_sum = weighted_sum(q_final, params.w)
        binding = order_sum >= k_g - cfg.binding_epsilon

        results.append(
            DayResult(
                t_idx=t_idx,
                demand=d_t.copy(),
                sales=sales,
                unmet=unmet,
                inv_start=inv_start,
                inv_eod=inv_eod.copy(),
                order_raw=q_raw.copy(),
                order=q_final.copy(),
                projection_applied=projection_applied,
                violation_amount=violation,
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
                binding=binding,
            )
        )
        cur = midnight_transition(inv_eod, cur.pipeline, q_final)

    return cur, results
