"""2-D-2: final A4 tuning at the frozen kappa (preregistration section 4).

Selects (tau, allocation_rule) once per cost ratio r in {3, 5} by total cost
on the validation window (mart.windows 'validation', d_1438..d_1605), at the
kappa frozen in 2-E (outputs/stage2_kappa_selection.json). Pre-fixed before
any results (decision log, Stage 2):

- Grid: tau in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 5/6, 0.85, 0.90,
  0.95, 0.99} (the preregistration fixes only the range [0.5, 0.99]; the grid
  includes both newsvendor critical fractiles r/(1+r): 0.75 for r=3, 5/6 for
  r=5) x allocation in {proportional, b_priority}.
- Objective: sum over validation-window bundle-days of h_i*I_t + b_i*u_t at
  the given r (warm-in days excluded, no terminal term -- one continuous
  168-day diagnostic run per bundle, not a 28-day episode).
- Each candidate warms itself in for 14 days (d_1424..d_1437) from the spec
  section 7 initial state; identical demand (observed M5 sales, p = 0) for
  every candidate; fully deterministic.
- Tie-break on equal cost: proportional before b_priority, then smaller tau.
- Step D: the binding share of the tuned A4 is reported but kappa is NEVER
  re-selected here.
- r = 10 evaluation reuses the r = 5 policy (no extra tuning).

Fast path: target levels S_{i,t} depend only on (tau, history), so they are
computed once for all taus with target_level_grid and evaluated through
A4FromTargets, whose equality with A4BaseStock is pinned by tests/test_a4.py.

Output: outputs/stage2_a4_tuning.json (idempotent overwrite).
Usage: python -m scripts.tune_a4
"""

import datetime
import json

import numpy as np

from core_pipeline.data.db import PROJECT_ROOT, config_sha256, connect, load_config
from core_pipeline.data.mart_reader import (
    cost_arrays,
    fetch_bundle_capacity,
    fetch_demand_by_bundle,
    fetch_sku_costs,
    fetch_window,
)
from core_pipeline.policies.a4_basestock import (
    ALLOCATION_B_PRIORITY,
    ALLOCATION_PROPORTIONAL,
    A4FromTargets,
    target_level_grid,
)
from core_pipeline.simulator.engine import (
    SkuParams,
    init_warmup_state,
    load_sim_config,
    run_window,
)

KAPPA_PATH = PROJECT_ROOT / "outputs" / "stage2_kappa_selection.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"

TAU_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 5.0 / 6.0, 0.85, 0.90, 0.95, 0.99)
ALLOCATION_ORDER = (ALLOCATION_PROPORTIONAL, ALLOCATION_B_PRIORITY)  # tie-break order
R_TUNE = (3, 5)


def precompute_targets(mat: np.ndarray, warm_start: int, val_end: int) -> np.ndarray:
    """targets[j, i, k] = S for day t = warm_start + j, SKU i, tau TAU_GRID[k],
    from the expanding history d_1..d_t (same semantics as A4BaseStock)."""
    n_days = val_end - warm_start + 1
    n_sku = mat.shape[1]
    taus = np.asarray(TAU_GRID)
    targets = np.empty((n_days, n_sku, len(taus)))
    for j in range(n_days):
        hist_len = warm_start + j            # rows 0..hist_len-1 = d_1..d_t
        for i in range(n_sku):
            targets[j, i, :] = target_level_grid(mat[:hist_len, i], taus)
    return targets


def main():
    cfg = load_config()
    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    kappa_sel = json.loads(KAPPA_PATH.read_text())
    kappa = float(kappa_sel["selected_kappa"])

    conn = connect(cfg)
    try:
        val_start, val_end = fetch_window(conn, "validation")
        kg = fetch_bundle_capacity(conn, kappa)
        demand = fetch_demand_by_bundle(conn, val_end)
        costs = fetch_sku_costs(conn)
    finally:
        conn.close()

    warm_days = sim_cfg.warmup_days
    lookback = sim_cfg.warmup_init_lookback_days
    warm_start = val_start - warm_days
    group_ids = sorted(demand)

    targets = {}
    for gid in group_ids:
        _, mat = demand[gid]
        targets[gid] = precompute_targets(mat, warm_start, val_end)
    print(f"targets precomputed: {len(group_ids)} bundles x "
          f"{val_end - warm_start + 1} days x {len(TAU_GRID)} taus")

    candidates = []
    for r in R_TUNE:
        for allocation in ALLOCATION_ORDER:
            for k, tau in enumerate(TAU_GRID):
                total_cost = 0.0
                binding_days = sku_days = unmet_zero_days = 0
                sales_sum = demand_sum = 0.0
                for gid in group_ids:
                    series, mat = demand[gid]
                    n_sku = len(series)
                    h, b = cost_arrays(series, costs, r)
                    params = SkuParams(series_ids=series, h=h, b=b,
                                       w=np.full(n_sku, sim_cfg.weight_wi))
                    hist = mat[: warm_start - 1]
                    window = mat[warm_start - 1: val_end]
                    recent_mean = mat[warm_start - 1 - lookback: warm_start - 1].mean(axis=0)
                    state = init_warmup_state(recent_mean, sim_cfg.lead_time_core, sim_cfg)
                    policy = A4FromTargets(targets[gid][:, :, k], warm_start, allocation)
                    _, days = run_window(
                        state, window, hist, policy, params,
                        k_g=kg[gid], cfg=sim_cfg, t0_idx=warm_start,
                    )
                    for res in days:
                        if val_start <= res.t_idx <= val_end:
                            total_cost += res.holding_cost + res.shortage_cost
                            binding_days += int(res.binding)
                            sku_days += res.unmet.size
                            unmet_zero_days += int(np.sum(res.unmet == 0.0))
                            sales_sum += float(res.sales.sum())
                            demand_sum += float(res.demand.sum())
                window_bundle_days = len(group_ids) * (val_end - val_start + 1)
                candidates.append({
                    "r": r,
                    "tau": tau,
                    "allocation": allocation,
                    "validation_cost": total_cost,
                    "binding_share": binding_days / window_bundle_days,
                    "csl_sku_day": unmet_zero_days / sku_days,
                    "fill_rate": sales_sum / demand_sum,
                })

    selected = {}
    for r in R_TUNE:
        pool = [c for c in candidates if c["r"] == r]
        best = min(pool, key=lambda c: (
            c["validation_cost"],
            ALLOCATION_ORDER.index(c["allocation"]),
            c["tau"],
        ))
        selected[str(r)] = {
            "r": r,
            "kappa": kappa,
            "tau": best["tau"],
            "allocation_rule": best["allocation"],
            "validation_cost": best["validation_cost"],
            "binding_share": best["binding_share"],
            "csl_sku_day": best["csl_sku_day"],
            "fill_rate": best["fill_rate"],
            "validation_window": {"start_idx": val_start, "end_idx": val_end},
        }
        print(f"r={r}: tau={best['tau']:.4f} allocation={best['allocation']} "
              f"cost={best['validation_cost']:.2f} csl={best['csl_sku_day']:.4f} "
              f"binding={best['binding_share']:.4f}")

    payload = {
        "task": "2-D-2 final A4 tuning",
        "date": datetime.date.today().isoformat(),
        "kappa": kappa,
        "kappa_source": "outputs/stage2_kappa_selection.json (2-E, frozen; Step D: no re-selection)",
        "tau_grid": list(TAU_GRID),
        "allocation_rules": list(ALLOCATION_ORDER),
        "objective": "total h*I + b*u over validation bundle-days at the given r; warm-in excluded; no terminal term",
        "tie_break": "lower cost, then proportional before b_priority, then smaller tau",
        "warm_in": {"days": warm_days, "start_idx": warm_start,
                    "init_lookback_days": lookback, "policy": "candidate itself"},
        "r_eval_10": "reuses the r = 5 selection (no extra tuning)",
        "selected": selected,
        "candidates": candidates,
        "data_config_sha256": config_sha256(),
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"written: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
