"""2-E: kappa selection by the pre-registered rule (preregistration v1.2 section 5.5).

Runs the FIXED reference A4 (tau = 5/6, proportional allocation, no tuning) on
the validation window (mart.windows 'validation', d_1438..d_1605) for each
kappa candidate in {1.0, 1.2, 1.5}, measures the binding share (bundle-days
with sum(w*q) >= K_g - binding_epsilon over all bundle-days in the window),
and applies the frozen selection rule:
  1) candidates whose share lies in [0.20, 0.50]; if several, the one closest
     to 1.2; 2) if none, the candidate whose share is closest to the interval;
  ties broken by distance to 1.2, then by the smaller kappa (deterministic).

Run details fixed before results (decision log, Stage 2):
- 14-day A4 warm-in (d_1424..d_1437) precedes the window so the measured days
  do not depend on the arbitrary initial state; warm-in days are excluded from
  the share. Initial state at d_1424 uses the spec section 7 rule (I = 7-day
  mean x 2 over d_1417..d_1423, each pipeline slot x 1).
- Demand = observed M5 sales (core p = 0), all of it inside the selection
  window (<= d_1605): no evaluation-window information is used.
- The run is fully deterministic (no randomness in reference A4 or demand).

Output: outputs/stage2_kappa_selection.json (idempotent overwrite).
Usage: python -m scripts.select_kappa
"""

import datetime
import json

import numpy as np

from core_pipeline.data.db import PROJECT_ROOT, config_sha256, connect, load_config
from core_pipeline.policies.a4_basestock import make_reference_a4
from core_pipeline.simulator.engine import (
    SkuParams,
    init_warmup_state,
    load_sim_config,
    run_window,
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "stage2_kappa_selection.json"
SHARE_LO, SHARE_HI, KAPPA_TARGET = 0.20, 0.50, 1.2


def kappa_column(kappa: float) -> str:
    return "k_g_kappa_" + f"{kappa:.1f}".replace(".", "_")


def fetch_inputs(conn, max_idx: int):
    cur = conn.cursor()
    cur.execute(
        "SELECT start_idx, end_idx FROM mart.windows WHERE window_name = 'validation'"
    )
    val_start, val_end = cur.fetchone()
    cur.execute("SELECT group_id FROM mart.bundles ORDER BY group_id")
    group_ids = [r[0] for r in cur.fetchall()]
    kg = {}
    for kappa in (1.0, 1.2, 1.5):
        cur.execute(
            f"SELECT group_id, {kappa_column(kappa)} FROM mart.bundles ORDER BY group_id"
        )
        kg[kappa] = dict(cur.fetchall())
    demand = {}
    for gid in group_ids:
        cur.execute(
            """
            SELECT series_id, period_idx, sales FROM mart.sku_daily
            WHERE group_id = %s AND period_idx <= %s
            ORDER BY series_id, period_idx
            """,
            (gid, max_idx),
        )
        rows = cur.fetchall()
        series = sorted({r[0] for r in rows})
        idx = {s: j for j, s in enumerate(series)}
        mat = np.zeros((max_idx, len(series)))
        for sid, p, sales in rows:
            mat[p - 1, idx[sid]] = sales
        demand[gid] = (tuple(series), mat)
    return val_start, val_end, group_ids, kg, demand


def binding_share_for_kappa(sim_cfg, group_ids, kg_map, demand, val_start, val_end):
    warm_days = sim_cfg.warmup_days
    lookback = sim_cfg.warmup_init_lookback_days
    warm_start = val_start - warm_days
    per_bundle = {}
    binding_total = days_total = 0
    for gid in group_ids:
        series, mat = demand[gid]
        n_sku = len(series)
        hist = mat[: warm_start - 1]                   # d_1 .. d_{warm_start-1}
        window = mat[warm_start - 1: val_end]          # d_warm_start .. d_val_end
        recent_mean = mat[warm_start - 1 - lookback: warm_start - 1].mean(axis=0)
        state = init_warmup_state(recent_mean, sim_cfg.lead_time_core, sim_cfg)
        params = SkuParams(
            series_ids=series,
            h=np.ones(n_sku),                           # unused by reference A4
            b=np.ones(n_sku),                           # proportional rule ignores b
            w=np.full(n_sku, sim_cfg.weight_wi),
        )
        _, days = run_window(
            state, window, hist, make_reference_a4(), params,
            k_g=float(kg_map[gid]), cfg=sim_cfg, t0_idx=warm_start,
        )
        measured = [r for r in days if val_start <= r.t_idx <= val_end]
        n_binding = sum(1 for r in measured if r.binding)
        per_bundle[gid] = {
            "binding_days": n_binding,
            "window_days": len(measured),
            "share": n_binding / len(measured),
            "k_g": float(kg_map[gid]),
        }
        binding_total += n_binding
        days_total += len(measured)
    return binding_total / days_total, per_bundle, binding_total, days_total


def interval_distance(share: float) -> float:
    if share < SHARE_LO:
        return SHARE_LO - share
    if share > SHARE_HI:
        return share - SHARE_HI
    return 0.0


def select_kappa(shares: dict[float, float]) -> tuple[float, str]:
    in_range = [k for k, s in shares.items() if SHARE_LO <= s <= SHARE_HI]
    if in_range:
        selected = min(in_range, key=lambda k: (abs(k - KAPPA_TARGET), k))
        if len(in_range) == 1:
            return selected, "only candidate with share in [0.20, 0.50]"
        return selected, "multiple candidates in range; closest to 1.2 (tie: smaller)"
    selected = min(
        shares, key=lambda k: (interval_distance(shares[k]), abs(k - KAPPA_TARGET), k)
    )
    return selected, (
        "no candidate in range; share closest to the interval "
        "(tie: closest to 1.2, then smaller)"
    )


def main():
    cfg = load_config()
    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    conn = connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT end_idx FROM mart.windows WHERE window_name = 'validation'"
        )
        max_idx = cur.fetchone()[0]
        val_start, val_end, group_ids, kg, demand = fetch_inputs(conn, max_idx)
    finally:
        conn.close()

    ref = make_reference_a4()
    results = {}
    shares = {}
    for kappa in sim_cfg.kappa_candidates:
        share, per_bundle, n_bind, n_days = binding_share_for_kappa(
            sim_cfg, group_ids, kg[kappa], demand, val_start, val_end
        )
        shares[kappa] = share
        results[f"{kappa:.1f}"] = {
            "overall_share": share,
            "binding_days": n_bind,
            "total_bundle_days": n_days,
            "per_bundle": per_bundle,
        }
        print(f"kappa {kappa:.1f}: binding share {share:.4f} ({n_bind}/{n_days})")

    selected, reason = select_kappa(shares)
    print(f"selected kappa: {selected:.1f} ({reason})")

    payload = {
        "task": "2-E kappa selection",
        "date": datetime.date.today().isoformat(),
        "rule": "preregistration v1.2 section 5.5: share in [0.20, 0.50] -> closest to 1.2; none -> closest to interval",
        "reference_policy": {
            "name": ref.name,
            "tau": ref.config.tau,
            "allocation": ref.config.allocation,
            "protection_days": ref.config.protection_days,
            "quantile_method": ref.config.quantile_method,
            "tuned": False,
        },
        "validation_window": {
            "start_idx": val_start, "end_idx": val_end,
            "days": val_end - val_start + 1,
        },
        "warm_in": {
            "days": sim_cfg.warmup_days,
            "start_idx": val_start - sim_cfg.warmup_days,
            "init_lookback_days": sim_cfg.warmup_init_lookback_days,
            "excluded_from_share": True,
        },
        "binding_epsilon": sim_cfg.binding_epsilon,
        "demand": "observed M5 sales, core p = 0, all <= selection end",
        "results": results,
        "selected_kappa": selected,
        "selection_reason": reason,
        "data_config_sha256": config_sha256(),
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"written: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
