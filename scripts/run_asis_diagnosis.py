"""2-H: AS-IS diagnosis run (preregistration section 5.5, spec section 10).

Runs the frozen policies over all evaluation folds, BEFORE any Stage 3+ model
exists, to report the realism of delta = 1%p and the CSL 90% floor. Criteria
are NOT changed by this diagnosis.

Policies (3): tuned Track A A4 (per r from outputs/stage2_a4_tuning.json),
A4-SL (Track B amendment: min validation cost among candidates with pooled
SKU-day validation CSL >= 0.90; "no qualifying policy" recorded when the set
is empty), and L0 rule CT (fictional client's fixed (s,S) playbook, untuned).

Scope: all folds x 10 bundles, core scenario (p = 0, pre-registered fold-level
demand shock), r in {3, 5, 10} with r = 10 reusing the r = 5 policies. Common
warm-up per r environment: r=3 -> A4@r3, r=5 -> A4@r5, r=10 -> A4@r5; within
one r every policy inherits the identical warm-up state and demand path.

Metric naming contract (decision log 2-H): realized_csl_below_90 counts
episodes whose SIMULATED CSL fell below 0.90 -- it is NOT a gate-block count;
gate_predicted_block_count exists only once the Stage 6 gate runs. The
capacity metrics are capacity_binding_days (orders reached K_g) and, for the
A4 family only, desired_capacity_exceed_days (pre-allocation raw demand above
K_g).

Early-view contract: these fold results serve ONLY the pre-registered AS-IS
diagnosis and acceptance-criteria realism report. They are not used to tune
Stage 3-6 models or the agent; agent prompts, action ranges, candidate rules
and the gate are frozen and committed before the final H3 evaluation; the
portfolio report discloses that this pre-defined AS-IS analysis was run.

Output: outputs/stage2_asis_diagnosis.json (idempotent overwrite).
Usage: python -m scripts.run_asis_diagnosis
"""

import datetime
import json

import numpy as np

from core_pipeline.data.db import PROJECT_ROOT, config_sha256, connect, load_config
from core_pipeline.data.mart_reader import (
    cost_arrays,
    fetch_bundle_capacity,
    fetch_demand_by_bundle,
    fetch_folds,
    fetch_sku_costs,
)
from core_pipeline.policies.a4_basestock import (
    ALLOCATION_PROPORTIONAL,
    A4BaseStock,
    A4Config,
)
from core_pipeline.policies.rule_ct import RuleControlTower
from core_pipeline.simulator.demand_paths import build_fold_paths
from core_pipeline.simulator.engine import SkuParams, load_sim_config
from core_pipeline.simulator.episode import (
    compute_common_warmup,
    run_policy_window,
)

TUNING_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "stage2_asis_diagnosis.json"

R_EVAL = (3, 5, 10)
POLICY_R = {3: 3, 5: 5, 10: 5}     # r=10 reuses the r=5 policies
A4SL_CSL_FLOOR = 0.90
ALLOCATION_ORDER = ("proportional", "b_priority")  # tie-break order (= tuning)


def select_a4_sl(candidates: list[dict], r: int) -> dict | None:
    """Track B amendment rule: among candidates at this r with pooled SKU-day
    validation CSL >= 0.90, the minimum validation cost; ties broken by
    proportional-first then smaller tau. None when no candidate qualifies."""
    pool = [c for c in candidates
            if c["r"] == r and c["csl_sku_day"] >= A4SL_CSL_FLOOR]
    if not pool:
        return None
    return min(pool, key=lambda c: (
        c["validation_cost"], ALLOCATION_ORDER.index(c["allocation"]), c["tau"],
    ))


def make_a4(entry: dict) -> A4BaseStock:
    return A4BaseStock(A4Config(tau=entry["tau"],
                                allocation=entry.get("allocation_rule",
                                                     entry.get("allocation"))))


def summarize(records: list, window_stats: list) -> dict:
    n_ep = len(records)
    sku_days = sum(rec.n_days * rec.n_sku for rec in records)
    satisfied = sum(rec.csl * rec.n_days * rec.n_sku for rec in records)
    csl_vals = np.array([rec.csl for rec in records])
    below = int(np.sum(csl_vals < A4SL_CSL_FLOOR))
    total_days = sum(w["n_days"] for w in window_stats)
    binding = sum(w["binding_days"] for w in window_stats)
    raw_exceed = [w["raw_exceed_days"] for w in window_stats
                  if w["raw_exceed_days"] is not None]
    return {
        "episodes": n_ep,
        "total_cost": float(sum(rec.cost for rec in records)),
        "pooled_csl_sku_day": satisfied / sku_days,
        "episode_csl_mean": float(csl_vals.mean()),
        "episode_csl_quantiles": {
            q: float(np.quantile(csl_vals, float(q))) for q in
            ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "realized_csl_below_90": below,
        "realized_csl_below_90_share": below / n_ep,
        "fill_rate_mean": float(np.mean([rec.fill_rate for rec in records])),
        "capacity_binding_days": int(binding),
        "capacity_binding_share": binding / total_days,
        "desired_capacity_exceed_days": (int(sum(raw_exceed)) if raw_exceed else None),
        "desired_capacity_exceed_share": (sum(raw_exceed) / total_days
                                          if raw_exceed else None),
    }


def bundle_characteristics(records: list, demand, folds, group_ids) -> dict:
    """Per-bundle demand scale / intermittency over the evaluation windows plus
    this run's per-bundle episode CSL and binding (low-CSL bundle traits)."""
    out = {}
    for gid in group_ids:
        _, mat = demand[gid]
        vals = []
        zeros = active = 0
        for f in folds:
            win = mat[f["test_start_idx"] - 1: f["test_end_idx"]]
            vals.append(win)
            for i in range(mat.shape[1]):
                pos = np.nonzero(mat[:, i] > 0)[0]
                if pos.size == 0:
                    continue
                days = np.arange(max(f["test_start_idx"] - 1, pos[0]),
                                 f["test_end_idx"])
                active += days.size
                zeros += int(np.sum(mat[days, i] == 0.0))
        allw = np.concatenate(vals, axis=0)
        recs = [r for r in records if r.group_id == gid]
        out[gid] = {
            "mean_daily_demand": float(allw.sum(axis=1).mean()),
            "zero_day_share_active": zeros / active if active else None,
            "episode_csl_mean": float(np.mean([r.csl for r in recs])),
            "realized_csl_below_90_share": float(np.mean(
                [r.csl < A4SL_CSL_FLOOR for r in recs])),
            "binding_day_share": float(sum(r.binding_days for r in recs)
                                       / sum(r.n_days for r in recs)),
        }
    return out


def main():
    cfg = load_config()
    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise RuntimeError("kappa_selected missing in configs/simulator.yaml")
    tuning = json.loads(TUNING_PATH.read_text())

    conn = connect(cfg)
    try:
        folds = fetch_folds(conn)
        kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
        max_idx = max(f["test_end_idx"] for f in folds)
        demand = fetch_demand_by_bundle(conn, max_idx)
        costs = fetch_sku_costs(conn)
    finally:
        conn.close()
    group_ids = sorted(demand)

    a4_by_r = {int(r): make_a4(sel) for r, sel in tuning["selected"].items()}
    a4sl_by_r = {r: select_a4_sl(tuning["candidates"], r) for r in (3, 5)}

    # demand paths: one build per fold (core scenario p = 0, shock included)
    paths = {}
    for f in folds:
        paths[f["fold"]] = build_fold_paths(
            demand, fold=f["fold"],
            warmup_start_row=f["warmup_start_idx"] - 1,
            test_start_row=f["test_start_idx"] - 1,
            test_end_row=f["test_end_idx"] - 1,
            p=0.0, cfg=sim_cfg,
        )

    all_records = {}
    window_stats = {}
    warm_cache = {}
    for r in R_EVAL:
        pr = POLICY_R[r]
        pols = {"a4": make_a4(tuning["selected"][str(pr)])}
        sl = a4sl_by_r[pr]
        if sl is not None:
            pols["a4_sl"] = A4BaseStock(A4Config(tau=sl["tau"],
                                                 allocation=sl["allocation"]))
        pols["l0_rule_ct"] = RuleControlTower()
        for name, p in pols.items():
            p.name = name

        for f in folds:
            fold = f["fold"]
            for gid in group_ids:
                bp = paths[fold][gid]
                series, _ = demand[gid]
                h, b = cost_arrays(series, costs, r)
                params = SkuParams(series_ids=series, h=h, b=b,
                                   w=np.full(len(series), sim_cfg.weight_wi))
                wkey = (fold, gid, pr)
                if wkey not in warm_cache:
                    warm_cache[wkey] = compute_common_warmup(
                        bp.warmup, bp.history, a4_by_r[pr], params, kg[gid],
                        sim_cfg, warmup_start_idx=f["warmup_start_idx"],
                    )
                state = warm_cache[wkey]
                hist = np.concatenate([bp.history, bp.warmup], axis=0)
                for name, pol in pols.items():
                    res = run_policy_window(
                        state, bp.test, hist, pol, params, kg[gid], sim_cfg,
                        t0_idx=f["test_start_idx"], fold=fold, group_id=gid,
                    )
                    key = (name, r)
                    all_records.setdefault(key, []).extend(res.records)
                    window_stats.setdefault(key, []).append({
                        "n_days": res.n_days,
                        "binding_days": res.binding_days,
                        "raw_exceed_days": res.raw_exceed_days,
                    })
        print(f"r={r}: policies {sorted(pols)} done "
              f"({len(folds)} folds x {len(group_ids)} bundles)")

    summary = {}
    per_bundle = {}
    episodes_out = []
    for (name, r), records in sorted(all_records.items()):
        summary[f"{name}@r{r}"] = summarize(records, window_stats[(name, r)])
        per_bundle[f"{name}@r{r}"] = bundle_characteristics(
            records, demand, folds, group_ids)
        episodes_out.extend({
            "policy": name, "r": r, "fold": rec.fold, "group_id": rec.group_id,
            "week": rec.week, "cost": rec.cost, "terminal_cost": rec.terminal_cost,
            "csl": rec.csl, "fill_rate": rec.fill_rate,
            "guardrail_blocked": rec.guardrail_blocked,
            "binding_days": rec.binding_days, "n_days": rec.n_days,
            "n_sku": rec.n_sku,
        } for rec in records)

    for key, s in summary.items():
        print(f"{key}: cost {s['total_cost']:.0f} pooled_csl "
              f"{s['pooled_csl_sku_day']:.4f} below90 "
              f"{s['realized_csl_below_90_share']:.3f} binding "
              f"{s['capacity_binding_share']:.3f}")

    payload = {
        "task": "2-H AS-IS diagnosis",
        "date": datetime.date.today().isoformat(),
        "contract": {
            "l0_nature": "fictional client's fixed rule playbook (MA7, s=3d, S=7d, untuned); not a measured real policy",
            "initial_state": "A4 common warm-up state, not measured client inventory; no claim of reproducing real operations",
            "metric_naming": "realized_csl_below_90 is a simulated-outcome count, distinct from gate_predicted_block_count (Stage 6 only)",
            "early_view": "fold results used only for this pre-registered diagnosis and criteria-realism report; not for Stage 3-6 tuning; agent config frozen+committed before H3; disclosed in the portfolio report",
            "criteria": "delta=1%p and CSL 90% floor are reported on, never changed here",
        },
        "kappa": sim_cfg.kappa_selected,
        "scenario": {"p": 0.0, "shock": "pre-registered fold-level path"},
        "warmup_policy_by_r": {"3": "a4@r3", "5": "a4@r5", "10": "a4@r5"},
        "a4_selected": tuning["selected"],
        "a4_sl_selected": {
            str(r): (sl if sl is None else {
                "tau": sl["tau"], "allocation": sl["allocation"],
                "validation_cost": sl["validation_cost"],
                "validation_pooled_csl": sl["csl_sku_day"],
            }) for r, sl in a4sl_by_r.items()
        },
        "summary": summary,
        "per_bundle": per_bundle,
        "cost_csl_frontier_validation": tuning["candidates"],
        "episodes": episodes_out,
        "data_config_sha256": config_sha256(),
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"written: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(episodes_out)} episode rows)")


if __name__ == "__main__":
    main()
