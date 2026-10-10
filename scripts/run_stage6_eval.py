"""Stage 6 rolling evaluation runner (ST6-06 staged execution, option A).

Step 1  --policies a4,a4_sl,l0_rule_ct : regression against Stage 2-H --
        per-episode comparison on (fold, group_id, week, policy, r) for
        cost / terminal_cost / csl / fill_rate / binding_days (rel 1e-9);
        any mismatch is investigated regardless of performance.
Step 2+ --policies ...,b : new policies surface EXECUTION DIAGNOSTICS ONLY
        on the console (records, binding days, LP hygiene counters); cost
        and CSL summaries stay unsurfaced until the full pre-registered set
        has run (ST6-06 (3)). Episode records are always written to the
        output file -- they are the inputs of the final confirmatory
        analysis, which happens once, after the full set (ST6-06 (4)).

Environment-cost contract at r = 10 (ST6-01 (4)): run_policy_window builds
SkuParams at the ENVIRONMENT r, LPForecastPolicy prices its LP from
Observation.h/b, so B at r=10 = b_r5 predictions + b = 10h LP by wiring.
Replay lineage (ST6-03 (5)): this runner recomputes the scenario demand-path
sha with the same rule as build_eval_predictions and hands it to the
replayer, which hard-fails on any mismatch.

Usage:
  python -m scripts.run_stage6_eval --policies a4,a4_sl,l0_rule_ct --tag step1
  python -m scripts.run_stage6_eval --policies a4,a4_sl,l0_rule_ct,b --tag step2
"""

import argparse
import datetime
import hashlib
import json

import numpy as np

from core_pipeline.data.db import (
    PROJECT_ROOT, config_sha256, connect, load_config,
)
from core_pipeline.data.mart_reader import (
    cost_arrays,
    fetch_bundle_capacity,
    fetch_demand_by_bundle,
    fetch_folds,
    fetch_sku_costs,
)
from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.policies.lp_policy import (
    EvalPredictionReplayer, LPForecastPolicy,
)
from core_pipeline.policies.rule_ct import RuleControlTower
from core_pipeline.simulator.demand_paths import build_fold_paths
from core_pipeline.simulator.engine import SkuParams, load_sim_config
from core_pipeline.simulator.episode import (
    compute_common_warmup,
    run_policy_window,
)

TUNING_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"
STAGE2_PATH = PROJECT_ROOT / "outputs" / "stage2_asis_env_ref.json"
OUT_DIR = PROJECT_ROOT / "outputs"
SCENARIO = "core"
SCENARIO_P = 0.0
R_EVAL = (3, 5, 10)
POLICY_R = {3: 3, 5: 5, 10: 5}
BASELINE_POLICIES = ("a4", "a4_sl", "l0_rule_ct")
NEW_POLICIES = ("b", "a1", "a2")
REG_REL_TOL = 1e-9
REG_FIELDS = ("cost", "terminal_cost", "csl", "fill_rate", "binding_days")


def make_a4(entry: dict) -> A4BaseStock:
    return A4BaseStock(A4Config(tau=entry["tau"],
                                allocation=entry.get("allocation_rule",
                                                     entry.get("allocation"))))

def select_a4_sl(candidates: list[dict], r: int) -> dict | None:
    qual = [c for c in candidates
            if c["r"] == r and c["csl_sku_day"] >= 0.90]
    if not qual:
        return None
    qual.sort(key=lambda c: (c["validation_cost"],
                             c["allocation"] != "proportional", c["tau"]))
    return qual[0]


def scenario_sha(paths: dict, group_ids: list[str]) -> str:
    sha = hashlib.sha256()
    for gid in group_ids:
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(
            paths[gid].test, dtype=np.float64).tobytes())
    return sha.hexdigest()


def regression_gate(episodes: list[dict], run_policies: list[str],
                    r_list: tuple[int, ...]) -> dict:
    ref = json.loads(STAGE2_PATH.read_text())["episodes"]
    keys = ("policy", "r", "fold", "group_id", "week")
    ref_map = {tuple(e[k] for k in keys): e for e in ref}
    checked = 0
    worst = 0.0
    mismatches = []
    env_drift_quanta: dict = {}
    for e in episodes:
        if e["policy"] not in BASELINE_POLICIES or e["r"] not in r_list:
            continue
        key = tuple(e[k] for k in keys)
        base = ref_map.get(key)
        if base is None:
            mismatches.append({"key": key, "field": "<missing in stage2>"})
            continue
        checked += 1
        for f in REG_FIELDS:
            a, b = float(e[f]), float(base[f])
            rel = abs(a - b) / max(abs(b), 1.0)
            if rel <= REG_REL_TOL:
                continue
            if f == "csl":
                q = (a - b) * e["n_sku"] * e["n_days"]
                if abs(q - round(q)) < 1e-6:
                    qi = int(round(q))
                    env_drift_quanta[qi] = env_drift_quanta.get(qi, 0) + 1
                    continue
            worst = max(worst, rel)
            mismatches.append(
                {"key": key, "field": f, "stage6": a, "stage2": b,
                 "rel": rel})
    expected = sum(1 for k in ref_map
                   if k[0] in run_policies and k[1] in r_list)
    return {"checked": checked, "expected": expected,
            "worst_rel": worst, "mismatches": mismatches,
            "env_drift_csl_quanta": {str(k): v for k, v
                                     in env_drift_quanta.items()}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="a4,a4_sl,l0_rule_ct")
    ap.add_argument("--folds", default="1-12")
    ap.add_argument("--tag", default="step1")
    args = ap.parse_args()
    run_policies = [p.strip() for p in args.policies.split(",")]
    unknown = [p for p in run_policies
               if p not in BASELINE_POLICIES + NEW_POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}")
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise SystemExit("kappa_selected missing in configs/simulator.yaml")
    tuning = json.loads(TUNING_PATH.read_text())

    conn = connect(load_config())
    try:
        folds_all = fetch_folds(conn)
        kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
        max_idx = max(f["test_end_idx"] for f in folds_all)
        demand = fetch_demand_by_bundle(conn, max_idx)
        costs = fetch_sku_costs(conn)
    finally:
        conn.close()
    folds = [f for f in folds_all if f["fold"] in fold_list]
    group_ids = sorted(demand)
    all_series = tuple(s for gid in group_ids for s in demand[gid][0])

    a4_by_r = {int(r): make_a4(sel) for r, sel in tuning["selected"].items()}
    a4sl_by_r = {r: select_a4_sl(tuning["candidates"], r) for r in (3, 5)}

    paths, path_sha = {}, {}
    for f in folds:
        paths[f["fold"]] = build_fold_paths(
            demand, fold=f["fold"],
            warmup_start_row=f["warmup_start_idx"] - 1,
            test_start_row=f["test_start_idx"] - 1,
            test_end_row=f["test_end_idx"] - 1,
            p=SCENARIO_P, cfg=sim_cfg,
        )
        path_sha[f["fold"]] = scenario_sha(paths[f["fold"]], group_ids)

    replayers: dict = {}

    def lp_replayer(model: str, f: dict) -> EvalPredictionReplayer:
        key = (model, f["fold"])
        if key not in replayers:
            replayers[key] = EvalPredictionReplayer(
                model=model, scenario=SCENARIO, fold=f["fold"],
                expected_origins=range(f["test_start_idx"],
                                       f["test_end_idx"] + 1),
                expected_series=all_series,
                expected_path_sha=path_sha[f["fold"]],
            )
        return replayers[key]

    episodes = []
    new_diag = {p: {"records": 0, "binding_days": 0, "n_solves": 0,
                    "neg_clipped_days": 0, "max_capacity_violation": 0.0}
                for p in NEW_POLICIES}
    warm_cache: dict = {}
    for r in R_EVAL:
        pr = POLICY_R[r]
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

                pols = {}
                if "a4" in run_policies:
                    pols["a4"] = make_a4(tuning["selected"][str(pr)])
                if "a4_sl" in run_policies and a4sl_by_r[pr] is not None:
                    pols["a4_sl"] = make_a4(a4sl_by_r[pr])
                if "l0_rule_ct" in run_policies:
                    pols["l0_rule_ct"] = RuleControlTower()
                if "a1" in run_policies:
                    pols["a1"] = LPForecastPolicy(
                        "a1", lp_replayer("tsmixer_a1", f), series)
                if "a2" in run_policies:
                    pols["a2"] = LPForecastPolicy(
                        "a2", lp_replayer(f"tsmixer_a2_r{POLICY_R[r]}", f),
                        series)
                if "b" in run_policies:
                    pols["b"] = LPForecastPolicy(
                        "b", lp_replayer(f"tsmixer_b_r{POLICY_R[r]}", f),
                        series)
                for name, p in pols.items():
                    p.name = name

                for name, pol in pols.items():
                    res = run_policy_window(
                        state, bp.test, hist, pol, params, kg[gid], sim_cfg,
                        t0_idx=f["test_start_idx"], fold=fold, group_id=gid,
                    )
                    episodes.extend({
                        "policy": name, "r": r, "fold": rec.fold,
                        "group_id": rec.group_id, "week": rec.week,
                        "cost": rec.cost,
                        "holding_cost": rec.holding_cost,
                        "shortage_cost": rec.shortage_cost,
                        "terminal_cost": rec.terminal_cost,
                        "csl": rec.csl, "fill_rate": rec.fill_rate,
                        "guardrail_blocked": rec.guardrail_blocked,
                        "binding_days": rec.binding_days,
                        "n_days": rec.n_days, "n_sku": rec.n_sku,
                    } for rec in res.records)
                    if name in NEW_POLICIES:
                        d = new_diag[name]
                        d["records"] += len(res.records)
                        d["binding_days"] += res.binding_days
                        d["n_solves"] += pol.n_solves
                        diag = pol.last_diagnostics()
                        d["neg_clipped_days"] += int(
                            diag.get("n_neg_clipped", 0))
                        d["max_capacity_violation"] = max(
                            d["max_capacity_violation"],
                            diag.get("capacity_violation", 0.0))
            print(f"r={r} fold {fold:02d} done "
                  f"({', '.join(sorted(pols))})")

    reg = None
    if any(p in BASELINE_POLICIES for p in run_policies):
        reg = regression_gate(
            episodes, [p for p in run_policies if p in BASELINE_POLICIES],
            R_EVAL)
        print(f"regression vs Stage 2-H: checked {reg['checked']}"
              f"/{reg['expected']} episodes, worst_rel "
              f"{reg['worst_rel']:.2e}, mismatches {len(reg['mismatches'])}, "
              f"env_drift_csl_quanta {reg['env_drift_csl_quanta']} (ST6-07)")
        for m in reg["mismatches"][:10]:
            print("  MISMATCH", m)
    for name in NEW_POLICIES:
        if name in run_policies:
            d = new_diag[name]
            print(f"{name}: execution diagnostics only (ST6-06 (3)) -- "
                  f"records {d['records']}, binding_days {d['binding_days']}, "
                  f"solves {d['n_solves']}, "
                  f"neg_clipped_days {d['neg_clipped_days']}, "
                  f"max_cap_violation {d['max_capacity_violation']:.2e}")

    out_path = OUT_DIR / f"stage6_eval_{SCENARIO}_{args.tag}.json"
    out_path.write_text(json.dumps({
        "task": "Stage 6 rolling evaluation (ST6-06 staged)",
        "date": datetime.date.today().isoformat(),
        "tag": args.tag,
        "policies": run_policies,
        "folds": fold_list,
        "r_eval": list(R_EVAL),
        "policy_r": {str(k): v for k, v in POLICY_R.items()},
        "scenario": {"name": SCENARIO, "p": SCENARIO_P,
                     "shock": "pre-registered fold-level path"},
        "kappa": sim_cfg.kappa_selected,
        "warmup_policy_by_r": {"3": "a4@r3", "5": "a4@r5", "10": "a4@r5"},
        "demand_path_sha256_by_fold": path_sha,
        "regression_vs_stage2": reg,
        "new_policy_diagnostics": {k: v for k, v in new_diag.items()
                                   if k in run_policies},
        "episodes": episodes,
        "environment": {"python": __import__("sys").version.split()[0],
                        "numpy": np.__version__,
                        "lock_file": "requirements-lock-stage6.txt"},
        "data_config_sha256": config_sha256(),
    }, indent=1) + "\n")
    print(f"written {out_path.relative_to(PROJECT_ROOT)} "
          f"({len(episodes)} episode rows)")


if __name__ == "__main__":
    main()