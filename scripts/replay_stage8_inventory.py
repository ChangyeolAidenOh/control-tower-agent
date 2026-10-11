"""Stage 8 instrumentation-only replay (ST8-04).

Re-runs selected (policy, r, fold) cells of the Stage 6 rolling evaluation
with ZERO changes to policy or engine code, to recover inventory-state
fields that the frozen stage6_eval_core_step6.json serialization dropped
(days_of_inventory exists in EpisodeRecord since commit c08f730; demand,
sales, inv_eod, order live in DayResult but were never written).

Setup mirrors scripts/run_stage6_eval.py exactly (constants, warm-up cache,
policy construction, replayer lineage). The per-policy execution mirrors
run_policy_window's body with the same frozen machinery (run_window,
DiagnosticsRecorder, terminal_cost, _aggregate_weeks) so that recomputed
episode records can be gated against the frozen JSON at rel <= 1e-9 before
any inventory number is adopted (ST8-04 integrity gate).

Pipeline (in-transit) reconstruction: start from the common warm-up state's
pipeline, then P_{t+1} = (P_t[1:], q_t) with q_t = DayResult.order. The
end-of-day in-transit metric is sum(P_t) + sum(q_t) (after sales and
ordering, before the midnight arrival -- same instant as inv_eod). The
reconstructed final pipeline must match run_window's final state
elementwise; any mismatch aborts the run.

Usage:
  python -m scripts.replay_stage8_inventory --policies l0_rule_ct --folds 1 --r 5 --tag probe_l0
  python -m scripts.replay_stage8_inventory --policies b --folds 1 --r 5 --tag probe_b
  python -m scripts.replay_stage8_inventory --policies l0_rule_ct,a4,a4_sl,b --folds 1-12 --r 5 --tag full
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import time

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
from core_pipeline.dfl.lp_problem import get_a3_saa_problem
from core_pipeline.policies.apto_policy import (
    APTOPolicy, load_safety_quantiles,
)
from core_pipeline.policies.lp_policy import (
    A3SAAPolicy, A5PerfectInfoPolicy, EvalPredictionReplayer,
    LPForecastPolicy, QuantileEvalReplayer,
)
from core_pipeline.policies.rule_ct import RuleControlTower
from core_pipeline.simulator.demand_paths import build_fold_paths
from core_pipeline.simulator.engine import (
    SkuParams, load_sim_config, run_window, terminal_cost,
)
from core_pipeline.simulator.episode import (
    DiagnosticsRecorder, _aggregate_weeks, compute_common_warmup,
)

TUNING_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"
FROZEN_PATH = PROJECT_ROOT / "outputs" / "stage6_eval_core_step6.json"
FROZEN_SHA = "1dc0cedafc3f452865fedaa32b8b5053f8338ab9238a0612f99023eb917614ce"
OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
SCENARIO = "core"
SCENARIO_P = 0.0
POLICY_R = {3: 3, 5: 5, 10: 5}
ALL_POLICIES = ("a4", "a4_sl", "l0_rule_ct", "b", "a1", "a2", "a3", "a5",
                "apto")
MART_LAST_IDX = 1941
GATE_REL_TOL = 1e-9
GATE_FLOAT_FIELDS = ("cost", "holding_cost", "shortage_cost",
                     "terminal_cost", "csl", "fill_rate")
GATE_INT_FIELDS = ("binding_days", "n_days", "n_sku")
WEEK_DAYS = 7


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


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def week_extensions(days: list, start_pipeline: np.ndarray,
                    final_pipeline: np.ndarray) -> list[dict]:
    """Per-week inventory/demand sums from the DayResult list, plus the
    deterministic pipeline reconstruction with an elementwise final check."""
    pipe = start_pipeline.copy()
    intransit_eod = []
    for d in days:
        intransit_eod.append(float(pipe.sum() + d.order.sum()))
        pipe = np.concatenate([pipe[:, 1:], d.order[:, None]], axis=1)
    if not np.array_equal(pipe, final_pipeline):
        max_abs = float(np.max(np.abs(pipe - final_pipeline)))
        raise SystemExit(
            f"pipeline reconstruction mismatch vs engine final state "
            f"(max abs {max_abs:.3e}) -- aborting")
    out = []
    n_weeks = (len(days) + WEEK_DAYS - 1) // WEEK_DAYS
    for wk in range(n_weeks):
        chunk = days[wk * WEEK_DAYS: (wk + 1) * WEEK_DAYS]
        it_chunk = intransit_eod[wk * WEEK_DAYS: (wk + 1) * WEEK_DAYS]
        out.append({
            "inv_eod_sum": float(sum(d.inv_eod.sum() for d in chunk)),
            "demand_sum": float(sum(d.demand.sum() for d in chunk)),
            "sales_sum": float(sum(d.sales.sum() for d in chunk)),
            "unmet_sum": float(sum(d.unmet.sum() for d in chunk)),
            "order_sum": float(sum(d.order.sum() for d in chunk)),
            "intransit_eod_sum": float(sum(it_chunk)),
        })
    return out


def gate_compare(row: dict, frozen: dict) -> list[dict]:
    fails = []
    for f in GATE_FLOAT_FIELDS:
        a, b = float(row[f]), float(frozen[f])
        rel = abs(a - b) / max(abs(b), 1.0)
        if rel > GATE_REL_TOL:
            fails.append({"field": f, "replay": a, "frozen": b, "rel": rel})
    for f in GATE_INT_FIELDS:
        if int(row[f]) != int(frozen[f]):
            fails.append({"field": f, "replay": row[f],
                          "frozen": frozen[f]})
    if bool(row["guardrail_blocked"]) != bool(frozen["guardrail_blocked"]):
        fails.append({"field": "guardrail_blocked",
                      "replay": row["guardrail_blocked"],
                      "frozen": frozen["guardrail_blocked"]})
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="l0_rule_ct,a4,a4_sl,b")
    ap.add_argument("--folds", default="1-12")
    ap.add_argument("--r", default="5")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()
    run_policies = [p.strip() for p in args.policies.split(",")]
    unknown = [p for p in run_policies if p not in ALL_POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}")
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))
    r_eval = tuple(int(x) for x in args.r.split(","))
    bad_r = [r for r in r_eval if r not in POLICY_R]
    if bad_r:
        raise SystemExit(f"unknown r {bad_r}")

    frozen_sha = file_sha256(FROZEN_PATH)
    if frozen_sha != FROZEN_SHA:
        raise SystemExit(
            f"frozen source sha mismatch: {frozen_sha} != {FROZEN_SHA}")
    frozen = json.loads(FROZEN_PATH.read_text())
    keys = ("policy", "r", "fold", "group_id", "week")
    frozen_map = {tuple(e[k] for k in keys): e for e in frozen["episodes"]}

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise SystemExit("kappa_selected missing in configs/simulator.yaml")
    tuning = json.loads(TUNING_PATH.read_text())

    conn = connect(load_config())
    try:
        folds_all = fetch_folds(conn)
        kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
        demand = fetch_demand_by_bundle(conn, MART_LAST_IDX)
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
        ref = frozen["demand_path_sha256_by_fold"].get(str(f["fold"]))
        if ref is not None and ref != path_sha[f["fold"]]:
            raise SystemExit(
                f"fold {f['fold']} demand path sha differs from frozen run")

    replayers: dict = {}
    q_replayers: dict = {}
    saa_cache: dict = {}

    def a3_replayer(f: dict) -> QuantileEvalReplayer:
        key = f["fold"]
        if key not in q_replayers:
            q_replayers[key] = QuantileEvalReplayer(
                model="tsmixer_a3", scenario=SCENARIO, fold=f["fold"],
                expected_origins=range(f["test_start_idx"],
                                       f["test_end_idx"] + 1),
                expected_series=all_series,
                expected_path_sha=path_sha[f["fold"]],
            )
        return q_replayers[key]

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

    apto_quantiles = {r: load_safety_quantiles(r, all_series)
                      for r in r_eval} if "apto" in run_policies else {}
    warm_cache: dict = {}
    episodes = []
    gate_fails = []
    gate_checked = 0
    timing: dict = {}
    for r in r_eval:
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
                if "a3" in run_policies:
                    prob = get_a3_saa_problem(
                        saa_cache, (gid, r), h, b,
                        np.full(len(series), sim_cfg.weight_wi),
                        float(kg[gid]), 29, sim_cfg.lead_time_core, 9)
                    pols["a3"] = A3SAAPolicy(
                        "a3", a3_replayer(f), series, prob)
                if "a5" in run_policies:
                    _, mat = demand[gid]
                    ts, te = f["test_start_idx"], f["test_end_idx"]
                    tail = mat[te:MART_LAST_IDX, :]
                    a5_path = np.concatenate([bp.test, tail], axis=0)
                    pols["a5"] = A5PerfectInfoPolicy(
                        "a5", a5_path, ts, MART_LAST_IDX, series)
                if "apto" in run_policies:
                    pols["apto"] = APTOPolicy(
                        "apto", lp_replayer("apto", f), series,
                        apto_quantiles[r])

                for name, p in pols.items():
                    p.name = name

                for name, pol in pols.items():
                    t0 = time.perf_counter()
                    recorder = DiagnosticsRecorder(pol)
                    final, days = run_window(
                        state.copy(), bp.test, hist, recorder, params,
                        k_g=kg[gid], cfg=sim_cfg,
                        t0_idx=f["test_start_idx"],
                    )
                    terminal = terminal_cost(final, params.h)
                    recs = _aggregate_weeks(days, params, sim_cfg, terminal,
                                            fold, gid, name)
                    exts = week_extensions(days, state.pipeline,
                                           final.pipeline)
                    elapsed = time.perf_counter() - t0
                    tk = (name, r)
                    agg = timing.setdefault(
                        tk, {"episodes": 0, "seconds": 0.0})
                    agg["episodes"] += len(recs)
                    agg["seconds"] += elapsed
                    for rec, ext in zip(recs, exts):
                        row = {
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
                            "days_of_inventory": rec.days_of_inventory,
                            **ext,
                        }
                        key = (name, r, rec.fold, rec.group_id, rec.week)
                        base = frozen_map.get(key)
                        if base is None:
                            gate_fails.append(
                                {"key": list(key),
                                 "field": "<missing in frozen>"})
                        else:
                            gate_checked += 1
                            for fail in gate_compare(row, base):
                                gate_fails.append(
                                    {"key": list(key), **fail})
                        episodes.append(row)
            print(f"r={r} fold {fold:02d} done "
                  f"({', '.join(sorted(pols))})")

    gate_pass = len(gate_fails) == 0 and gate_checked == len(episodes)
    print(f"integrity gate: checked {gate_checked}/{len(episodes)} episodes, "
          f"fails {len(gate_fails)}, pass {gate_pass}")
    for m in gate_fails[:10]:
        print("  GATE FAIL", m)
    for (name, r), agg in sorted(timing.items()):
        per_ep = agg["seconds"] / max(agg["episodes"], 1)
        print(f"timing {name} r={r}: {agg['seconds']:.1f}s total, "
              f"{agg['episodes']} episodes, {per_ep:.3f}s/episode")

    pooled = {}
    if gate_pass:
        for (name, r) in sorted({(e["policy"], e["r"]) for e in episodes}):
            rows = [e for e in episodes
                    if e["policy"] == name and e["r"] == r]
            n_days = sum(e["n_days"] for e in rows)
            inv_sum = sum(e["inv_eod_sum"] for e in rows)
            dem_sum = sum(e["demand_sum"] for e in rows)
            sal_sum = sum(e["sales_sum"] for e in rows)
            it_sum = sum(e["intransit_eod_sum"] for e in rows)
            pooled[f"{name}_r{r}"] = {
                "episodes": len(rows),
                "pooled_doi": (inv_sum / n_days) / (dem_sum / n_days)
                if dem_sum > 0 else None,
                "pooled_intransit_days": (it_sum / n_days)
                / (dem_sum / n_days) if dem_sum > 0 else None,
                "pooled_fill_rate_qty": sal_sum / dem_sum
                if dem_sum > 0 else None,
                "mean_daily_inv_eod": inv_sum / n_days,
                "mean_daily_intransit_eod": it_sum / n_days,
                "mean_daily_demand": dem_sum / n_days,
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"stage8_inventory_diagnostics_{args.tag}.json"
    out_path.write_text(json.dumps({
        "task": "Stage 8 inventory diagnostics replay (ST8-04)",
        "date": datetime.date.today().isoformat(),
        "tag": args.tag,
        "policies": run_policies,
        "folds": fold_list,
        "r_eval": list(r_eval),
        "source": {"path": FROZEN_PATH.name, "sha256": frozen_sha},
        "git_commit": git_head(),
        "definitions": {
            "days_of_inventory": "EpisodeRecord.days_of_inventory "
            "(episode.py, commit c08f730): mean daily SKU-sum EOD on-hand "
            "/ mean daily SKU-sum exogenous demand",
            "pooled_doi": "sum(inv_eod over days) / sum(exogenous demand "
            "over days), common evaluation window",
            "intransit_eod": "sum(P_t) + sum(q_t) at end of day (after "
            "sales and ordering, before midnight arrival); reconstructed "
            "from warm-up final pipeline + daily executed orders, verified "
            "elementwise against the engine final state",
            "pooled_fill_rate_qty": "sum(sales) / sum(exogenous demand)",
        },
        "gate": {"checked": gate_checked, "fails": gate_fails,
                 "rel_tol": GATE_REL_TOL, "pass": gate_pass},
        "timing": {f"{k[0]}_r{k[1]}": v for k, v in sorted(timing.items())},
        "pooled": pooled,
        "episodes": episodes,
        "environment": {"python": __import__("sys").version.split()[0],
                        "numpy": np.__version__,
                        "lock_file": "requirements-lock-stage6.txt"},
        "data_config_sha256": config_sha256(),
    }, indent=1) + "\n")
    print(f"written {out_path.relative_to(PROJECT_ROOT)} "
          f"({len(episodes)} episode rows, gate_pass={gate_pass})")
    if not gate_pass:
        print("gate failed: inventory results are NOT adopted (ST8-04)")


if __name__ == "__main__":
    main()
