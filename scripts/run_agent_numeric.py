"""ST7-04: numeric-search arm runner with performance-viewing controls.

Smoke protocol (ST7-04, fold 1 default): the console surfaces EXECUTION
facts only -- wiring success, exceptions, input/output integrity booleans,
candidate slot counts, wall time. Policy-performance quantities (cost, CSL,
gate pass rate, fallback rate) are computed, written to the output JSON and
sha-hashed, but NEVER printed; they stay sealed until the LLM prompt and
adapter are frozen (ST7-04). The full-480 official run reuses this runner
with --folds 1-12 AFTER that freeze; no code path differs between smoke and
official beyond the fold list and tag.

Wiring per (fold, bundle), r = 5 single environment (ST7-01):
  schedule        outputs/agent_search_schedule.json, canonical-sha verified
                  against regeneration-free stored hash before use;
  scenario paths  build_fold_paths (frozen seeds) + the demand-path sha rule
                  of build_eval_predictions, handed to the selection
                  replayer (lineage hard-fail on mismatch);
  warm-up         compute_common_warmup with the Track A tuned A4 @ r5
                  (tau read from stage2_a4_tuning.json, never hardcoded);
  selection       AgentSelectionReplayer origins T-1+7w (ST7-03 parquet);
  execution       run_agent_window (weekly sequential, scale mode).

Run:
  python -m scripts.run_agent_numeric --folds 1 --tag smoke
"""

import argparse
import datetime
import hashlib
import json
import time

import numpy as np

from core_pipeline.agent.search import actions_from_entries, canonical_sha
from core_pipeline.agent.selection_replay import AgentSelectionReplayer
from core_pipeline.agent.weekly_runner import (
    N_WEEKS,
    WINDOW_DAYS,
    run_agent_window,
)
from core_pipeline.data import db
from core_pipeline.data.db import PROJECT_ROOT
from core_pipeline.data.mart_reader import (
    cost_arrays,
    fetch_bundle_capacity,
    fetch_demand_by_bundle,
    fetch_folds,
    fetch_sku_costs,
)
from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.simulator.demand_paths import build_fold_paths
from core_pipeline.simulator.engine import SkuParams, load_sim_config
from core_pipeline.simulator.episode import compute_common_warmup

TUNING_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"
SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "agent_search_schedule.json"
OUT_DIR = PROJECT_ROOT / "outputs"
SCENARIO_P = 0.0
R_ENV = 5
MART_LAST_IDX = 1941
POLICY_NAME = "agent_numeric"


def scenario_sha(paths: dict, group_ids: list[str]) -> str:
    """Same rule as build_eval_predictions.scenario_for_fold (ST6-03 (5))."""
    sha = hashlib.sha256()
    for gid in group_ids:
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(
            paths[gid].test, dtype=np.float64).tobytes())
    return sha.hexdigest()


def load_schedule() -> tuple[dict, str]:
    stored = json.loads(SCHEDULE_PATH.read_text())
    sha = canonical_sha(stored["schedule"])
    if sha != stored["schedule_sha256"]:
        raise SystemExit(
            f"schedule sha mismatch: stored {stored['schedule_sha256'][:12]} "
            f"!= recomputed {sha[:12]}")
    return stored["schedule"], sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1")
    ap.add_argument("--tag", type=str, default="smoke")
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise SystemExit("kappa_selected missing in configs/simulator.yaml")
    tuning = json.loads(TUNING_PATH.read_text())
    sel = tuning["selected"][str(R_ENV)]
    base_tau = float(sel["tau"])
    warmup_a4 = A4BaseStock(A4Config(tau=base_tau,
                                     allocation=sel["allocation_rule"]))

    schedule, schedule_sha = load_schedule()
    print(f"schedule ok episodes={len(schedule['episodes'])} "
          f"sha={schedule_sha[:12]}")

    db.load_dotenv()
    conn = db.connect(db.load_config())
    try:
        folds_all = fetch_folds(conn)
        kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
        demand = fetch_demand_by_bundle(conn, MART_LAST_IDX)
        costs = fetch_sku_costs(conn)
    finally:
        conn.close()
    folds = [f for f in folds_all if f["fold"] in fold_list]
    if len(folds) != len(fold_list):
        raise SystemExit(f"folds {fold_list} not all present in mart")
    group_ids = sorted(demand)
    all_series = tuple(s for gid in group_ids for s in demand[gid][0])

    episodes = []
    week_rows = []
    timings = []
    integrity_ok = True
    for f in folds:
        fold = f["fold"]
        ts = f["test_start_idx"]
        paths = build_fold_paths(
            demand, fold=fold,
            warmup_start_row=f["warmup_start_idx"] - 1,
            test_start_row=ts - 1,
            test_end_row=f["test_end_idx"] - 1,
            p=SCENARIO_P, cfg=sim_cfg,
        )
        path_sha = scenario_sha(paths, group_ids)
        sel_origins = [ts - 1 + 7 * w for w in range(N_WEEKS)]
        replayer = AgentSelectionReplayer(
            fold=fold,
            expected_origins=sel_origins,
            expected_series=all_series,
            expected_path_sha=path_sha,
        )
        print(f"fold {fold:02d} replayer ok origins={sel_origins} "
              f"path_sha={path_sha[:12]}")

        for rank, gid in enumerate(group_ids):
            bp = paths[gid]
            series, _ = demand[gid]
            h, b = cost_arrays(series, costs, R_ENV)
            params = SkuParams(series_ids=series, h=h, b=b,
                               w=np.full(len(series), sim_cfg.weight_wi))
            state = compute_common_warmup(
                bp.warmup, bp.history, warmup_a4, params, kg[gid],
                sim_cfg, warmup_start_idx=f["warmup_start_idx"],
            )
            hist = np.concatenate([bp.history, bp.warmup], axis=0)

            def provider(w, ctx, _fold=fold, _gid=gid):
                entries = schedule["episodes"][f"f{_fold}|{_gid}|w{w}"]
                return actions_from_entries(entries, base_tau,
                                            demand[_gid][0])

            def forecast(origin_idx, _rep=replayer, _series=series):
                return _rep.get(origin_idx, _series)

            t0 = time.perf_counter()
            res = run_agent_window(
                state, bp.test, hist, provider, forecast, series, params,
                kg[gid], sim_cfg,
                test_start_idx=ts, fold=fold, group_id=gid,
                base_tau=base_tau, policy_name=POLICY_NAME,
            )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed / N_WEEKS)

            cost_identity = abs(
                sum(r.cost for r in res.records) - res.window_cost
            ) <= 1e-9 * max(1.0, abs(res.window_cost))
            terminal_last_only = all(
                r.terminal_cost == 0.0 for r in res.records[:-1])
            days_ok = res.n_days == WINDOW_DAYS
            records_ok = len(res.records) == N_WEEKS
            slots = sum(w.gate["evaluated_candidates"] for w in res.weeks)
            ok = cost_identity and terminal_last_only and days_ok and records_ok
            integrity_ok = integrity_ok and ok
            print(f"fold {fold:02d} {gid} ok={ok} days={res.n_days} "
                  f"records={len(res.records)} slots={slots} "
                  f"elapsed={elapsed:.1f}s")

            episodes.extend({
                "policy": POLICY_NAME, "r": R_ENV, "fold": rec.fold,
                "group_id": rec.group_id, "week": rec.week,
                "cost": rec.cost, "holding_cost": rec.holding_cost,
                "shortage_cost": rec.shortage_cost,
                "terminal_cost": rec.terminal_cost,
                "csl": rec.csl, "fill_rate": rec.fill_rate,
                "guardrail_blocked": rec.guardrail_blocked,
                "binding_days": rec.binding_days,
                "n_days": rec.n_days, "n_sku": rec.n_sku,
            } for rec in res.records)
            week_rows.extend({
                "fold": fold, "group_id": gid, "week": wk.week,
                "origin_idx": wk.origin_idx, "gate": wk.gate,
                "fallback": wk.fallback,
                "selected_slot": wk.selected_slot,
                "selected_tau": wk.selected_tau,
                "selected_multipliers": list(map(list,
                                                 wk.selected_multipliers)),
                "selected_mean_cost": wk.selected_mean_cost,
                "scaling": wk.scaling,
            } for wk in res.weeks)

    out_path = OUT_DIR / f"agent_numeric_{args.tag}.json"
    payload = {
        "task": "ST7-04 numeric-search arm run",
        "tag": args.tag,
        "date": datetime.date.today().isoformat(),
        "folds": fold_list,
        "r_env": R_ENV,
        "base_tau": base_tau,
        "schedule_sha256": schedule_sha,
        "policy": POLICY_NAME,
        "performance_withheld_until_llm_freeze": args.tag == "smoke",
        "episodes": episodes,
        "weeks": week_rows,
    }
    out_path.write_text(json.dumps(payload, sort_keys=True))
    file_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    med = float(np.median(timings)) if timings else float("nan")
    print(f"integrity_all_ok={integrity_ok} episode_rows={len(episodes)} "
          f"week_rows={len(week_rows)}")
    print(f"median_episode_s={med:.2f} total_s={sum(timings) * N_WEEKS:.0f}")
    print(f"wrote {out_path.name} sha256={file_sha[:16]} "
          f"(performance metrics sealed, ST7-04)")


if __name__ == "__main__":
    main()