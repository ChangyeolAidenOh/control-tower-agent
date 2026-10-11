"""ST7-09: common-state H5 reconstruction -- zero additional LLM calls.

For every episode (fold x bundle x week, 480), rebuild the LLM arm's
selection-time state by re-advancing the stored selected actions through
the frozen engine, then evaluate BOTH arms' candidate sets at that SAME
state with the identical 9-scenario / 10-day / gate / D4-rank machinery:
  - LLM candidates  : re-parsed from the stored raw generation responses
                      (outputs/agent_llm_raw/official/*_gen.json)
  - numeric         : regenerated from the frozen schedule (sha-verified)
Arm achieved cost = gate-passing winner's mean_cost; if ALL candidates are
rejected, the no-action A4 action's 10-day evaluation cost (same function,
same state -- identical for both arms by construction, hence a tie).

Integrity anchors (ST7-09; ANY failure aborts before writing):
  A1  re-advanced realized path == official episode records (cost/csl,
      <=1e-9 rel) for all 480 rows
  A2  re-selected LLM winner == stored week log (slot, action, mean_cost)
      for every non-fallback week; fallback weeks re-select to None

Console stays inside the ST7-04 seal: anchors / counts / timings only.
Costs and verdicts go to outputs/agent_h5_common_state.json (sha printed);
scripts/report_agent_h3h5.py is the joint-disclosure reader.

Run:  python -m scripts.replay_h5_common_state
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from core_pipeline.agent import llm_adapter as la
from core_pipeline.agent.candidates import make_action
from core_pipeline.agent.gate import (
    SELECTION_DAYS,
    evaluate_candidate,
    select_candidate,
)
from core_pipeline.agent.actions import AgentA4Policy
from core_pipeline.agent.search import actions_from_entries, canonical_sha
from core_pipeline.agent.selection_replay import AgentSelectionReplayer
from core_pipeline.agent.weekly_runner import N_WEEKS, WINDOW_DAYS
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
from core_pipeline.simulator.engine import SkuParams, load_sim_config, run_window, terminal_cost
from core_pipeline.simulator.episode import WEEK_DAYS, compute_common_warmup, _aggregate_weeks

TUNING_PATH = PROJECT_ROOT / "outputs" / "stage2_a4_tuning.json"
LLM_PATH = PROJECT_ROOT / "outputs" / "agent_llm_official.json"
SCHED_PATH = PROJECT_ROOT / "outputs" / "agent_search_schedule.json"
RAW_DIR = PROJECT_ROOT / "outputs" / "agent_llm_raw" / "official"
OUT_PATH = PROJECT_ROOT / "outputs" / "agent_h5_common_state.json"
SCENARIO_P = 0.0
R_ENV = 5
MART_LAST_IDX = 1941
REL_TOL = 1e-9
BIG_KG = 1e18


def rel_ok(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def scenario_sha(paths: dict, group_ids: list[str]) -> str:
    sha = hashlib.sha256()
    for gid in group_ids:
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(
            paths[gid].test, dtype=np.float64).tobytes())
    return sha.hexdigest()


def main() -> None:
    t_run0 = time.time()
    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    floor = sim_cfg.episode_csl_floor
    sel = json.loads(TUNING_PATH.read_text())["selected"][str(R_ENV)]
    base_tau = float(sel["tau"])
    warmup_a4 = A4BaseStock(A4Config(tau=base_tau,
                                     allocation=sel["allocation_rule"]))

    official = json.loads(LLM_PATH.read_text())
    weeks_by_key = {(w["fold"], w["group_id"], w["week"]): w
                    for w in official["weeks"]}
    eps_by_key = {(e["fold"], e["group_id"], e["week"]): e
                  for e in official["episodes"]}
    if len(weeks_by_key) != 480 or len(eps_by_key) != 480:
        raise SystemExit("official LLM JSON does not hold 480 weeks/episodes")

    stored = json.loads(SCHED_PATH.read_text())
    if canonical_sha(stored["schedule"]) != stored["schedule_sha256"]:
        raise SystemExit("schedule sha self-check failed")
    sched_eps = stored["schedule"]["episodes"]

    db.load_dotenv()
    conn = db.connect(db.load_config())
    try:
        folds_all = fetch_folds(conn)
        kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
        demand = fetch_demand_by_bundle(conn, MART_LAST_IDX)
        costs = fetch_sku_costs(conn)
    finally:
        conn.close()
    group_ids = sorted(demand)
    all_series = tuple(s for gid in group_ids for s in demand[gid][0])

    rows = []
    a1_ok = a2_ok = 0
    n_noact_cap_reeval = 0
    for fold_no in sorted({f["fold"] for f in folds_all}):
        f = next(x for x in folds_all if x["fold"] == fold_no)
        ts = f["test_start_idx"]
        paths = build_fold_paths(
            demand, fold=fold_no,
            warmup_start_row=f["warmup_start_idx"] - 1,
            test_start_row=ts - 1, test_end_row=f["test_end_idx"] - 1,
            p=SCENARIO_P, cfg=sim_cfg,
        )
        replayer = AgentSelectionReplayer(
            fold=fold_no,
            expected_origins=[ts - 1 + 7 * w for w in range(N_WEEKS)],
            expected_series=all_series,
            expected_path_sha=scenario_sha(paths, group_ids),
        )

        for gid in group_ids:
            bp = paths[gid]
            series, _ = demand[gid]
            h, b = cost_arrays(series, costs, R_ENV)
            pr = SkuParams(series_ids=series, h=h, b=b,
                           w=np.full(len(series), sim_cfg.weight_wi))
            state = compute_common_warmup(
                bp.warmup, bp.history, warmup_a4, pr, kg[gid], sim_cfg,
                warmup_start_idx=f["warmup_start_idx"],
            )
            base_hist = np.concatenate([bp.history, bp.warmup], axis=0)
            no_action = make_action(None, {}, base_tau, series)
            days_all = []

            for w in range(N_WEEKS):
                t_w = ts - 1 + 7 * w
                hist_sel = np.concatenate(
                    [base_hist, bp.test[: 7 * w]], axis=0)
                common = state.copy()
                scen = replayer.get(t_w, series)[:SELECTION_DAYS]
                stored_wk = weeks_by_key[(fold_no, gid, w + 1)]

                # --- LLM arm: re-parse stored raw generation response
                raw = json.loads(
                    (RAW_DIR / f"f{fold_no}_{gid}_w{w}_gen.json").read_text())
                acts_l = []
                if raw["outcome"] == la.OUTCOME_OK:
                    parse = la.process_generation_response(
                        raw["text"], series, base_tau)
                    if not parse.parse_failed:
                        acts_l = [c.action for c in parse.valid]
                evals_l = [evaluate_candidate(
                    i, a, scen, common, hist_sel, series, pr, kg[gid],
                    sim_cfg, t0_idx=t_w + 1, csl_floor=floor)
                    for i, a in enumerate(acts_l)]
                win_l = select_candidate(evals_l)

                # --- anchor A2
                if stored_wk["fallback"]:
                    ok2 = win_l is None
                else:
                    ok2 = (win_l is not None
                           and win_l.slot == stored_wk["selected_slot"]
                           and rel_ok(win_l.mean_cost,
                                      stored_wk["selected_mean_cost"])
                           and win_l.action.tau_eff
                           == stored_wk["selected_tau"]
                           and list(map(list, win_l.action.multipliers))
                           == stored_wk["selected_multipliers"])
                a2_ok += int(ok2)
                if not ok2:
                    raise SystemExit(
                        f"anchor A2 failed f{fold_no} {gid} w{w + 1}")

                # --- numeric arm: regenerate from the frozen schedule
                acts_n = actions_from_entries(
                    sched_eps[f"f{fold_no}|{gid}|w{w}"], base_tau, series)
                evals_n = [evaluate_candidate(
                    i, a, scen, common, hist_sel, series, pr, kg[gid],
                    sim_cfg, t0_idx=t_w + 1, csl_floor=floor)
                    for i, a in enumerate(acts_n)]
                win_n = select_candidate(evals_n)

                # --- no-action evaluation (shared fallback cost)
                ev0 = evaluate_candidate(
                    0, no_action, scen, common, hist_sel, series, pr,
                    kg[gid], sim_cfg, t0_idx=t_w + 1, csl_floor=floor)
                noact_cost = ev0.mean_cost
                if not np.isfinite(noact_cost):
                    n_noact_cap_reeval += 1
                    noact_cost = evaluate_candidate(
                        0, no_action, scen, common, hist_sel, series, pr,
                        BIG_KG, sim_cfg, t0_idx=t_w + 1,
                        csl_floor=floor).mean_cost

                c_l = win_l.mean_cost if win_l is not None else noact_cost
                c_n = win_n.mean_cost if win_n is not None else noact_cost
                rows.append({
                    "fold": fold_no, "group_id": gid, "week": w + 1,
                    "origin_idx": t_w,
                    "cost_llm": float(c_l), "cost_numeric": float(c_n),
                    "llm_fallback": win_l is None,
                    "numeric_fallback": win_n is None,
                    "llm_candidates": len(acts_l),
                    "numeric_candidates": len(acts_n),
                    "noaction_cost": float(noact_cost),
                })

                # --- advance the realized LLM path (anchored action)
                act = win_l.action if win_l is not None else no_action
                policy = AgentA4Policy(
                    act, series, action_start_idx=t_w + 1,
                    tol_rel=sim_cfg.feasibility_tol_rel,
                    capacity_mode="scale",
                )
                state, days = run_window(
                    state, bp.test[7 * w: 7 * (w + 1)], hist_sel, policy,
                    pr, k_g=kg[gid], cfg=sim_cfg, t0_idx=t_w + 1,
                )
                days_all.extend(days)

            term = terminal_cost(state, pr.h)
            recs = _aggregate_weeks(days_all, pr, sim_cfg, term, fold_no,
                                    gid, "agent_llm")
            if len(days_all) != WINDOW_DAYS or len(recs) != N_WEEKS:
                raise SystemExit(f"replay shape off f{fold_no} {gid}")
            for rec in recs:
                ref = eps_by_key[(fold_no, gid, rec.week)]
                if not (rel_ok(rec.cost, ref["cost"])
                        and rel_ok(rec.csl, ref["csl"])):
                    raise SystemExit(
                        f"anchor A1 failed f{fold_no} {gid} wk{rec.week}")
                a1_ok += 1
            print(f"f{fold_no:02d} {gid} anchors ok "
                  f"(A1 {a1_ok}/480 cum, A2 {a2_ok}/480 cum)")

    payload = {
        "task": "ST7-09 common-state H5 reconstruction",
        "freeze_payload_sha256": official["freeze_payload_sha256"],
        "schedule_sha256": stored["schedule_sha256"],
        "csl_floor": floor, "r_env": R_ENV,
        "anchors_all_ok": a1_ok == 480 and a2_ok == 480,
        "anchor_a1_ok": a1_ok, "anchor_a2_ok": a2_ok,
        "noaction_capacity_reeval_weeks": n_noact_cap_reeval,
        "episodes": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, sort_keys=True))
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"anchors A1={a1_ok}/480 A2={a2_ok}/480 "
          f"noaction_cap_reeval={n_noact_cap_reeval} "
          f"wall_s={time.time() - t_run0:.0f}")
    print(f"wrote {OUT_PATH.name} sha256={sha[:16]} "
          f"(costs sealed until joint disclosure, ST7-04)")


if __name__ == "__main__":
    main()