"""ST7-06/ST7-07: LLM-arm runner -- smoke and official modes (ST7-04).

Pre-flight hard gates (both modes): the freeze manifest must match the live
code (payload sha, adapter file sha, schema sha, SDK version) and
ANTHROPIC_API_KEY must be set -- any mismatch aborts BEFORE the first paid
call. Smoke runs the frozen episode set (fold 1, ranks 0-1); official runs
--folds (default 1-12) over every bundle = 480 episodes, 960 calls under
the frozen contract -- no prompt, model or parameter change of any kind.

Console surfaces technical-contract facts only: outcomes, tokens, latency,
legality counters, spend and the per-episode selection-time median (the
numeric arm's official time budget, ST7-01). Performance quantities are
written to the sealed JSON with sha256, never printed (ST7-04). Raw
responses are stored in full. Budget exhaustion (conservative guard) and
auth failures ABORT the run (reviewer table, ST7-06); timeouts/429/5xx are
recorded as tech errors with zero extra calls and the episode falls back.

Run:
  python -m scripts.run_agent_llm --mode smoke
  caffeinate -i python -m scripts.run_agent_llm --mode official --tag official
"""

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from core_pipeline.agent import llm_adapter as la
from core_pipeline.agent.search import canonical_sha
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
FREEZE_PATH = PROJECT_ROOT / "outputs" / "agent_llm_freeze.json"
ADAPTER_PATH = PROJECT_ROOT / "core_pipeline" / "agent" / "llm_adapter.py"
OUT_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = OUT_DIR / "agent_llm_raw"
SCENARIO_P = 0.0
R_ENV = 5
MART_LAST_IDX = 1941
POLICY_NAME = "agent_llm"


def verify_freeze() -> dict:
    manifest = json.loads(FREEZE_PATH.read_text())
    import anthropic
    checks = {
        "payload_sha": (manifest["freeze_payload_sha256"]
                        == canonical_sha(la.freeze_payload())),
        "adapter_sha": (manifest["adapter_file_sha256"]
                        == hashlib.sha256(ADAPTER_PATH.read_bytes()).hexdigest()),
        "schema_sha": manifest["schema_sha256"] == la.schema_sha(),
        "sdk_version": manifest["anthropic_sdk_version"] == anthropic.__version__,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            f"freeze verification failed: {failed}; re-freeze (with a "
            "decision-log entry) before any paid call (ST7-06)")
    return manifest


def scenario_sha(paths: dict, group_ids: list[str]) -> str:
    sha = hashlib.sha256()
    for gid in group_ids:
        sha.update(gid.encode())
        sha.update(np.ascontiguousarray(
            paths[gid].test, dtype=np.float64).tobytes())
    return sha.hexdigest()


def record_dict(rec: la.CallRecord) -> dict:
    return {
        "outcome": rec.outcome, "stop_reason": rec.stop_reason,
        "request_id": rec.request_id, "input_tokens": rec.input_tokens,
        "output_tokens": rec.output_tokens, "latency_s": rec.latency_s,
        "response_sha256": rec.response_sha256, "error": rec.error,
    }


def save_raw(tag: str, name: str, rec: la.CallRecord) -> None:
    path = RAW_DIR / tag / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record_dict(rec), "text": rec.text,
                                "raw": rec.raw}, sort_keys=True))


def replayer_get(replayer: AgentSelectionReplayer, series):
    def get(origin_idx: int):
        return replayer.get(origin_idx, series)
    return get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "official"], default="smoke")
    ap.add_argument("--folds", type=str, default="1-12")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    tag = args.tag or args.mode

    db.load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; aborting before "
                         "any paid call")
    manifest = verify_freeze()
    print(f"freeze ok payload={manifest['freeze_payload_sha256'][:12]} "
          f"sdk={manifest['anthropic_sdk_version']} mode={args.mode}")

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    tuning = json.loads(TUNING_PATH.read_text())
    sel = tuning["selected"][str(R_ENV)]
    base_tau = float(sel["tau"])
    warmup_a4 = A4BaseStock(A4Config(tau=base_tau,
                                     allocation=sel["allocation_rule"]))

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

    if args.mode == "smoke":
        smoke = manifest["smoke_episodes"]
        fold_list = [int(smoke["fold"])]
        gid_by_fold = {fold_list[0]: [group_ids[int(r)]
                                      for r in smoke["bundle_ranks"]]}
    else:
        lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
        fold_list = list(range(int(lo), int(hi) + 1))
        gid_by_fold = {f: group_ids for f in fold_list}

    client = la.make_client()
    ledger = la.CostLedger()
    episodes, week_rows = [], []
    sel_times, gen_latencies, expl_latencies = [], [], []
    outcome_counts: dict[str, int] = {}
    valid_slots = slot_total = 0
    integrity_ok = True
    run_t0 = time.time()

    for fold_no in fold_list:
        f = next(x for x in folds_all if x["fold"] == fold_no)
        ts = f["test_start_idx"]
        paths = build_fold_paths(
            demand, fold=fold_no,
            warmup_start_row=f["warmup_start_idx"] - 1,
            test_start_row=ts - 1, test_end_row=f["test_end_idx"] - 1,
            p=SCENARIO_P, cfg=sim_cfg,
        )
        path_sha = scenario_sha(paths, group_ids)
        replayer = AgentSelectionReplayer(
            fold=fold_no,
            expected_origins=[ts - 1 + 7 * w for w in range(N_WEEKS)],
            expected_series=all_series,
            expected_path_sha=path_sha,
        )

        for gid in gid_by_fold[fold_no]:
            bp = paths[gid]
            series, _ = demand[gid]
            h, b = cost_arrays(series, costs, R_ENV)
            pr = SkuParams(series_ids=series, h=h, b=b,
                           w=np.full(len(series), sim_cfg.weight_wi))
            state = compute_common_warmup(
                bp.warmup, bp.history, warmup_a4, pr, kg[gid], sim_cfg,
                warmup_start_idx=f["warmup_start_idx"],
            )
            hist = np.concatenate([bp.history, bp.warmup], axis=0)
            gen_meta: dict[int, dict] = {}

            def provider(w, ctx, _gid=gid, _series=series, _ts=ts,
                         _fold=fold_no):
                expected = _ts - 1 + 7 * w
                if ctx.origin_idx != expected:
                    raise RuntimeError(
                        f"origin mismatch: ctx {ctx.origin_idx} != {expected}")
                prompt = la.build_generation_prompt(
                    _gid, _series, h, b,
                    recent_demand=ctx.hist[-la.RECENT_DAYS:],
                    inv=ctx.state.inv, pipeline=ctx.state.pipeline,
                    k_g=float(kg[_gid]), base_tau=base_tau,
                    week=w + 1, prev_week=ctx.prev,
                )
                rec = la.call_generation(client, ledger, la.GEN_SYSTEM,
                                         prompt)
                save_raw(tag, f"f{_fold}_{_gid}_w{w}_gen", rec)
                if rec.outcome in la.FATAL_OUTCOMES:
                    raise SystemExit(f"fatal: {rec.outcome} {rec.error}")
                parse = None
                if rec.outcome == la.OUTCOME_OK:
                    parse = la.process_generation_response(
                        rec.text, _series, base_tau)
                stats = (parse.stats() if parse is not None
                         and not parse.parse_failed else
                         {"candidate_slots_used": 0, "valid_candidates": 0,
                          "invalid_candidates": 0,
                          "duplicate_valid_slots": 0, "truncated_count": 0})
                gen_meta[w] = {
                    "call": record_dict(rec),
                    "parse_failed": parse is None or parse.parse_failed,
                    "failure_reason": (None if parse is None
                                       else parse.failure_reason),
                    **stats}
                gen_latencies.append(rec.latency_s)
                outcome_counts[rec.outcome] = (
                    outcome_counts.get(rec.outcome, 0) + 1)
                nonlocal_counts["valid"] += stats["valid_candidates"]
                nonlocal_counts["slots"] += stats["candidate_slots_used"]
                if parse is None or parse.parse_failed:
                    return []
                return [c.action for c in parse.valid]

            nonlocal_counts = {"valid": 0, "slots": 0}
            res = run_agent_window(
                state, bp.test, hist, provider,
                replayer_get(replayer, series),
                series, pr, kg[gid], sim_cfg,
                test_start_idx=ts, fold=fold_no, group_id=gid,
                base_tau=base_tau, policy_name=POLICY_NAME,
            )
            valid_slots += nonlocal_counts["valid"]
            slot_total += nonlocal_counts["slots"]

            for wk in res.weeks:
                expl_prompt = la.build_explanation_prompt(
                    gid, wk.week,
                    {"tau_eff": wk.selected_tau,
                     "multipliers": wk.selected_multipliers},
                    wk.gate)
                rec = la.call_explanation(client, ledger, expl_prompt)
                save_raw(tag, f"f{fold_no}_{gid}_w{wk.week - 1}_expl", rec)
                if rec.outcome in la.FATAL_OUTCOMES:
                    raise SystemExit(f"fatal: {rec.outcome} {rec.error}")
                expl_latencies.append(rec.latency_s)
                outcome_counts[rec.outcome] = (
                    outcome_counts.get(rec.outcome, 0) + 1)
                sel_times.append(wk.selection_time_s)
                week_rows.append({
                    "fold": fold_no, "group_id": gid, "week": wk.week,
                    "origin_idx": wk.origin_idx, "gate": wk.gate,
                    "fallback": wk.fallback,
                    "selected_slot": wk.selected_slot,
                    "selected_tau": wk.selected_tau,
                    "selected_multipliers": list(
                        map(list, wk.selected_multipliers)),
                    "selected_mean_cost": wk.selected_mean_cost,
                    "scaling": wk.scaling,
                    "selection_time_s": wk.selection_time_s,
                    "candidates_provided": wk.candidates_provided,
                    "budget_exceeded": wk.budget_exceeded,
                    "generation": gen_meta[wk.week - 1],
                    "explanation": {
                        **record_dict(rec),
                        "text": (rec.text
                                 if rec.outcome == la.OUTCOME_OK else "")},
                })

            ok = (res.n_days == WINDOW_DAYS and len(res.records) == N_WEEKS
                  and abs(sum(r.cost for r in res.records) - res.window_cost)
                  <= 1e-9 * max(1.0, abs(res.window_cost)))
            integrity_ok = integrity_ok and ok
            print(f"f{fold_no:02d} {gid} ok={ok} "
                  f"valid={nonlocal_counts['valid']}/"
                  f"{nonlocal_counts['slots']} "
                  f"sel_med={np.median([w_.selection_time_s for w_ in res.weeks]):.1f}s "
                  f"spent=${ledger.spent_usd:.3f}")
            episodes.extend({
                "policy": POLICY_NAME, "r": R_ENV, "fold": rec_.fold,
                "group_id": rec_.group_id, "week": rec_.week,
                "cost": rec_.cost, "holding_cost": rec_.holding_cost,
                "shortage_cost": rec_.shortage_cost,
                "terminal_cost": rec_.terminal_cost,
                "csl": rec_.csl, "fill_rate": rec_.fill_rate,
                "guardrail_blocked": rec_.guardrail_blocked,
                "binding_days": rec_.binding_days,
                "n_days": rec_.n_days, "n_sku": rec_.n_sku,
            } for rec_ in res.records)

    sel_median = float(np.median(sel_times)) if sel_times else 0.0
    n_calls = ledger.calls
    mean_cost = ledger.spent_usd / n_calls if n_calls else 0.0
    out_path = OUT_DIR / f"agent_llm_{tag}.json"
    payload = {
        "task": f"ST7 LLM-arm {args.mode} run",
        "tag": tag,
        "date": datetime.date.today().isoformat(),
        "freeze_payload_sha256": manifest["freeze_payload_sha256"],
        "mode": args.mode, "folds": fold_list, "r_env": R_ENV,
        "policy": POLICY_NAME,
        "ledger": {"calls": ledger.calls,
                   "input_tokens": ledger.input_tokens,
                   "output_tokens": ledger.output_tokens,
                   "spent_usd": ledger.spent_usd},
        "selection_time_median_s": sel_median,
        "outcome_counts": outcome_counts,
        "performance_withheld_until_official": args.mode == "smoke",
        "episodes": episodes,
        "weeks": week_rows,
    }
    out_path.write_text(json.dumps(payload, sort_keys=True))
    file_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"integrity_all_ok={integrity_ok} episode_rows={len(episodes)} "
          f"wall_s={time.time() - run_t0:.0f}")
    print(f"outcomes={outcome_counts} legal_slots={valid_slots}/{slot_total}")
    print(f"calls={n_calls} tokens_in={ledger.input_tokens} "
          f"tokens_out={ledger.output_tokens} "
          f"spent_usd={ledger.spent_usd:.4f} mean_call_usd={mean_cost:.5f}")
    print(f"selection_time_median_s={sel_median:.2f} "
          f"gen_mean={np.mean(gen_latencies) if gen_latencies else 0:.1f}s "
          f"expl_mean={np.mean(expl_latencies) if expl_latencies else 0:.1f}s")
    print(f"wrote {out_path.name} sha256={file_sha[:16]} "
          f"(performance sealed until joint disclosure, ST7-04)")


if __name__ == "__main__":
    main()