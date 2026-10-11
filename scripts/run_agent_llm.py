"""ST7-06 Step 3: LLM-arm runner, smoke mode (frozen contract, ST7-04).

Pre-flight hard gates: the freeze manifest must match the live code (payload
sha, adapter file sha, schema sha, SDK version) and ANTHROPIC_API_KEY must be
set -- any mismatch aborts BEFORE the first paid call. Smoke episodes come
from the frozen manifest (fold 1, bundle ranks 0-1, weeks 0-3 = 8 episodes,
<= 16 calls). Technical-contract facts only reach the console: call outcome,
stop_reason, tokens, latency, candidate legality/truncation, cost spent and
the 960-call extrapolation (L1). Performance quantities (cost, CSL, gate
pass, fallback) are computed, written to the sealed JSON and sha-recorded,
never printed (ST7-04). Raw responses are stored in full with sha256 (L4).

Timing split (L6): selection_time_s covers generation + parse + scenario
evaluation + gate + ranking + realized execution of the window loop;
explanation_time_s is measured separately after the window and is excluded
from any H5 time budget.

Run:
  python -m scripts.run_agent_llm --tag smoke
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="smoke")
    args = ap.parse_args()

    db.load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; aborting before "
                         "any paid call")
    manifest = verify_freeze()
    smoke = manifest["smoke_episodes"]
    fold_no = int(smoke["fold"])
    ranks = [int(r) for r in smoke["bundle_ranks"]]
    print(f"freeze ok payload={manifest['freeze_payload_sha256'][:12]} "
          f"sdk={manifest['anthropic_sdk_version']} "
          f"smoke=f{fold_no} ranks={ranks} max_calls={smoke['n_calls_max']}")

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
    f = next(x for x in folds_all if x["fold"] == fold_no)
    ts = f["test_start_idx"]
    group_ids = sorted(demand)
    smoke_gids = [group_ids[r] for r in ranks]
    all_series = tuple(s for gid in group_ids for s in demand[gid][0])

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

    client = la.make_client()
    ledger = la.CostLedger()
    episodes, week_rows, call_rows = [], [], []
    gen_latencies, expl_latencies = [], []
    integrity_ok = True

    for gid in smoke_gids:
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

        def provider(w, ctx, _gid=gid, _series=series):
            expected = ts - 1 + 7 * w
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
            rec = la.call_generation(client, ledger, la.GEN_SYSTEM, prompt)
            save_raw(args.tag, f"f{fold_no}_{_gid}_w{w}_gen", rec)
            if rec.outcome == la.OUTCOME_FATAL_AUTH:
                raise SystemExit(f"fatal auth error: {rec.error}")
            parse = None
            if rec.outcome == la.OUTCOME_OK:
                parse = la.process_generation_response(rec.text, _series,
                                                       base_tau)
            stats = (parse.stats() if parse is not None
                     and not parse.parse_failed else
                     {"candidate_slots_used": 0, "valid_candidates": 0,
                      "invalid_candidates": 0, "duplicate_valid_slots": 0,
                      "truncated_count": 0})
            gen_meta[w] = {"call": record_dict(rec),
                           "parse_failed": parse is None or parse.parse_failed,
                           "failure_reason": (None if parse is None
                                              else parse.failure_reason),
                           **stats}
            gen_latencies.append(rec.latency_s)
            print(f"f{fold_no} {_gid} w{w + 1} gen={rec.outcome} "
                  f"stop={rec.stop_reason} in={rec.input_tokens} "
                  f"out={rec.output_tokens} lat={rec.latency_s:.1f}s "
                  f"valid={stats['valid_candidates']}/"
                  f"{stats['candidate_slots_used']} "
                  f"trunc={stats['truncated_count']}")
            if parse is None or parse.parse_failed:
                return []
            return [c.action for c in parse.valid]

        t0 = time.perf_counter()
        res = run_agent_window(
            state, bp.test, hist, provider, replayer_get(replayer, series),
            series, pr, kg[gid], sim_cfg,
            test_start_idx=ts, fold=fold_no, group_id=gid,
            base_tau=base_tau, policy_name=POLICY_NAME,
        )
        selection_time_s = time.perf_counter() - t0

        for wk in res.weeks:
            expl_prompt = la.build_explanation_prompt(
                gid, wk.week,
                {"tau_eff": wk.selected_tau,
                 "multipliers": wk.selected_multipliers},
                wk.gate)
            rec = la.call_explanation(client, ledger, expl_prompt)
            save_raw(args.tag, f"f{fold_no}_{gid}_w{wk.week - 1}_expl", rec)
            expl_latencies.append(rec.latency_s)
            print(f"f{fold_no} {gid} w{wk.week} expl={rec.outcome} "
                  f"in={rec.input_tokens} out={rec.output_tokens} "
                  f"lat={rec.latency_s:.1f}s")
            week_rows.append({
                "fold": fold_no, "group_id": gid, "week": wk.week,
                "origin_idx": wk.origin_idx, "gate": wk.gate,
                "fallback": wk.fallback,
                "selected_slot": wk.selected_slot,
                "selected_tau": wk.selected_tau,
                "selected_multipliers": list(map(list,
                                                 wk.selected_multipliers)),
                "selected_mean_cost": wk.selected_mean_cost,
                "scaling": wk.scaling,
                "generation": gen_meta[wk.week - 1],
                "explanation": {**record_dict(rec),
                                "text": rec.text if rec.outcome == la.OUTCOME_OK
                                else ""},
            })

        ok = (res.n_days == WINDOW_DAYS and len(res.records) == N_WEEKS
              and abs(sum(r.cost for r in res.records) - res.window_cost)
              <= 1e-9 * max(1.0, abs(res.window_cost)))
        integrity_ok = integrity_ok and ok
        print(f"f{fold_no} {gid} ok={ok} days={res.n_days} "
              f"selection_time={selection_time_s:.1f}s")
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
        call_rows.append({"group_id": gid,
                          "selection_time_s": selection_time_s})

    gen_mean = float(np.mean(gen_latencies)) if gen_latencies else 0.0
    expl_mean = float(np.mean(expl_latencies)) if expl_latencies else 0.0
    n_calls = ledger.calls
    mean_cost = ledger.spent_usd / n_calls if n_calls else 0.0
    projected_960 = mean_cost * 960

    out_path = OUT_DIR / f"agent_llm_{args.tag}.json"
    payload = {
        "task": "ST7-06 LLM-arm smoke run",
        "tag": args.tag,
        "date": datetime.date.today().isoformat(),
        "freeze_payload_sha256": manifest["freeze_payload_sha256"],
        "fold": fold_no, "bundles": smoke_gids, "r_env": R_ENV,
        "policy": POLICY_NAME,
        "ledger": {"calls": ledger.calls,
                   "input_tokens": ledger.input_tokens,
                   "output_tokens": ledger.output_tokens,
                   "spent_usd": ledger.spent_usd},
        "performance_withheld_until_official": True,
        "episodes": episodes,
        "weeks": week_rows,
        "selection_times": call_rows,
    }
    out_path.write_text(json.dumps(payload, sort_keys=True))
    file_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"integrity_all_ok={integrity_ok} episode_rows={len(episodes)}")
    print(f"calls={n_calls} tokens_in={ledger.input_tokens} "
          f"tokens_out={ledger.output_tokens} "
          f"spent_usd={ledger.spent_usd:.4f} mean_call_usd={mean_cost:.5f}")
    print(f"projected_official_960_calls_usd={projected_960:.2f} "
          f"(budget {la.TOTAL_BUDGET_USD:.2f}, smoke incl.)")
    print(f"latency gen_mean={gen_mean:.1f}s expl_mean={expl_mean:.1f}s")
    print(f"wrote {out_path.name} sha256={file_sha[:16]} "
          f"(performance sealed, ST7-04)")


def replayer_get(replayer: AgentSelectionReplayer, series):
    def get(origin_idx: int):
        return replayer.get(origin_idx, series)
    return get


if __name__ == "__main__":
    main()