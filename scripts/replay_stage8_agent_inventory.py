"""Stage 8 agent inventory replay (ST8-05).

Re-executes the OFFICIAL LLM-arm agent run with candidate generation
replaced by a strict read of the stored raw responses
(outputs/agent_llm_raw/<tag>/f{fold}_{gid}_w{w}_gen.json) -- the same
frozen parser (llm_adapter.process_generation_response, sha-verified
against the freeze manifest) turns the stored text into the same actions,
candidate evaluation / gating / ranking re-run deterministically on the
AgentSelectionReplayer parquets, and run_agent_window (with the ST8-05
days field) executes the identical rollout. NO Anthropic client is
constructed anywhere in this script: zero API calls by construction.
Explanation calls are skipped entirely (recorded as action-irrelevant,
ST7-08).

Dual integrity gate against outputs/agent_llm_official.json:
  episodes: cost / holding / shortage / terminal / csl / fill_rate at
            rel <= 1e-9 (bit-exact share reported), ints and bools exact;
  weeks:    fallback, selected_slot, selected_tau, selected_multipliers,
            selected_mean_cost (NaN==NaN on fallback), candidates_provided,
            budget_exceeded and the gate stats must match
            (selection_time_s excluded -- wall-clock remeasurement).
State checks: per-day inv_start must equal the reconstructed midnight
chain elementwise; inv_eod finite and non-negative; the reconstructed
final state must reproduce the returned terminal valuation at rel 1e-9;
3,360 operating-day records expected on the full population.
Any failure -> inventory results are NOT adopted (ST8-05).

Usage:
  python -m scripts.replay_stage8_agent_inventory --folds 1 --tag agent_probe
  python -m scripts.replay_stage8_agent_inventory --folds 1-12 --tag agent
"""

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import time

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
from core_pipeline.data.db import PROJECT_ROOT, config_sha256
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
OFFICIAL_PATH = PROJECT_ROOT / "outputs" / "agent_llm_official.json"
OFFICIAL_SHA = ("15215d470e3cb04d11c19bab6607819b82360f791d782e5dfad387632"
                "e5e5048")
RAW_DIR = PROJECT_ROOT / "outputs" / "agent_llm_raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
SCENARIO_P = 0.0
R_ENV = 5
MART_LAST_IDX = 1941
POLICY_NAME = "agent_llm"
GATE_REL_TOL = 1e-9
EP_FLOAT_FIELDS = ("cost", "holding_cost", "shortage_cost",
                   "terminal_cost", "csl", "fill_rate")
EP_INT_FIELDS = ("binding_days", "n_days", "n_sku")
WEEK_DAYS = 7


def verify_freeze_offline() -> dict:
    """Parser/schema freeze check without the SDK: no client is ever made,
    so the sdk_version entry is skipped (recorded, not silently dropped)."""
    manifest = json.loads(FREEZE_PATH.read_text())
    checks = {
        "payload_sha": (manifest["freeze_payload_sha256"]
                        == canonical_sha(la.freeze_payload())),
        "adapter_sha": (manifest["adapter_file_sha256"]
                        == hashlib.sha256(
                            ADAPTER_PATH.read_bytes()).hexdigest()),
        "schema_sha": manifest["schema_sha256"] == la.schema_sha(),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"freeze verification failed: {failed}; the "
                         "stored responses cannot be replayed with the "
                         "current parser -- agent DOI withheld (ST8-05)")
    return manifest


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


def replayer_get(replayer: AgentSelectionReplayer, series):
    def get(origin_idx: int):
        return replayer.get(origin_idx, series)
    return get


def rel_ok(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) / max(abs(b), 1.0) <= GATE_REL_TOL


def gate_episode(row: dict, frozen: dict, fails: list, key) -> int:
    exact = 0
    for f in EP_FLOAT_FIELDS:
        a, b = float(row[f]), float(frozen[f])
        if not rel_ok(a, b):
            fails.append({"key": list(key), "field": f,
                          "replay": a, "frozen": b})
        elif a == b:
            exact += 1
    for f in EP_INT_FIELDS:
        if int(row[f]) != int(frozen[f]):
            fails.append({"key": list(key), "field": f,
                          "replay": row[f], "frozen": frozen[f]})
    if bool(row["guardrail_blocked"]) != bool(frozen["guardrail_blocked"]):
        fails.append({"key": list(key), "field": "guardrail_blocked",
                      "replay": row["guardrail_blocked"],
                      "frozen": frozen["guardrail_blocked"]})
    return exact


def gate_week(wk, frozen: dict, fails: list, key) -> None:
    if bool(wk.fallback) != bool(frozen["fallback"]):
        fails.append({"key": list(key), "field": "fallback",
                      "replay": wk.fallback, "frozen": frozen["fallback"]})
    if wk.selected_slot != frozen["selected_slot"]:
        fails.append({"key": list(key), "field": "selected_slot",
                      "replay": wk.selected_slot,
                      "frozen": frozen["selected_slot"]})
    if not rel_ok(float(wk.selected_tau), float(frozen["selected_tau"])):
        fails.append({"key": list(key), "field": "selected_tau",
                      "replay": wk.selected_tau,
                      "frozen": frozen["selected_tau"]})
    if not rel_ok(float(wk.selected_mean_cost),
                  float(frozen["selected_mean_cost"])):
        fails.append({"key": list(key), "field": "selected_mean_cost",
                      "replay": wk.selected_mean_cost,
                      "frozen": frozen["selected_mean_cost"]})
    reps = [list(m) for m in map(list, wk.selected_multipliers)]
    if reps != [list(m) for m in frozen["selected_multipliers"]]:
        fails.append({"key": list(key), "field": "selected_multipliers",
                      "replay": reps,
                      "frozen": frozen["selected_multipliers"]})
    if int(wk.candidates_provided) != int(frozen["candidates_provided"]):
        fails.append({"key": list(key), "field": "candidates_provided",
                      "replay": wk.candidates_provided,
                      "frozen": frozen["candidates_provided"]})
    if bool(wk.budget_exceeded) != bool(frozen["budget_exceeded"]):
        fails.append({"key": list(key), "field": "budget_exceeded",
                      "replay": wk.budget_exceeded,
                      "frozen": frozen["budget_exceeded"]})
    for gk, gv in wk.gate.items():
        fv = frozen["gate"].get(gk)
        if isinstance(gv, bool):
            if bool(gv) != bool(fv):
                fails.append({"key": list(key), "field": f"gate.{gk}",
                              "replay": gv, "frozen": fv})
        elif not rel_ok(float(gv), float(fv)):
            fails.append({"key": list(key), "field": f"gate.{gk}",
                          "replay": gv, "frozen": fv})


def state_checks_run(days, start_state, terminal: float, h: np.ndarray,
                     key, fails: list) -> list[float]:
    inv = start_state.inv.copy()
    pipe = start_state.pipeline.copy()
    intransit = []
    for d in days:
        if not np.array_equal(d.inv_start, inv):
            fails.append({"key": list(key), "field": "inv_start_chain",
                          "t_idx": int(d.t_idx)})
            return intransit
        if not np.all(np.isfinite(d.inv_eod)) or np.any(d.inv_eod < 0):
            fails.append({"key": list(key), "field": "inv_eod_invalid",
                          "t_idx": int(d.t_idx)})
            return intransit
        intransit.append(float(pipe.sum() + d.order.sum()))
        inv = d.inv_eod + pipe[:, 0]
        pipe = np.concatenate([pipe[:, 1:], d.order[:, None]], axis=1)
    term = float(h @ (inv + pipe.sum(axis=1)))
    if not rel_ok(term, terminal):
        fails.append({"key": list(key), "field": "terminal_reconstruction",
                      "replay": term, "frozen": terminal})
    return intransit


def week_extensions(days, intransit: list[float]) -> list[dict]:
    out = []
    n_weeks = (len(days) + WEEK_DAYS - 1) // WEEK_DAYS
    for wk in range(n_weeks):
        chunk = days[wk * WEEK_DAYS: (wk + 1) * WEEK_DAYS]
        it_chunk = intransit[wk * WEEK_DAYS: (wk + 1) * WEEK_DAYS]
        out.append({
            "inv_eod_sum": float(sum(d.inv_eod.sum() for d in chunk)),
            "demand_sum": float(sum(d.demand.sum() for d in chunk)),
            "sales_sum": float(sum(d.sales.sum() for d in chunk)),
            "unmet_sum": float(sum(d.unmet.sum() for d in chunk)),
            "order_sum": float(sum(d.order.sum() for d in chunk)),
            "intransit_eod_sum": float(sum(it_chunk)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1-12")
    ap.add_argument("--tag", default="agent")
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    official_sha = file_sha256(OFFICIAL_PATH)
    if official_sha != OFFICIAL_SHA:
        raise SystemExit(f"official sha mismatch: {official_sha}")
    official = json.loads(OFFICIAL_PATH.read_text())
    raw_tag = official["tag"]
    raw_root = RAW_DIR / raw_tag
    if not raw_root.is_dir():
        raise SystemExit(f"raw response dir missing: {raw_root} -- agent "
                         "DOI withheld (ST8-05)")
    manifest = verify_freeze_offline()
    print(f"freeze ok payload={manifest['freeze_payload_sha256'][:12]} "
          f"(offline: sdk_version check skipped, no client constructed)")
    ep_keys = ("policy", "r", "fold", "group_id", "week")
    official_ep = {tuple(e[k] for k in ep_keys): e
                   for e in official["episodes"]}
    official_wk = {(w["fold"], w["group_id"], w["week"]): w
                   for w in official["weeks"]}

    db.load_dotenv()
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

    episodes = []
    ep_fails: list = []
    wk_fails: list = []
    state_fails: list = []
    ep_checked = ep_exact = wk_checked = 0
    total_day_records = 0
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
            hist = np.concatenate([bp.history, bp.warmup], axis=0)

            def provider(w, ctx, _gid=gid, _series=series, _ts=ts,
                         _fold=fold_no):
                expected = _ts - 1 + 7 * w
                if ctx.origin_idx != expected:
                    raise RuntimeError(
                        f"origin mismatch: ctx {ctx.origin_idx} != "
                        f"{expected}")
                raw_path = raw_root / f"f{_fold}_{_gid}_w{w}_gen.json"
                if not raw_path.exists():
                    raise SystemExit(f"stored response missing: {raw_path} "
                                     "-- agent DOI withheld (ST8-05)")
                rec = json.loads(raw_path.read_text())
                if rec["outcome"] in la.FATAL_OUTCOMES:
                    raise SystemExit(
                        f"stored fatal outcome at {raw_path.name}")
                parse = None
                if rec["outcome"] == la.OUTCOME_OK:
                    parse = la.process_generation_response(
                        rec["text"], _series, base_tau)
                if parse is None or parse.parse_failed:
                    return []
                return [c.action for c in parse.valid]

            t0 = time.perf_counter()
            res = run_agent_window(
                state, bp.test, hist, provider,
                replayer_get(replayer, series),
                series, pr, kg[gid], sim_cfg,
                test_start_idx=ts, fold=fold_no, group_id=gid,
                base_tau=base_tau, policy_name=POLICY_NAME,
            )
            elapsed = time.perf_counter() - t0
            if res.n_days != WINDOW_DAYS or len(res.days) != WINDOW_DAYS:
                raise SystemExit(f"f{fold_no} {gid}: {res.n_days} days, "
                                 f"{len(res.days)} day records")
            total_day_records += len(res.days)

            intransit = state_checks_run(
                list(res.days), state, res.terminal, h,
                (fold_no, gid), state_fails)
            exts = (week_extensions(list(res.days), intransit)
                    if len(intransit) == WINDOW_DAYS else [{}] * N_WEEKS)

            for wk in res.weeks:
                wkey = (fold_no, gid, wk.week)
                base_w = official_wk.get(wkey)
                if base_w is None:
                    wk_fails.append({"key": list(wkey),
                                     "field": "<missing in official>"})
                else:
                    wk_checked += 1
                    gate_week(wk, base_w, wk_fails, wkey)

            for rec_, ext in zip(res.records, exts):
                row = {
                    "policy": POLICY_NAME, "r": R_ENV, "fold": rec_.fold,
                    "group_id": rec_.group_id, "week": rec_.week,
                    "cost": rec_.cost, "holding_cost": rec_.holding_cost,
                    "shortage_cost": rec_.shortage_cost,
                    "terminal_cost": rec_.terminal_cost,
                    "csl": rec_.csl, "fill_rate": rec_.fill_rate,
                    "guardrail_blocked": rec_.guardrail_blocked,
                    "binding_days": rec_.binding_days,
                    "n_days": rec_.n_days, "n_sku": rec_.n_sku,
                    "days_of_inventory": rec_.days_of_inventory,
                    **ext,
                }
                key = (POLICY_NAME, R_ENV, rec_.fold, rec_.group_id,
                       rec_.week)
                base = official_ep.get(key)
                if base is None:
                    ep_fails.append({"key": list(key),
                                     "field": "<missing in official>"})
                else:
                    ep_checked += 1
                    ep_exact += gate_episode(row, base, ep_fails, key)
                episodes.append(row)
            print(f"f{fold_no:02d} {gid} replayed {elapsed:.1f}s "
                  f"(api_calls=0)")

    n_fails = len(ep_fails) + len(wk_fails) + len(state_fails)
    gate_pass = (n_fails == 0 and ep_checked == len(episodes)
                 and wk_checked == len(episodes))
    print(f"integrity gate: episodes {ep_checked}/{len(episodes)} "
          f"(bit-exact floats {ep_exact}/{ep_checked * len(EP_FLOAT_FIELDS)}"
          f"), weeks {wk_checked}, day records {total_day_records}, "
          f"fails ep={len(ep_fails)} wk={len(wk_fails)} "
          f"state={len(state_fails)}, pass {gate_pass}")
    for m in (ep_fails + wk_fails + state_fails)[:10]:
        print("  GATE FAIL", m)

    pooled = {}
    if gate_pass:
        n_days = sum(e["n_days"] for e in episodes)
        inv_sum = sum(e["inv_eod_sum"] for e in episodes)
        dem_sum = sum(e["demand_sum"] for e in episodes)
        sal_sum = sum(e["sales_sum"] for e in episodes)
        it_sum = sum(e["intransit_eod_sum"] for e in episodes)
        pooled[f"{POLICY_NAME}_r{R_ENV}"] = {
            "episodes": len(episodes),
            "pooled_doi": (inv_sum / n_days) / (dem_sum / n_days)
            if dem_sum > 0 else None,
            "pooled_intransit_days": (it_sum / n_days) / (dem_sum / n_days)
            if dem_sum > 0 else None,
            "pooled_fill_rate_qty": sal_sum / dem_sum
            if dem_sum > 0 else None,
            "mean_daily_inv_eod": inv_sum / n_days,
            "mean_daily_intransit_eod": it_sum / n_days,
            "mean_daily_demand": dem_sum / n_days,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"stage8_inventory_diagnostics_{args.tag}.json"
    out_path.write_text(json.dumps({
        "task": "Stage 8 agent inventory diagnostics replay (ST8-05)",
        "date": datetime.date.today().isoformat(),
        "tag": args.tag,
        "policy": POLICY_NAME, "r_env": R_ENV, "folds": fold_list,
        "source": {"path": OFFICIAL_PATH.name, "sha256": official_sha,
                   "raw_tag": raw_tag,
                   "freeze_payload_sha256":
                       manifest["freeze_payload_sha256"]},
        "api_calls": 0,
        "git_commit": git_head(),
        "definitions": {
            "days_of_inventory": "EpisodeRecord.days_of_inventory: mean "
            "daily SKU-sum EOD on-hand / mean daily SKU-sum exogenous "
            "demand",
            "pooled_doi": "sum(inv_eod over days) / sum(exogenous demand "
            "over days), common evaluation window",
            "intransit_eod": "sum(P_t) + sum(q_t) at end of day; "
            "reconstructed via the midnight chain, verified by the "
            "per-day inv_start identity and the terminal valuation",
            "pooled_fill_rate_qty": "sum(sales) / sum(exogenous demand)",
        },
        "gate": {"episodes_checked": ep_checked,
                 "bit_exact_floats": ep_exact,
                 "weeks_checked": wk_checked,
                 "day_records": total_day_records,
                 "fails": ep_fails + wk_fails + state_fails,
                 "rel_tol": GATE_REL_TOL, "pass": gate_pass},
        "wall_s": time.time() - run_t0,
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
        print("gate failed: agent inventory results are NOT adopted "
              "(ST8-05)")


if __name__ == "__main__":
    main()
