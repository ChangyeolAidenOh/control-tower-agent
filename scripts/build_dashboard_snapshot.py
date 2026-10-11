"""Build the dashboard presentation snapshot from frozen canonical outputs.

Contract: docs/stage9_dashboard_contract.md v1.1 (ST9-01..05).
- Verify Tier A anchors by sha256 before extraction; abort on mismatch.
- Copy frozen values only; no new aggregation or re-judgment.
- Numeric/bool/None leaves pass; string leaves require a whitelist entry.
- H3/H5 verdicts: report_agent_h3h5.py serializes no JSON, so the frozen
  values are transcribed constants cross-checked against input anchors
  (freeze payload / schedule sha) at build time (ST9-05).
- Write dashboard/assets/display_snapshot.json and display_manifest.json.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dashboard" / "assets"

WHITELIST_VERSION = "wl-2"
CONTRACT = "docs/stage9_dashboard_contract.md v1.1"

TIER_A = {
    "outputs/stage6_eval_core_step6.json": "1dc0cedafc3f452865fedaa32b8b5053f8338ab9238a0612f99023eb917614ce",
    "outputs/agent_llm_official.json": "15215d470e3cb04d11c19bab6607819b82360f791d782e5dfad387632e5e5048",
    "outputs/stage6_confirmatory.json": "7a685b572e7c1e9379d51c1b12b3cefaf687fda4e41ed87f72f4bd974234b180",
    "outputs/deck/s5_agg.json": "cf204bdaf9793daafea08c5f3953128b1edf2b0bab7594845ac0eb5b9c1c9577",
}

TIER_B = [
    "outputs/reports/stage8_lagging_metrics.json",
    "outputs/reports/stage3_report.json",
    "outputs/reports/stage4_report.json",
    "outputs/agent_numeric_official.json",
    "outputs/agent_h5_common_state.json",
]

FREEZE_PAYLOAD_PREFIX = "98c23b07278c7cb1"
SCHEDULE_PREFIX = "a648dde54b58ed7c"

H3_H5_FROZEN = {
    "source": (
        "ST7-10 frozen adjudication; computed by scripts/report_agent_h3h5.py, "
        "reported in docs/agent_h3h5_report.md (untracked)"
    ),
    "serialization_note": (
        "report_agent_h3h5.py emits a markdown report only; no frozen JSON "
        "serialization exists. Values are transcribed constants; input anchors "
        "(freeze payload, schedule) are cross-checked at build time (ST9-05)."
    ),
    "anchors": {
        "freeze_payload_sha256_prefix": FREEZE_PAYLOAD_PREFIX,
        "schedule_sha256_prefix": SCHEDULE_PREFIX,
    },
    "h3": {
        "definition": (
            "Delta_CT = C(L0 rule CT) - C(Agent), r=5, paired, 480 episodes; "
            "one-sided 95% lower bound via 2-week moving-block bootstrap"
        ),
        "point_estimate": 240.6614,
        "primary_lower_95_one_sided": 205.7306,
        "two_sided_95": [200.2241, 279.7910],
        "sensitivity_lower_bounds": {"L2": 205.7306, "L1": 204.1791, "L4": 207.0767},
        "pass": True,
        "csl_delta_descriptive": {
            "definition": "CSL delta (Agent - L0), descriptive, reported jointly always",
            "point": -0.0473,
            "two_sided_95": [-0.0530, -0.0423],
        },
    },
    "h5": {
        "definition": (
            "common-state 10-day selection cost, strict win / 480, "
            "threshold > 0.50 (min 241 wins); ties never count as wins"
        ),
        "strict_win": 211,
        "tie": 205,
        "tie_both_fallback": 120,
        "loss": 64,
        "p_win": 0.4396,
        "threshold_min_wins": 241,
        "pass": False,
        "tie_excluded_rate_auxiliary": 0.7673,
        "auxiliary_note": "tie-excluded rate 0.7673 is auxiliary, not used for judgment",
    },
}

CASE_SELECTION_RULE = (
    "Sort weeks by (fold, week, group_id) ascending. "
    "C1 = first with fallback == False and gate.all_rejected == False; "
    "C2 = first with fallback == True; "
    "C3 = first with fallback == False and 0 < gate.gate_passed_candidates < "
    "gate.evaluated_candidates and key != C1 key; "
    "C4 = first with fallback == False and gate.gate_passed_candidates == "
    "gate.evaluated_candidates. "
    "Deduplicate by (fold, week, group_id). Mechanical, text-blind rule "
    "(ST9-03, amended ST9-04 after C1/C3 key collision, before deployment)."
)

_DROP = object()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(relpath: str):
    with open(ROOT / relpath, "r", encoding="utf-8") as f:
        return json.load(f)


def allowed(path: str, allow) -> bool:
    return any(path == a or path.startswith(a + ".") for a in allow)


def copy_block(obj, allow=frozenset(), path=""):
    """Copy numeric/bool/None leaves; keep strings only on whitelisted paths."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            c = copy_block(v, allow, p)
            if c is not _DROP:
                out[k] = c
        return out
    if isinstance(obj, list):
        items = [copy_block(v, allow, path) for v in obj]
        return [i for i in items if i is not _DROP]
    if isinstance(obj, str):
        return obj if allowed(path, allow) else _DROP
    return obj


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def pick_ci(entry: dict) -> dict:
    return {
        "point_estimate": entry["point_estimate"],
        "pass": entry["pass"],
        "primary": dict(entry["primary"]),
    }


def build_confirmatory(c: dict) -> dict:
    results = {}
    for r_key, r_val in c["results"].items():
        fam = {}
        for family in ("h1", "h1_sl"):
            fam[family] = {base: pick_ci(e) for base, e in r_val[family].items()}
        fam["h1_b"] = pick_ci(r_val["h1_b"])
        fam["a5_reference"] = copy_block(
            r_val["a5_reference"],
            allow={"a5_reference.judgment"},
            path="a5_reference",
        )
        fam["h1_pass_intersection"] = r_val["h1_pass_intersection"]
        fam["h1_sl_pass_all"] = r_val["h1_sl_pass_all"]
        results[r_key] = fam
    return {
        "input_sha256": c["input_sha256"],
        "h0": copy_block(c["h0"]),
        "sl_delta": c.get("sl_delta"),
        "B": c.get("B"),
        "blocks": copy_block(c.get("blocks", {})),
        "r_interpretation": c.get("r_interpretation", ""),
        "results": results,
        "limits": list(c["limits"]),
    }


def build_dfl(s5: dict, confirmatory: dict) -> dict:
    allow = {
        "policies.a4.display_name", "policies.a4_sl.display_name",
        "policies.l0_rule_ct.display_name", "policies.a1.display_name",
        "policies.a2.display_name", "policies.b.display_name",
        "policies.a3.display_name", "policies.a5.display_name",
        "policies.apto.display_name",
        "display_names", "naming_note", "statistic_note", "frontier_note",
        "frontier_operational", "reference_policies",
    }
    block = copy_block(
        {
            "policies": s5["policies"],
            "display_names": s5.get("display_names", {}),
            "naming_note": s5.get("naming_note", ""),
            "statistic_note": s5.get("statistic_note", ""),
            "frontier_note": s5.get("frontier_note", ""),
            "frontier_operational": s5.get("frontier_operational", []),
            "reference_policies": s5.get("reference_policies", []),
            "population": s5.get("population", {}),
        },
        allow=allow,
    )
    block["confirmatory"] = confirmatory
    return block


def build_evaluation(lag: dict) -> dict:
    allow = {
        "definitions",
        "population.key", "population.agent_note",
        "financial_bridge.delta.note", "financial_bridge.delta.rho_source",
        "leading_indicators.label",
    }
    return copy_block(
        {
            "definitions": lag["definitions"],
            "gates": lag["gates"],
            "population": lag["population"],
            "lagging_metrics": lag["lagging_metrics"],
            "comparisons": lag["comparisons"],
            "financial_bridge": lag["financial_bridge"],
            "leading_indicators": lag["leading_indicators"],
        },
        allow=allow,
    )


WEEK_SCALARS = [
    "fold", "week", "origin_idx", "selected_tau", "selected_slot",
    "candidates_provided", "fallback", "budget_exceeded",
    "selection_time_s", "selected_mean_cost",
]


def extract_case(w: dict) -> dict:
    case = {k: w[k] for k in WEEK_SCALARS if k in w}
    case["group_id"] = w["group_id"]
    case["selected_multipliers"] = list(w.get("selected_multipliers", []))
    case["gate"] = copy_block(w["gate"])
    case["scaling"] = copy_block(w["scaling"])
    gen = {k: v for k, v in w["generation"].items() if k != "call"}
    case["generation"] = copy_block(gen, allow={"failure_reason"}, path="")
    case["explanation"] = {
        "outcome": w["explanation"]["outcome"],
        "text": w["explanation"]["text"],
    }
    return case


def select_cases(weeks: list) -> list:
    order = sorted(weeks, key=lambda w: (w["fold"], w["week"], w["group_id"]))

    def key(w):
        return (w["fold"], w["week"], w["group_id"])

    def first(pred):
        for w in order:
            if pred(w):
                return w
        return None

    c1 = first(lambda w: not w["fallback"] and not w["gate"]["all_rejected"])
    c1_key = key(c1) if c1 is not None else None
    c2 = first(lambda w: w["fallback"])
    c3 = first(
        lambda w: not w["fallback"]
        and 0 < w["gate"]["gate_passed_candidates"] < w["gate"]["evaluated_candidates"]
        and key(w) != c1_key
    )
    c4 = first(
        lambda w: not w["fallback"]
        and w["gate"]["gate_passed_candidates"] == w["gate"]["evaluated_candidates"]
    )
    seen, cases = set(), []
    for label, w in (
        ("C1_normal", c1),
        ("C2_fallback", c2),
        ("C3_partial_gate", c3),
        ("C4_full_pass", c4),
    ):
        if w is None:
            continue
        k = key(w)
        if k in seen:
            continue
        seen.add(k)
        case = extract_case(w)
        case["case_label"] = label
        cases.append(case)
    return cases


def check_h3h5_anchors(agent: dict, numeric: dict, h5_common: dict) -> None:
    checks = [
        ("agent_llm_official.freeze_payload_sha256",
         agent.get("freeze_payload_sha256", ""), FREEZE_PAYLOAD_PREFIX),
        ("agent_h5_common_state.freeze_payload_sha256",
         h5_common.get("freeze_payload_sha256", ""), FREEZE_PAYLOAD_PREFIX),
        ("agent_h5_common_state.schedule_sha256",
         h5_common.get("schedule_sha256", ""), SCHEDULE_PREFIX),
        ("agent_numeric_official.schedule_sha256",
         numeric.get("schedule_sha256", ""), SCHEDULE_PREFIX),
    ]
    for name, actual, prefix in checks:
        if not str(actual).startswith(prefix):
            raise SystemExit(
                f"H3/H5 ANCHOR MISMATCH {name}: expected prefix {prefix}, got {actual}"
            )


def build_approval(agent: dict, numeric: dict, h5_common: dict, s5: dict) -> dict:
    check_h3h5_anchors(agent, numeric, h5_common)
    agent_summary_allow = {
        "display_name", "policy_label", "decision", "marker_note", "source_sha256",
    }
    run_meta = {
        "selection_time_median_s": agent["selection_time_median_s"],
        "outcome_counts": copy_block(agent["outcome_counts"]),
        "ledger": copy_block(agent["ledger"]),
        "r_env": agent["r_env"],
        "n_weeks": len(agent["weeks"]),
    }
    return {
        "agent_summary": copy_block(s5["agent"], allow=agent_summary_allow),
        "agent_exclusion_reason": s5.get("agent_exclusion_reason", ""),
        "run_meta": run_meta,
        "h3_h5": H3_H5_FROZEN,
        "cases": select_cases(agent["weeks"]),
        "case_selection_rule": CASE_SELECTION_RULE,
    }


def build_forecast_summary(s3: dict, s4: dict) -> dict:
    s3_allow = {
        "masks", "notes",
        "lead_time.eval_set", "lead_time.evaluation_mask_id",
        "lead_time.interpretation_note", "lead_time.source",
    }
    s4_allow = {"chronos_caveat", "h2_input_decision"}
    return {
        "stage3": copy_block(
            {
                "daily": s3["daily"],
                "adaptation": s3["adaptation"],
                "horizon": s3["horizon"],
                "lead_time": s3["lead_time"],
                "masks": s3["masks"],
                "notes": s3["notes"],
            },
            allow=s3_allow,
        ),
        "stage4": copy_block(
            {
                "decile9": s4["decile9"],
                "point_main": s4["point_main"],
                "point_cost_aware": s4["point_cost_aware"],
                "h2_citation_primary_full_period": s4["h2_citation_primary_full_period"],
                "a3_crossing_rate_before_by_fold": s4["a3_crossing_rate_before_by_fold"],
                "frozen_anchors": s4["frozen_anchors"],
                "equivalence_check": s4["equivalence_check"],
                "chronos_caveat": s4["chronos_caveat"],
                "h2_input_decision": s4["h2_input_decision"],
            },
            allow=s4_allow,
        ),
    }


def main() -> int:
    sources = {}
    for relpath, expected in TIER_A.items():
        actual = sha256_file(ROOT / relpath)
        if actual != expected:
            print(f"ANCHOR MISMATCH {relpath}")
            print(f"  expected {expected}")
            print(f"  actual   {actual}")
            return 1
        sources[relpath] = actual
    for relpath in TIER_B:
        sources[relpath] = sha256_file(ROOT / relpath)

    s5 = load_json("outputs/deck/s5_agg.json")
    confirmatory_raw = load_json("outputs/stage6_confirmatory.json")
    lag = load_json("outputs/reports/stage8_lagging_metrics.json")
    agent = load_json("outputs/agent_llm_official.json")
    numeric = load_json("outputs/agent_numeric_official.json")
    h5_common = load_json("outputs/agent_h5_common_state.json")
    s3 = load_json("outputs/reports/stage3_report.json")
    s4 = load_json("outputs/reports/stage4_report.json")

    snapshot = {
        "meta": {
            "contract": CONTRACT,
            "whitelist_version": WHITELIST_VERSION,
            "simulation_banner": (
                "All figures are simulation results on the M5 subset; "
                "no measured client performance is shown."
            ),
            "naming": "LLM interface decision-support pipeline (ST7-09)",
        },
        "dfl": build_dfl(s5, build_confirmatory(confirmatory_raw)),
        "evaluation": build_evaluation(lag),
        "approval": build_approval(agent, numeric, h5_common, s5),
        "forecast_summary": build_forecast_summary(s3, s4),
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    snap_path = ASSETS / "display_snapshot.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1, sort_keys=True)
    snap_sha = sha256_file(snap_path)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_dashboard_snapshot.py",
        "git_commit": git_commit(),
        "contract": CONTRACT,
        "whitelist_version": WHITELIST_VERSION,
        "tier_a_verified": True,
        "h3_h5_anchor_check": "passed",
        "sources": sources,
        "case_selection_rule": CASE_SELECTION_RULE,
        "case_texts_review": "passed 2026-10-11 (C1, C2); new cases from ST9-04 require review",
        "pending": ["forecast_extract (9-D)"],
        "snapshot_sha256": snap_sha,
    }
    man_path = ASSETS / "display_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"snapshot {snap_path.relative_to(ROOT)} sha256 {snap_sha}")
    print(f"manifest {man_path.relative_to(ROOT)}")
    print(f"cases {len(snapshot['approval']['cases'])}")
    print("pending: forecast_extract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
