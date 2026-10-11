"""S5 chart technical-statistics aggregation under decisions ST8-01 / ST8-02.

Scope: deterministic summary of frozen artifacts. No re-simulation,
no re-training, no judgment use.

v5: phase 2 now computes the agent point (ST8-02). After the schema and
key-set checks pass, the agent realized path from agent_llm_official.json
is aggregated with the same formulas (mean episode cost, pooled SKU-day
CSL) and written into the manifest under 'agent' as a separate source.
Per ST8-02 the agent is displayed with its own marker and excluded from
the non-dominated frontier computation, which keeps the ST8-01 9-policy
definition unchanged.

Phase 1 (9 policies, r=5 core population):
  V1 identical 480 (fold, group_id, week) keys across all policies
  V2 r=5 only, kappa/scenario recorded from the frozen file
  V3 file sha256 == stage6_confirmatory.json input_sha256
  V4a recomputed episode-mean CSL vs stage6_report section 5 frozen table
      (tol 5e-4, 3dp rounding; the table's own statistic)
  V4b 480-episode cost totals for a4/a4_sl/l0 vs env_ref episode sums
      (ST6-06 baseline; tol max(1e-6, 1e-9 * total))
  Any failure aborts aggregation (fallback = outline option 1).

Phase 2 (agent, ST8-02):
  schema check, key-set identity vs the 480 evaluation keys, r_env check;
  on pass: aggregate the agent point and record it in the manifest
  (frontier unchanged); on fail: manifest records agent as excluded
  with the reason.

Output: outputs/deck/s5_agg.json
Run: python -m scripts.build_deck_s5_agg
"""

import hashlib
import json
from pathlib import Path

EVAL_PATH = Path("outputs/stage6_eval_core_step6.json")
CONFIRMATORY_PATH = Path("outputs/stage6_confirmatory.json")
ENV_REF_PATH = Path("outputs/stage2_asis_env_ref.json")
AGENT_PATH = Path("outputs/agent_llm_official.json")
OUT_PATH = Path("outputs/deck/s5_agg.json")

R_EVAL = 5
N_EPISODES = 480

DISPLAY_NAMES = {
    "b": "B",
    "a1": "A1",
    "a2": "A2",
    "a3": "A3",
    "a4": "A4",
    "a4_sl": "A4-SL",
    "apto": "A-PTO",
    "l0_rule_ct": "L0",
    "a5": "A5",
}

POLICY_ALIASES = {"l0": "l0_rule_ct", "l0_rule_ct": "l0_rule_ct"}

FROZEN_EPISODE_MEAN_CSL_R5 = {
    "b": 0.655,
    "a1": 0.715,
    "a2": 0.906,
    "a3": 0.376,
    "a4": 0.792,
    "a4_sl": 0.880,
    "apto": 0.843,
    "l0_rule_ct": 0.854,
    "a5": 0.913,
}
CSL_TOL = 5e-4

STAGE2_DISPLAY_TOTALS_R5 = {"a4": 492166.0, "a4_sl": 572623.0, "l0_rule_ct": 615583.0}
V4B_POLICIES = ("a4", "a4_sl", "l0_rule_ct")

REFERENCE_POLICIES = {"a5"}

AGENT_REQUIRED_FIELDS = {"fold", "group_id", "week", "cost", "csl", "n_sku", "n_days"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("VERIFICATION FAILED: " + msg + " -- aggregation aborted (fallback = option 1)")


def normalize_policy(name):
    if name in DISPLAY_NAMES:
        return name
    if name in POLICY_ALIASES:
        return POLICY_ALIASES[name]
    return None


def aggregate(rows):
    total_cost = sum(e["cost"] for e in rows)
    sku_days = sum(e["n_sku"] * e["n_days"] for e in rows)
    satisfied = sum(e["csl"] * e["n_sku"] * e["n_days"] for e in rows)
    return {
        "mean_episode_cost": total_cost / len(rows),
        "total_cost_480": total_cost,
        "pooled_sku_day_csl": satisfied / sku_days,
        "episode_mean_csl": sum(e["csl"] for e in rows) / len(rows),
        "sku_days": sku_days,
        "n_episodes": len(rows),
    }


def main():
    eval_sha = sha256_of(EVAL_PATH)
    confirmatory = json.loads(CONFIRMATORY_PATH.read_text())
    frozen_sha = confirmatory["input_sha256"]
    if eval_sha != frozen_sha:
        fail(f"V3 sha mismatch: computed {eval_sha} vs frozen {frozen_sha}")
    print(f"V3 pass: sha256 {eval_sha[:16]}... matches confirmatory input_sha256")

    data = json.loads(EVAL_PATH.read_text())
    episodes = [e for e in data["episodes"] if e["r"] == R_EVAL]

    by_policy = {}
    for e in episodes:
        by_policy.setdefault(e["policy"], []).append(e)

    policies = sorted(by_policy)
    expected = sorted(FROZEN_EPISODE_MEAN_CSL_R5)
    if policies != expected:
        fail(f"V1 policy set mismatch: {policies} vs expected {expected}")

    key_sets = {
        p: {(e["fold"], e["group_id"], e["week"]) for e in rows}
        for p, rows in by_policy.items()
    }
    ref_keys = key_sets[policies[0]]
    if len(ref_keys) != N_EPISODES:
        fail(f"V1 key count {len(ref_keys)} != {N_EPISODES} for {policies[0]}")
    for p, ks in key_sets.items():
        if len(by_policy[p]) != N_EPISODES:
            fail(f"V1 episode count {len(by_policy[p])} != {N_EPISODES} for {p}")
        if ks != ref_keys:
            fail(f"V1 key set differs for {p}")
    print(f"V1 pass: {len(policies)} policies share identical {N_EPISODES} (fold, group_id, week) keys")

    kappa = data.get("kappa")
    scenario = data.get("scenario")
    print(f"V2 recorded from frozen file: r={R_EVAL}, kappa={kappa}, scenario={scenario}")

    agg = {}
    for p, rows in by_policy.items():
        agg[p] = {"display_name": DISPLAY_NAMES[p], **aggregate(rows)}

    for p, ref in FROZEN_EPISODE_MEAN_CSL_R5.items():
        got = agg[p]["episode_mean_csl"]
        if abs(got - ref) > CSL_TOL:
            fail(
                f"V4a episode-mean CSL mismatch for {p}: recomputed {got:.6f} vs frozen {ref} "
                f"(pooled for diagnosis: {agg[p]['pooled_sku_day_csl']:.6f})"
            )
    print("V4a pass: episode-mean CSL matches stage6_report section 5 table within 5e-4 for all 9 policies")

    env_ref_sha = sha256_of(ENV_REF_PATH)
    env_ref = json.loads(ENV_REF_PATH.read_text())
    ref_eps = env_ref["episodes"]
    row0 = ref_eps[0]
    required = {"policy", "r", "cost"}
    missing = required - set(row0.keys())
    if missing:
        fail(f"V4b env_ref episode schema missing {sorted(missing)}; actual keys {sorted(row0.keys())}")
    env_totals = {}
    for e in ref_eps:
        if e["r"] != R_EVAL:
            continue
        p = normalize_policy(e["policy"])
        if p is None:
            fail(f"V4b env_ref unknown policy name '{e['policy']}'")
        env_totals[p] = env_totals.get(p, 0.0) + e["cost"]
    for p in V4B_POLICIES:
        if p not in env_totals:
            fail(f"V4b env_ref has no r={R_EVAL} episodes for {p}; found {sorted(env_totals)}")
        got = agg[p]["total_cost_480"]
        ref = env_totals[p]
        tol = max(1e-6, 1e-9 * abs(ref))
        if abs(got - ref) > tol:
            fail(f"V4b cost total mismatch for {p}: recomputed {got:.6f} vs env_ref {ref:.6f} (tol {tol:.2e})")
    print("V4b pass: a4/a4_sl/l0 480-episode cost totals match env_ref sums (ST6-06 baseline)")

    drift_vs_display = {
        p: agg[p]["total_cost_480"] - STAGE2_DISPLAY_TOTALS_R5[p] for p in V4B_POLICIES
    }
    print("informational drift vs original Stage 2 display totals (ST6-06 env drift, documented):")
    for p, dv in drift_vs_display.items():
        print(f"  {DISPLAY_NAMES[p]:6s} {dv:+.2f}")

    reg = data.get("regression_vs_stage2")

    operational = [p for p in policies if p not in REFERENCE_POLICIES]
    frontier = []
    for p in operational:
        cp, sp = agg[p]["mean_episode_cost"], agg[p]["pooled_sku_day_csl"]
        dominated = any(
            (agg[q]["mean_episode_cost"] <= cp and agg[q]["pooled_sku_day_csl"] >= sp)
            and (agg[q]["mean_episode_cost"] < cp or agg[q]["pooled_sku_day_csl"] > sp)
            for q in operational
            if q != p
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: agg[p]["mean_episode_cost"])
    print(f"frontier (non-dominated operational, a5 excluded): {frontier}")

    print()
    print("phase 2: agent point (ST8-02)")
    agent_sha = sha256_of(AGENT_PATH)
    agent = json.loads(AGENT_PATH.read_text())
    a_eps = agent["episodes"]
    agent_entry = None
    agent_exclusion_reason = None
    missing = AGENT_REQUIRED_FIELDS - set(a_eps[0].keys())
    if missing:
        agent_exclusion_reason = f"schema missing fields {sorted(missing)}"
    elif len(a_eps) != N_EPISODES:
        agent_exclusion_reason = f"episode count {len(a_eps)} != {N_EPISODES}"
    elif agent.get("r_env") != R_EVAL:
        agent_exclusion_reason = f"r_env {agent.get('r_env')} != {R_EVAL}"
    else:
        a_keys = {(e["fold"], e["group_id"], e["week"]) for e in a_eps}
        if a_keys != ref_keys:
            agent_exclusion_reason = "key set differs from evaluation 480 keys"
    if agent_exclusion_reason is None:
        agent_entry = {
            "display_name": "Agent",
            "policy_label": agent.get("policy"),
            **aggregate(a_eps),
            "source_file": str(AGENT_PATH),
            "source_sha256": agent_sha,
            "decision": "ST8-02",
            "marker_note": (
                "realized path of the LLM-interfaced decision-support pipeline; "
                "single realization, contract-level reproducibility (model nondeterminism); "
                "displayed with its own marker, excluded from the frontier computation"
            ),
        }
        print(
            f"  Agent  cost {agent_entry['mean_episode_cost']:10.2f}  "
            f"pooled {agent_entry['pooled_sku_day_csl']:.4f}  ep_mean {agent_entry['episode_mean_csl']:.4f}"
        )
    else:
        print(f"  agent excluded: {agent_exclusion_reason}")

    manifest = {
        "decision": "ST8-01 + ST8-02",
        "purpose": "S5 deck chart technical statistics -- not used for any judgment",
        "input_file": str(EVAL_PATH),
        "input_sha256": eval_sha,
        "population": {
            "r": R_EVAL,
            "kappa": kappa,
            "scenario": scenario,
            "n_episodes_per_policy": N_EPISODES,
            "key": "(fold, group_id, week)",
        },
        "aggregation": {
            "x": "mean_episode_cost = sum(cost) / 480 (terminal attribution unchanged)",
            "y": "pooled_sku_day_csl = sum(csl * n_sku * n_days) / sum(n_sku * n_days)",
        },
        "display_names": DISPLAY_NAMES,
        "naming_note": "evaluation JSON policy key 'l0_rule_ct' corresponds to the 'l0' column of stage6_report section 5",
        "statistic_note": (
            "stage6_report section 5 table statistic = unweighted episode-mean CSL "
            "(identified empirically; a2 episode mean 0.905573 -> 0.906, pooled 0.905284 -> 0.905). "
            "Chart Y axis = pooled SKU-day CSL per ST8-01; V4a cross-checks the table's own statistic. "
            "Both statistics stored per policy; they diverge at the 3rd decimal where bundle sizes differ."
        ),
        "v4b_note": (
            "cost cross-check reference = outputs/stage2_asis_env_ref.json per ST6-06 "
            "(numpy environment drift, one flipped l0 order fold5 B03; original Stage 2 display "
            "totals differ accordingly and the gap is recorded below, informational only). "
            "Evaluation file's own regression gate vs env_ref: see regression_vs_stage2."
        ),
        "env_ref_file": str(ENV_REF_PATH),
        "env_ref_sha256": env_ref_sha,
        "drift_vs_stage2_display_totals": drift_vs_display,
        "regression_vs_stage2": reg,
        "verifications": {
            "v1_identical_keys": True,
            "v2_population": {"r": R_EVAL, "kappa": kappa, "scenario": scenario},
            "v3_sha_match": True,
            "v4a_reference_statistic": "episode_mean_csl",
            "v4a_tol": CSL_TOL,
            "v4b_reference": "env_ref episode sums",
            "v4b_tol": "max(1e-6, 1e-9 * total)",
        },
        "policies": agg,
        "frontier_operational": frontier,
        "frontier_note": (
            "frontier is a point-estimate descriptive set; pairwise statistical judgments exist "
            "only for the pre-registered comparisons (H1: A2/A3/A4, H1-b: A-PTO)"
        ),
        "reference_policies": sorted(REFERENCE_POLICIES),
        "agent": agent_entry,
        "agent_exclusion_reason": agent_exclusion_reason,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"wrote {OUT_PATH}")
    for p in policies:
        marker = " (reference)" if p in REFERENCE_POLICIES else ""
        print(
            f"  {DISPLAY_NAMES[p]:6s} cost {agg[p]['mean_episode_cost']:10.2f}  "
            f"pooled {agg[p]['pooled_sku_day_csl']:.4f}  ep_mean {agg[p]['episode_mean_csl']:.4f}{marker}"
        )


if __name__ == "__main__":
    main()
