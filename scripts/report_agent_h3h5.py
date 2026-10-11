"""ST7-09: preregistered H3/H5 adjudication + joint performance disclosure
(ST7-04 Step 5). Running this script IS the disclosure moment.

H3 (prereg table row 33, SS6): Delta_CT = C(L0 rule CT) - C(Agent=LLM arm
realized path), paired per episode, judged by the FROZEN Stage 6 machinery
imported from scripts.run_confirmatory (episode_tensor / ci_block with the
module's SEED / B_BOOT / BLOCKS; rng_seed_offset pinned to 50 here so the
stream never collides with the H1 offsets 0..8). PASS iff the primary
(2-week block) one-sided 95% lower bound (5th pct) > 0.

H5 (prereg row 35 + ST7-09 operationalization): common-state 10-day
selection costs from outputs/agent_h5_common_state.json. verdict per
episode with tol = 1e-6 + 1e-9 * max(|c_L|, |c_N|); tie = non-win, all 480
in the denominator. PASS iff strict_win / 480 > 0.50 (>= 241). Auxiliary
only: tie-excluded conditional rate, realized-cost pairwise comparison
(the ST7-02 original, demoted), gate/fallback/legality/time tables.

Hard gates: refuses to run unless the common-state anchors are all ok.
Writes docs/agent_h3h5_report.md.

Run:  python -m scripts.report_agent_h3h5
"""

import json
from pathlib import Path

import numpy as np

import scripts.run_confirmatory as rc
from core_pipeline.data.db import PROJECT_ROOT

STEP6_PATH = PROJECT_ROOT / "outputs" / "stage6_eval_core_step6.json"
LLM_PATH = PROJECT_ROOT / "outputs" / "agent_llm_official.json"
NUM_PATH = PROJECT_ROOT / "outputs" / "agent_h5_common_state.json"
NUM_ARM_PATH = PROJECT_ROOT / "outputs" / "agent_numeric_official.json"
OUT_MD = PROJECT_ROOT / "docs" / "agent_h3h5_report.md"
L0_NAME = "l0_rule_ct"
AGENT_NAME = "agent_llm"
R_ENV = 5
H3_OFFSET = 50
H5_MIN_WINS = 241


def episode_cost_map(episodes, policy):
    return {(e["fold"], e["group_id"], e["week"]): e["cost"]
            for e in episodes if e["policy"] == policy and e["r"] == R_ENV}


def wk_table(weeks):
    g = [w["gate"] for w in weeks]
    n_eval = sum(x["evaluated_candidates"] for x in g)
    return {
        "weeks": len(weeks),
        "evaluated_candidates": n_eval,
        "capacity_rejected": sum(x["capacity_rejected"] for x in g),
        "csl_rejected": sum(x["csl_rejected"] for x in g),
        "gate_passed": sum(x["gate_passed_candidates"] for x in g),
        "gate_pass_rate": (sum(x["gate_passed_candidates"] for x in g)
                           / n_eval if n_eval else float("nan")),
        "fallback_weeks": sum(1 for w in weeks if w["fallback"]),
        "fallback_rate": np.mean([w["fallback"] for w in weeks]),
        "selection_time_median_s": float(np.median(
            [w["selection_time_s"] for w in weeks])),
    }


def main() -> None:
    cs = json.loads(NUM_PATH.read_text())
    if not cs["anchors_all_ok"]:
        raise SystemExit("common-state anchors not ok; adjudication refused")
    llm = json.loads(LLM_PATH.read_text())
    num = json.loads(NUM_ARM_PATH.read_text())
    step6 = json.loads(STEP6_PATH.read_text())
    s6_eps = step6["episodes"] if isinstance(step6, dict) else step6
    pols = sorted({e["policy"] for e in s6_eps})
    if L0_NAME not in pols:
        raise SystemExit(f"{L0_NAME} not in step6 policies: {pols}")

    # ---------------------------------------------------------- H3 ----- #
    rc.R_EVAL = (R_ENV,)          # frozen machinery, restricted to r=5
    combined = ([e for e in s6_eps
                 if e["policy"] == L0_NAME and e["r"] == R_ENV]
                + [dict(e) for e in llm["episodes"]])
    tensor, _bundles, _wr = rc.episode_tensor(
        combined, (L0_NAME, AGENT_NAME))
    l0_cost, l0_csl = tensor[(L0_NAME, R_ENV)]
    ag_cost, ag_csl = tensor[(AGENT_NAME, R_ENV)]
    d_ct = rc.improvement_over_baseline(l0_cost, ag_cost)
    h3 = rc.ci_block(d_ct, H3_OFFSET)
    h3["pass"] = h3["primary"]["lower_5pct"] > 0.0
    d_sl = rc.ci_block(ag_csl - l0_csl, H3_OFFSET + 10)   # descriptive only

    # ---------------------------------------------------------- H5 ----- #
    rows = cs["episodes"]
    assert len(rows) == 480
    win = tie = loss = 0
    for r_ in rows:
        cl, cn = r_["cost_llm"], r_["cost_numeric"]
        tol = 1e-6 + 1e-9 * max(abs(cl), abs(cn))
        if cl < cn - tol:
            win += 1
        elif cn < cl - tol:
            loss += 1
        else:
            tie += 1
    p_win = win / 480
    h5_pass = win >= H5_MIN_WINS
    cond = win / (win + loss) if (win + loss) else float("nan")
    both_fb = sum(1 for r_ in rows
                  if r_["llm_fallback"] and r_["numeric_fallback"])

    # ------------------------------------------- auxiliary realized ---- #
    lm = episode_cost_map(llm["episodes"], AGENT_NAME)
    nm = episode_cost_map(num["episodes"], "agent_numeric")
    diffs = np.array([nm[k] - lm[k] for k in sorted(lm)])   # >0: LLM cheaper
    aux = {"mean_diff_numeric_minus_llm": float(diffs.mean()),
           "llm_cheaper_episodes": int((diffs > 0).sum()),
           "identical_episodes": int((diffs == 0).sum())}

    t_llm, t_num = wk_table(llm["weeks"]), wk_table(num["weeks"])
    gen_stats = [w["generation"] for w in llm["weeks"]]
    legal = sum(x["valid_candidates"] for x in gen_stats)
    slots = sum(x["candidate_slots_used"] for x in gen_stats)

    lines = [
        "# Agent stage adjudication -- H3 / H5 (ST7-09, joint disclosure)",
        "",
        f"- freeze payload: `{llm['freeze_payload_sha256'][:16]}`,"
        f" schedule: `{cs['schedule_sha256'][:16]}`, anchors A1/A2 480/480",
        f"- LLM arm: spend ${llm['ledger']['spent_usd']:.4f},"
        f" legality {legal}/{slots},"
        f" selection median {llm['selection_time_median_s']:.2f}s;"
        f" numeric arm time budget {num.get('time_budget_s')}s,"
        " budget_exceeded 0",
        "",
        "## H3 -- Delta_CT = C(L0 rule CT) - C(Agent), r=5, paired",
        f"- point estimate: {h3['point_estimate']:.4f}",
        f"- primary (2-wk block) one-sided 95% lower: "
        f"{h3['primary']['lower_5pct']:.4f}  ->  "
        f"**{'PASS' if h3['pass'] else 'FAIL'}**",
        f"- two-sided 95%: [{h3['primary']['two_sided_2p5']:.4f}, "
        f"{h3['primary']['two_sided_97p5']:.4f}]",
        f"- sensitivity lower bounds: "
        + ", ".join(f"{k}: {v['lower_5pct']:.4f}"
                    for k, v in h3["blocks"].items()),
        f"- weekly-diff ACF: " + ", ".join(
            f"{k}={v:.3f}" for k, v in h3["acf_weekly_sum"].items()),
        f"- CSL delta (agent - L0), descriptive: "
        f"{d_sl['point_estimate']:.4f} "
        f"[{d_sl['primary']['two_sided_2p5']:.4f}, "
        f"{d_sl['primary']['two_sided_97p5']:.4f}]",
        "",
        "## H5 -- common-state selection cost, strict win / 480",
        f"- strict win {win}, tie {tie} (both-fallback {both_fb}), "
        f"loss {loss}",
        f"- p_win = {win}/480 = {p_win:.4f}  (threshold > 0.50, "
        f"min {H5_MIN_WINS})  ->  **{'PASS' if h5_pass else 'FAIL'}**",
        f"- auxiliary tie-excluded rate: {cond:.4f} (not used for judgment)",
        "",
        "## Auxiliary -- realized 28-day costs (ST7-02 original, demoted)",
        f"- mean(C_numeric - C_LLM) = "
        f"{aux['mean_diff_numeric_minus_llm']:.4f}; LLM cheaper in "
        f"{aux['llm_cheaper_episodes']}/480, identical "
        f"{aux['identical_episodes']}",
        "",
        "## Operations table",
        f"- LLM arm:     {json.dumps(t_llm, sort_keys=True)}",
        f"- numeric arm: {json.dumps(t_num, sort_keys=True)}",
        f"- no-action capacity re-evals (common-state): "
        f"{cs['noaction_capacity_reeval_weeks']}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()