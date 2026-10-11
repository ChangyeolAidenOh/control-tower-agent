"""Stage 8 lagging-metric report v2 (ST8-04..09; logged ST8-07/09).

Deterministic aggregation of frozen artifacts only -- no re-simulation,
no judgment use. Four lagging metrics on the r=5 core population
(480 episodes/policy, 9 Stage 6 policies + the agent realized path):

  total cost       mean episode cost + component means (holding/shortage/
                   terminal), cross-anchored to s5_agg at rel 1e-9 -- this
                   transitively pins the deck's L0-Agent gap (ST8-01/02);
  CSL              pooled SKU-day CSL (primary) and episode-mean CSL
                   (labeled; they diverge at the 3rd decimal, ST8-01);
  service          unmet SKU-day share = 1 - pooled SKU-day CSL (exact,
  shortfall        all policies) -- per simulator_spec section 8 this is
                   NOT labeled a stockout rate (ST8-09 naming); quantity
                   stockout rate = 1 - pooled fill (sum unmet / sum
                   exogenous demand) for the five replay-covered cells,
                   1 - episode-mean fill_rate (labeled "episode-mean,
                   not pooled") elsewhere;
  days of inv.     pooled DOI and in-transit days for the five replay
                   cells (ST8-04/05 gates); other cells recorded as not
                   computed (inventory units absent from the frozen
                   artifact).

Financial bridge inputs (ST8-09): h_i = rho * mean sale price identity
gives sale-price-valued inventory = mean daily holding cost / rho, exact
from the frozen episode holding costs, per policy -- inputs for the
CONDITIONAL financial scenario in docs/business_case.md (cost ratio
gamma and capital rate k are NOT applied here; assumption boundary only).
Units: M5 sale-price USD, 10-bundle subset.

Leading indicators (forecast freshness, selection latency, rollback/
fallback rate) are reported from the frozen official run with the label
"precedence unverified".

Gates (any failure aborts, nothing written):
  G1 step6 sha == confirmatory input_sha256 == pinned
  G2 s5_agg sha pinned; recomputed cost/CSL statistics match s5_agg
     per policy and for the agent at rel 1e-9
  G3 replay diagnostics: gate.pass true, source shas match, expected
     populations (1,920 / 480), sales + unmet == demand identity
  G4 stored pooled blocks match recomputation from episode rows

Output: outputs/reports/stage8_lagging_metrics.json
Run: python -m scripts.report_stage8_lagging
"""

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

STEP6_PATH = Path("outputs/stage6_eval_core_step6.json")
STEP6_SHA = ("1dc0cedafc3f452865fedaa32b8b5053f8338ab9238a0612f99023eb9176"
             "14ce")
CONFIRMATORY_PATH = Path("outputs/stage6_confirmatory.json")
S5_PATH = Path("outputs/deck/s5_agg.json")
S5_SHA = ("cf204bdaf9793daafea08c5f3953128b1edf2b0bab7594845ac0eb5b9c1c957"
          "7")
AGENT_PATH = Path("outputs/agent_llm_official.json")
AGENT_SHA = ("15215d470e3cb04d11c19bab6607819b82360f791d782e5dfad387632e5e"
             "5048")
REPLAY_FULL_PATH = Path(
    "outputs/reports/stage8_inventory_diagnostics_full.json")
REPLAY_AGENT_PATH = Path(
    "outputs/reports/stage8_inventory_diagnostics_agent.json")
OUT_PATH = Path("outputs/reports/stage8_lagging_metrics.json")

R_EVAL = 5
N_EPISODES = 480
REL_TOL = 1e-9
# simulator_spec cost section / mart.cost_meta (Stage 1-D): h_i = RHO * p_i
RHO = 0.401871
REPLAY_POLICIES = ("l0_rule_ct", "a4", "a4_sl", "b")
AGENT_POLICY = "agent_llm"
COMPARISONS = (("l0_rule_ct", "agent_llm"), ("l0_rule_ct", "a4"),
               ("l0_rule_ct", "a4_sl"), ("l0_rule_ct", "b"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str):
    raise SystemExit("GATE FAILED: " + msg + " -- report aborted")


def rel_ok(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(b), 1.0) <= REL_TOL


def agg_cost_csl(rows: list[dict]) -> dict:
    n = len(rows)
    total = sum(e["cost"] for e in rows)
    sku_days = sum(e["n_sku"] * e["n_days"] for e in rows)
    satisfied = sum(e["csl"] * e["n_sku"] * e["n_days"] for e in rows)
    total_days = sum(e["n_days"] for e in rows)
    holding_total = sum(e["holding_cost"] for e in rows)
    return {
        "n_episodes": n,
        "mean_episode_cost": total / n,
        "total_cost_480": total,
        "mean_holding_cost": holding_total / n,
        "mean_shortage_cost": sum(e["shortage_cost"] for e in rows) / n,
        "mean_terminal_cost": sum(e["terminal_cost"] for e in rows) / n,
        "mean_daily_holding_cost": holding_total / total_days,
        "pooled_sku_day_csl": satisfied / sku_days,
        "episode_mean_csl": sum(e["csl"] for e in rows) / n,
        "episode_mean_fill_rate": sum(e["fill_rate"] for e in rows) / n,
        "sku_days": sku_days,
    }


def pooled_from_replay(rows: list[dict]) -> dict:
    n_days = sum(e["n_days"] for e in rows)
    inv = sum(e["inv_eod_sum"] for e in rows)
    dem = sum(e["demand_sum"] for e in rows)
    sal = sum(e["sales_sum"] for e in rows)
    unm = sum(e["unmet_sum"] for e in rows)
    it = sum(e["intransit_eod_sum"] for e in rows)
    if not rel_ok(sal + unm, dem):
        fail(f"sales+unmet != demand identity ({sal + unm} vs {dem})")
    return {
        "episodes": len(rows),
        "pooled_doi": (inv / n_days) / (dem / n_days),
        "pooled_intransit_days": (it / n_days) / (dem / n_days),
        "pooled_fill_rate_qty": sal / dem,
        "mean_daily_inv_eod": inv / n_days,
        "mean_daily_intransit_eod": it / n_days,
        "mean_daily_demand": dem / n_days,
    }


def check_pooled(stored: dict, recomputed: dict, cell: str):
    for k, v in recomputed.items():
        if k == "episodes":
            if stored[k] != v:
                fail(f"{cell} pooled episodes {stored[k]} != {v}")
        elif not rel_ok(float(stored[k]), float(v)):
            fail(f"{cell} pooled {k} mismatch: stored {stored[k]} vs "
                 f"recomputed {v}")


def main() -> None:
    # G1 -- frozen evaluation source
    step6_sha = sha256_of(STEP6_PATH)
    if step6_sha != STEP6_SHA:
        fail(f"G1 step6 sha {step6_sha}")
    confirmatory = json.loads(CONFIRMATORY_PATH.read_text())
    if confirmatory["input_sha256"] != step6_sha:
        fail("G1 confirmatory input_sha256 mismatch")
    print(f"G1 pass: step6 sha {step6_sha[:16]}... == confirmatory anchor")

    data = json.loads(STEP6_PATH.read_text())
    rows_r5 = [e for e in data["episodes"] if e["r"] == R_EVAL]
    by_policy: dict[str, list] = {}
    for e in rows_r5:
        by_policy.setdefault(e["policy"], []).append(e)
    for p, rows in by_policy.items():
        if len(rows) != N_EPISODES:
            fail(f"policy {p} has {len(rows)} episodes != {N_EPISODES}")

    agent_sha = sha256_of(AGENT_PATH)
    if agent_sha != AGENT_SHA:
        fail(f"agent official sha {agent_sha}")
    agent = json.loads(AGENT_PATH.read_text())
    if len(agent["episodes"]) != N_EPISODES or agent["r_env"] != R_EVAL:
        fail("agent official population mismatch")

    metrics = {p: agg_cost_csl(rows) for p, rows in by_policy.items()}
    metrics[AGENT_POLICY] = agg_cost_csl(agent["episodes"])

    # G2 -- cross-anchor to the frozen deck aggregation (ST8-01/02)
    s5_sha = sha256_of(S5_PATH)
    if s5_sha != S5_SHA:
        fail(f"G2 s5_agg sha {s5_sha}")
    s5 = json.loads(S5_PATH.read_text())
    anchor_fields = ("mean_episode_cost", "total_cost_480",
                     "pooled_sku_day_csl", "episode_mean_csl")
    for p, entry in s5["policies"].items():
        for f in anchor_fields:
            if not rel_ok(metrics[p][f], float(entry[f])):
                fail(f"G2 {p}.{f}: {metrics[p][f]} vs s5 {entry[f]}")
    if s5.get("agent") is None:
        fail("G2 s5_agg has no agent entry")
    for f in anchor_fields:
        if not rel_ok(metrics[AGENT_POLICY][f], float(s5["agent"][f])):
            fail(f"G2 agent.{f}: {metrics[AGENT_POLICY][f]} vs "
                 f"s5 {s5['agent'][f]}")
    print("G2 pass: cost/CSL statistics match s5_agg at rel 1e-9 "
          "(9 policies + agent; deck L0-Agent gap transitively pinned)")

    # G3/G4 -- replay diagnostics (ST8-04/05)
    rep_full = json.loads(REPLAY_FULL_PATH.read_text())
    rep_agent = json.loads(REPLAY_AGENT_PATH.read_text())
    if not rep_full["gate"]["pass"] or not rep_agent["gate"]["pass"]:
        fail("G3 replay gate not passed")
    if rep_full["source"]["sha256"] != step6_sha:
        fail("G3 full replay source sha mismatch")
    if rep_agent["source"]["sha256"] != agent_sha:
        fail("G3 agent replay source sha mismatch")
    rep_rows: dict[str, list] = {}
    for e in rep_full["episodes"]:
        if e["r"] == R_EVAL:
            rep_rows.setdefault(e["policy"], []).append(e)
    rep_rows[AGENT_POLICY] = rep_agent["episodes"]
    expected = set(REPLAY_POLICIES) | {AGENT_POLICY}
    if set(rep_rows) != expected:
        fail(f"G3 replay policy set {sorted(rep_rows)} != "
             f"{sorted(expected)}")
    inventory = {}
    for p, rows in rep_rows.items():
        if len(rows) != N_EPISODES:
            fail(f"G3 replay {p} has {len(rows)} episodes")
        pooled = pooled_from_replay(rows)
        stored_key = f"{p}_r{R_EVAL}"
        stored = (rep_full if p != AGENT_POLICY else rep_agent)["pooled"]
        check_pooled(stored[stored_key], pooled, stored_key)
        inventory[p] = pooled
    print(f"G3/G4 pass: replay populations ({len(REPLAY_POLICIES)}x480 + "
          f"agent 480), demand identity, stored pooled == recomputed")

    # assemble the four lagging metrics (ST8-09 naming: spec section 8
    # forbids labeling 1 - CSL a stockout rate)
    lagging = {}
    for p, m in metrics.items():
        inv = inventory.get(p)
        entry = {
            "display_name": (s5["display_names"].get(p, "Agent")
                             if p != AGENT_POLICY else "Agent"),
            "total_cost": {
                "mean_episode_cost": m["mean_episode_cost"],
                "total_cost_480": m["total_cost_480"],
                "mean_holding_cost": m["mean_holding_cost"],
                "mean_shortage_cost": m["mean_shortage_cost"],
                "mean_terminal_cost": m["mean_terminal_cost"],
            },
            "csl": {
                "pooled_sku_day_csl": m["pooled_sku_day_csl"],
                "episode_mean_csl": m["episode_mean_csl"],
                "label": "primary = pooled SKU-day; episode mean kept "
                         "separately (ST8-01 statistic split)",
            },
            "service_shortfall": {
                "unmet_sku_day_share_pooled": (
                    1.0 - m["pooled_sku_day_csl"]),
                "quantity_stockout_rate": (
                    (1.0 - inv["pooled_fill_rate_qty"]) if inv
                    else (1.0 - m["episode_mean_fill_rate"])),
                "quantity_label": ("pooled (sum unmet / sum exogenous "
                                   "demand), replay-covered" if inv else
                                   "episode-mean fill_rate complement -- "
                                   "NOT pooled (demand weights not "
                                   "serialized in the frozen artifact)"),
                "naming_note": "unmet SKU-day share = 1 - pooled SKU-day "
                               "CSL; per simulator_spec section 8 this is "
                               "NOT a stockout-rate conversion -- the "
                               "quantity statistic is the stockout rate "
                               "(ST8-09)",
            },
            "days_of_inventory": ({
                "pooled_doi_onhand": inv["pooled_doi"],
                "pooled_intransit_days": inv["pooled_intransit_days"],
                "pooled_total_position_days": (
                    inv["pooled_doi"] + inv["pooled_intransit_days"]),
                "mean_daily_inv_eod": inv["mean_daily_inv_eod"],
                "mean_daily_intransit_eod": inv["mean_daily_intransit_eod"],
                "mean_daily_demand": inv["mean_daily_demand"],
            } if inv else {
                "status": "not computed",
                "reason": "inventory units absent from the frozen episode "
                          "artifact; replay not performed for this cell "
                          "(ST8-04 scope)",
            }),
        }
        lagging[p] = entry

    # financial bridge inputs (ST8-09): sale-price-valued on-hand inventory
    # from the rho identity; gamma / k_annual NOT applied here
    bridge = {}
    for p, m in metrics.items():
        bridge[p] = {
            "mean_daily_holding_cost": m["mean_daily_holding_cost"],
            "sale_price_valued_onhand_usd": (
                m["mean_daily_holding_cost"] / RHO),
        }
    l0b, agb = bridge["l0_rule_ct"], bridge[AGENT_POLICY]
    bridge_delta = {
        "l0_minus_agent_sale_price_valued_usd": (
            l0b["sale_price_valued_onhand_usd"]
            - agb["sale_price_valued_onhand_usd"]),
        "note": "conditional-scenario input ONLY: delta WC(gamma) = gamma "
                "* this delta, annual capital effect = delta WC * "
                "k_annual -- gamma and k_annual are explicit assumptions "
                "in docs/business_case.md, never applied here. One-off "
                "working-capital release and annual flows are never "
                "summed. Units: M5 sale-price USD, 10-bundle subset; "
                "identity: daily holding cost = rho * sum_i p_i * I_i "
                "(h_i = rho * p_i, EOD on-hand basis, in-transit "
                "excluded).",
        "rho": RHO,
        "rho_source": "simulator_spec cost section / mart.cost_meta "
                      "(Stage 1-D)",
    }

    # pairwise descriptive comparisons (%p and relative % side by side)
    comparisons = {}
    for base, other in COMPARISONS:
        mb, mo = metrics[base], metrics[other]
        ib, io = inventory.get(base), inventory.get(other)
        c = {
            "base": base, "other": other,
            "cost_diff_mean_episode": (mb["mean_episode_cost"]
                                       - mo["mean_episode_cost"]),
            "cost_diff_rel_pct": 100.0 * (mb["mean_episode_cost"]
                                          - mo["mean_episode_cost"])
            / mb["mean_episode_cost"],
            "csl_diff_pp": 100.0 * (mo["pooled_sku_day_csl"]
                                    - mb["pooled_sku_day_csl"]),
            "csl_diff_rel_pct": 100.0 * (mo["pooled_sku_day_csl"]
                                         - mb["pooled_sku_day_csl"])
            / mb["pooled_sku_day_csl"],
            "unmet_share_diff_pp": 100.0 * (
                (1 - mo["pooled_sku_day_csl"])
                - (1 - mb["pooled_sku_day_csl"])),
        }
        if ib and io:
            c.update({
                "fill_qty_diff_pp": 100.0 * (io["pooled_fill_rate_qty"]
                                             - ib["pooled_fill_rate_qty"]),
                "doi_onhand_diff_days": (io["pooled_doi"]
                                         - ib["pooled_doi"]),
                "doi_onhand_diff_rel_pct": 100.0 * (
                    io["pooled_doi"] - ib["pooled_doi"])
                / ib["pooled_doi"],
                "total_position_diff_days": (
                    (io["pooled_doi"] + io["pooled_intransit_days"])
                    - (ib["pooled_doi"] + ib["pooled_intransit_days"])),
            })
        comparisons[f"{base}_vs_{other}"] = c

    # leading indicators -- precedence unverified (handoff section 5)
    weeks = agent["weeks"]
    sel_times = sorted(w["selection_time_s"] for w in weeks)
    fallback_n = sum(1 for w in weeks if w["fallback"])
    origin_gaps = set()
    by_cell: dict[tuple, list] = {}
    for w in weeks:
        by_cell.setdefault((w["fold"], w["group_id"]), []).append(w)
    for cell_weeks in by_cell.values():
        o = sorted(x["origin_idx"] for x in cell_weeks)
        origin_gaps.update(np.diff(o).tolist())
    if origin_gaps != {7}:
        fail(f"forecast cadence gaps {sorted(origin_gaps)} != {{7}}")
    leading = {
        "label": "LEADING INDICATORS -- precedence UNVERIFIED: operational "
                 "observations from the frozen official run; no claim that "
                 "they predict the lagging outcomes (no lead-lag analysis "
                 "performed)",
        "forecast_freshness": {
            "selection_forecast_cadence_days": 7,
            "verified": "origin_idx gaps == 7 for all 120 "
                        "(fold, bundle) cells",
            "note": "weekly re-forecast at candidate-selection origins; "
                    "design constant confirmed in the artifact",
        },
        "selection_latency_s": {
            "median": float(np.median(sel_times)),
            "p05": float(np.percentile(sel_times, 5)),
            "p95": float(np.percentile(sel_times, 95)),
            "official_recorded_median": agent["selection_time_median_s"],
        },
        "rollback_fallback_rate": {
            "fallback_weeks": fallback_n,
            "total_weeks": len(weeks),
            "rate": fallback_n / len(weeks),
            "note": "fallback = all candidates gate-rejected -> no-action "
                    "base policy for the week (closest operational analog "
                    "of a rollback)",
        },
    }

    manifest = {
        "task": "Stage 8 lagging-metric report v2",
        "decision": "ST8-04..09 execution; logged ST8-07/09",
        "date": datetime.date.today().isoformat(),
        "purpose": "technical statistics for the Stage 8 business case -- "
                   "no judgment use, confirmatory verdicts unchanged",
        "population": {"r": R_EVAL, "n_episodes_per_policy": N_EPISODES,
                       "key": "(fold, group_id, week)",
                       "agent_note": s5["agent"]["marker_note"]},
        "sources": {
            "step6": {"path": str(STEP6_PATH), "sha256": step6_sha},
            "s5_agg": {"path": str(S5_PATH), "sha256": s5_sha},
            "agent_official": {"path": str(AGENT_PATH),
                               "sha256": agent_sha},
            "replay_full": {"path": str(REPLAY_FULL_PATH),
                            "sha256": sha256_of(REPLAY_FULL_PATH)},
            "replay_agent": {"path": str(REPLAY_AGENT_PATH),
                             "sha256": sha256_of(REPLAY_AGENT_PATH)},
        },
        "definitions": {
            "total_cost": "episode cost = holding + shortage + terminal "
                          "(attribution unchanged from Stage 6)",
            "csl_pooled": "sum(csl * n_sku * n_days) / sum(n_sku * n_days)",
            "unmet_sku_day_share": "1 - pooled SKU-day CSL (NOT a "
                                   "stockout-rate conversion, spec "
                                   "section 8 / ST8-09)",
            "stockout_rate_quantity_pooled": "sum(unmet) / sum(exogenous "
                                             "demand) -- replay cells "
                                             "only",
            "doi_onhand": "sum(EOD on-hand) / sum(exogenous demand), "
                          "common evaluation window",
            "intransit_days": "sum(EOD in-transit incl. same-day order) / "
                              "sum(exogenous demand); reported separately "
                              "from on-hand, never summed into DOI "
                              "silently",
            "financial_bridge": "sale_price_valued_onhand_usd = mean "
                                "daily holding cost / rho; see "
                                "financial_bridge.delta.note",
        },
        "gates": {"g1_source_anchor": True, "g2_s5_cross_anchor": True,
                  "g3_replay_integrity": True, "g4_pooled_recompute": True,
                  "rel_tol": REL_TOL},
        "lagging_metrics": lagging,
        "financial_bridge": {"per_policy": bridge, "delta": bridge_delta},
        "comparisons": comparisons,
        "leading_indicators": leading,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False)
                        + "\n")

    order = sorted(metrics, key=lambda p: metrics[p]["mean_episode_cost"])
    print()
    print("policy   cost/ep   pooledCSL  unmet-sh  qty-SO       "
          "DOI(on-hand)  in-transit")
    for p in order:
        m, inv = metrics[p], inventory.get(p)
        qty = (1 - inv["pooled_fill_rate_qty"]) if inv \
            else (1 - m["episode_mean_fill_rate"])
        qlab = "pooled" if inv else "ep-mean"
        doi = f"{inv['pooled_doi']:.3f}" if inv else "--"
        it = f"{inv['pooled_intransit_days']:.3f}" if inv else "--"
        name = lagging[p]["display_name"]
        print(f"{name:8s} {m['mean_episode_cost']:8.2f}  "
              f"{m['pooled_sku_day_csl']:.4f}   "
              f"{1 - m['pooled_sku_day_csl']:.4f}  "
              f"{qty:.4f} {qlab:7s}  {doi:>6s}        {it:>6s}")
    c = comparisons["l0_rule_ct_vs_agent_llm"]
    print()
    print(f"L0 -> Agent: cost {-c['cost_diff_mean_episode']:+.2f}/ep "
          f"({-c['cost_diff_rel_pct']:+.2f}%), "
          f"CSL {c['csl_diff_pp']:+.2f}%p ({c['csl_diff_rel_pct']:+.2f}%), "
          f"fill(qty) {c['fill_qty_diff_pp']:+.2f}%p, "
          f"DOI {c['doi_onhand_diff_days']:+.3f}d "
          f"({c['doi_onhand_diff_rel_pct']:+.1f}%), "
          f"total position {c['total_position_diff_days']:+.3f}d")
    print(f"financial bridge (inputs only, gamma/k not applied): "
          f"sale-price-valued on-hand L0 "
          f"{l0b['sale_price_valued_onhand_usd']:.2f} vs Agent "
          f"{agb['sale_price_valued_onhand_usd']:.2f} USD, delta "
          f"{bridge_delta['l0_minus_agent_sale_price_valued_usd']:.2f}")
    print(f"leading (precedence unverified): latency median "
          f"{leading['selection_latency_s']['median']:.2f}s, fallback "
          f"{fallback_n}/{len(weeks)} = {fallback_n / len(weeks):.3f}, "
          f"forecast cadence 7d verified")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
