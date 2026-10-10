"""ST6-01 (5) + ST5-03 (8) remainder: LP parity degeneracy diagnosis.

Compares the TRAINING path (CvxpyLayer, core_pipeline/dfl/lp_layer) against
the EVALUATION path (solve_a5) on identical 29-day windows (to_lp_window,
ST5-04 common axis). Implementation-consistency diagnostic only -- never a
policy-performance selection criterion.

Pre-registered sample (ST6-01 (5), fixed before any evaluation results):
  base   24 validation origins (outputs/b_sample_schedule.json, ST5-06 set)
         x 10 bundles x r in {3, 5} x 2 initial states = 960 comparisons;
  states S0 = zero on-hand, zero pipeline;
         S1 = spec section 7 warm-up start rule via engine
              init_warmup_state (trailing 7-day mean demand x2 on-hand,
              x1 per pipeline slot), computed at the origin;
  demand LP input = realized demand days o+1..o+28 from mart (deterministic,
         no model forward);
  synth  tie-inducing synthetic cases (uniform costs, symmetric demand,
         binding capacity), reported separately.

Per case: objective rel diff (pass <= 1e-6), feasibility of both first-day
orders, first-day order diff (L1, Linf), degeneracy flag (objective match
AND Linf > 1e-4 -- multiple-optima diagnosis, not an automatic failure).
Any solver failure is a hard stop (ST6-01 (6)).

Torch + diffcp + Clarabel share this process (the train_b-proven stack);
run this script in its own process, never inside the pytest session.

Run:
  python -m scripts.diagnose_lp_parity
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from core_pipeline.data import db
from core_pipeline.data.mart_reader import (
    cost_arrays, fetch_bundle_capacity, fetch_demand_by_bundle,
    fetch_sku_costs,
)
from core_pipeline.simulator.engine import init_warmup_state, load_sim_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = PROJECT_ROOT / "outputs" / "b_sample_schedule.json"
OUT_PATH = PROJECT_ROOT / "outputs" / "stage6_lp_parity.json"
R_SET = (3, 5)
OBJ_TOL = 1e-6
DEG_TOL = 1e-4
NEG_TOL = 1e-9
CAP_REL = 1e-6
WINDOW = 29
FORECAST_DAYS = 28


def first_day_checks(q0: np.ndarray, w: np.ndarray, k_g: float) -> dict:
    wsum = math.fsum((w * np.clip(q0, 0.0, None)).tolist())
    return {
        "min_order": float(q0.min()),
        "weighted_sum": float(wsum),
        "nonneg_ok": bool(q0.min() >= -NEG_TOL),
        "capacity_ok": bool(wsum <= k_g * (1.0 + CAP_REL)),
    }


def compare_case(layer, solve_a5, to_lp_window, downstream_cost, torch,
                 solver_args, demand28, inv, pipe, h, b, w, k_g) -> dict:
    demand_29, inv0, pipeline0 = to_lp_window(demand28, inv, pipe)
    d_t = torch.from_numpy(np.asarray(demand_29, dtype=np.float64))
    inv_t = torch.from_numpy(np.asarray(inv0, dtype=np.float64))
    pipe_t = torch.from_numpy(np.asarray(pipeline0, dtype=np.float64))
    h_t = torch.from_numpy(np.asarray(h, dtype=np.float64))
    b_t = torch.from_numpy(np.asarray(b, dtype=np.float64))

    t0 = time.perf_counter()
    (q_layer,) = layer(d_t, inv_t, pipe_t, solver_args=solver_args)
    layer_s = time.perf_counter() - t0
    obj_layer = float(downstream_cost(
        q_layer, d_t, inv_t, pipe_t, h_t, b_t).item())
    q_layer0 = q_layer.detach().numpy()[0]

    t0 = time.perf_counter()
    sol = solve_a5(demand_29, inv0, pipeline0, h, b, w, k_g)
    solve_s = time.perf_counter() - t0
    if sol.status != "optimal":
        raise RuntimeError(f"solve_a5 status {sol.status!r} (hard stop)")
    q_solve0 = np.asarray(sol.orders[0], dtype=np.float64)

    rel = abs(obj_layer - sol.objective) / max(abs(sol.objective), 1e-9)
    diff = np.abs(q_layer0 - q_solve0)
    linf = float(diff.max()) if diff.size else 0.0
    obj_pass = bool(rel <= OBJ_TOL)
    return {
        "objective_layer": obj_layer,
        "objective_solve": float(sol.objective),
        "objective_rel_diff": float(rel),
        "objective_pass": obj_pass,
        "first_day_l1": float(diff.sum()),
        "first_day_linf": linf,
        "degenerate": bool(obj_pass and linf > DEG_TOL),
        "layer_first_day": first_day_checks(q_layer0, w, k_g),
        "solve_first_day": first_day_checks(q_solve0, w, k_g),
        "layer_s": float(layer_s),
        "solve_s": float(solve_s),
    }


def synthetic_cases(r: int) -> list[dict]:
    """Deterministic tie-inducing instances: uniform costs, symmetric
    demand, binding capacity (any feasible symmetric split is optimal)."""
    n = 4
    h = np.ones(n)
    b = float(r) * np.ones(n)
    w = np.ones(n)
    spike = np.zeros((FORECAST_DAYS, n))
    spike[2, :] = 10.0
    flat = np.full((FORECAST_DAYS, n), 3.0)
    return [
        {"case_id": f"synth_spike_binding_r{r}", "demand28": spike,
         "h": h, "b": b, "w": w, "k_g": 20.0},
        {"case_id": f"synth_flat_binding_r{r}", "demand28": flat,
         "h": h, "b": b, "w": w, "k_g": 6.0},
    ]


def main() -> None:
    schedule = json.loads(SCHEDULE_PATH.read_text())
    origins = [int(o) for o in schedule["validation"]["origins"]]
    if len(origins) != 24:
        raise RuntimeError(
            f"expected 24 validation origins (ST5-06), got {len(origins)}")

    sim_cfg = load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")
    if sim_cfg.kappa_selected is None:
        raise RuntimeError("kappa_selected missing in configs/simulator.yaml")
    lead = int(sim_cfg.lead_time_core)
    lookback = int(sim_cfg.warmup_init_lookback_days)
    max_idx = max(origins) + FORECAST_DAYS

    db.load_dotenv()
    conn = db.connect(db.load_config())
    demand = fetch_demand_by_bundle(conn, max_idx)
    costs = fetch_sku_costs(conn)
    kg = fetch_bundle_capacity(conn, float(sim_cfg.kappa_selected))
    conn.close()
    group_ids = sorted(demand)
    print(f"origins={len(origins)} [{origins[0]}..{origins[-1]}] "
          f"bundles={len(group_ids)} kappa={sim_cfg.kappa_selected}")

    import torch
    torch.set_default_dtype(torch.float64)
    from core_pipeline.dfl.lp_layer import (
        SOLVER_ARGS, build_bundle_layer, downstream_cost,
    )
    from core_pipeline.dfl.lp_problem import solve_a5
    from core_pipeline.dfl.lp_window import to_lp_window

    records = []
    t_all = time.time()
    for r in R_SET:
        for gid in group_ids:
            series, mat = demand[gid]
            h, b = cost_arrays(series, costs, r)
            w = np.ones(len(series))
            layer = build_bundle_layer(h, b, w, float(kg[gid]), WINDOW, lead)
            t_b = time.time()
            for o in origins:
                demand28 = np.asarray(mat[o:o + FORECAST_DAYS],
                                      dtype=np.float64)
                if demand28.shape[0] != FORECAST_DAYS:
                    raise RuntimeError(
                        f"{gid} origin {o}: demand rows "
                        f"{demand28.shape[0]} != {FORECAST_DAYS}")
                recent = mat[o - lookback:o].mean(axis=0)
                s1 = init_warmup_state(recent, lead, sim_cfg)
                states = {
                    "S0": (np.zeros(len(series)),
                           np.zeros((len(series), lead))),
                    "S1": (np.asarray(s1.inv, dtype=np.float64),
                           np.asarray(s1.pipeline, dtype=np.float64)),
                }
                for sname, (inv, pipe) in states.items():
                    res = compare_case(
                        layer, solve_a5, to_lp_window, downstream_cost,
                        torch, SOLVER_ARGS, demand28, inv, pipe,
                        h, b, w, float(kg[gid]))
                    records.append({
                        "kind": "base", "r": r, "bundle": gid,
                        "origin": o, "state": sname, **res,
                    })
            print(f"r={r} {gid} done ({time.time() - t_b:.0f}s)")

    synth_records = []
    for r in R_SET:
        for case in synthetic_cases(r):
            n = case["h"].shape[0]
            layer = build_bundle_layer(
                case["h"], case["b"], case["w"], case["k_g"], WINDOW, lead)
            for sname, (inv, pipe) in {
                "S0": (np.zeros(n), np.zeros((n, lead))),
                "S1": (np.full(n, 6.0), np.full((n, lead), 3.0)),
            }.items():
                res = compare_case(
                    layer, solve_a5, to_lp_window, downstream_cost,
                    torch, SOLVER_ARGS, case["demand28"], inv, pipe,
                    case["h"], case["b"], case["w"], case["k_g"])
                synth_records.append({
                    "kind": "synthetic_tie", "r": r,
                    "case_id": case["case_id"], "state": sname, **res,
                })

    def summarize(recs):
        if not recs:
            return {}
        rels = [x["objective_rel_diff"] for x in recs]
        linfs = [x["first_day_linf"] for x in recs]
        deg = [x for x in recs if x["degenerate"]]
        feas = all(x["layer_first_day"]["nonneg_ok"]
                   and x["layer_first_day"]["capacity_ok"]
                   and x["solve_first_day"]["nonneg_ok"]
                   and x["solve_first_day"]["capacity_ok"] for x in recs)
        return {
            "n": len(recs),
            "objective_pass": sum(x["objective_pass"] for x in recs),
            "objective_rel_max": max(rels),
            "first_day_linf_max": max(linfs),
            "first_day_linf_p50": float(np.median(linfs)),
            "all_first_days_feasible": feas,
            "n_degenerate": len(deg),
            "degenerate_cases": [
                {k: x[k] for k in x
                 if k in ("r", "bundle", "origin", "state", "case_id",
                          "objective_rel_diff", "first_day_linf",
                          "first_day_l1")}
                for x in deg],
        }

    payload = {
        "task": "ST6-01 (5) LP parity degeneracy diagnosis",
        "sample": {
            "origins": origins,
            "schedule_sha256": schedule["sha256"],
            "bundles": group_ids,
            "r_set": list(R_SET),
            "states": {
                "S0": "zero on-hand, zero pipeline",
                "S1": "engine init_warmup_state at origin "
                      "(spec section 7 start rule)",
            },
            "demand_input": "realized demand o+1..o+28 from mart "
                            "(no model forward)",
            "tolerances": {"objective_rel": OBJ_TOL,
                           "degeneracy_linf": DEG_TOL},
        },
        "summary_base": summarize(records),
        "summary_synthetic": summarize(synth_records),
        "records_base": records,
        "records_synthetic": synth_records,
        "runtime_s": float(time.time() - t_all),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1))
    sb, ss = payload["summary_base"], payload["summary_synthetic"]
    print(f"base: {sb['objective_pass']}/{sb['n']} objective pass "
          f"(rel_max {sb['objective_rel_max']:.2e}), "
          f"linf_max {sb['first_day_linf_max']:.3e}, "
          f"degenerate {sb['n_degenerate']}, "
          f"feasible={sb['all_first_days_feasible']}")
    print(f"synthetic: {ss['objective_pass']}/{ss['n']} objective pass, "
          f"degenerate {ss['n_degenerate']}")
    print(f"written {OUT_PATH} ({payload['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()