"""Stage 5 (ST5-05): measure the per-sample costs of the B training loop
and apply the frozen budget formula.

Separated timings (reviewer-corrected 5F-2): t_train = forward + LP +
backward per (bundle, origin); t_val = no-grad forward + LP + downstream
regret; t_oracle = one solve_a5 + pricing (built once per train/valid
sample, epoch-independent); T_compile = sum of the 10 bundle-layer
compile times for this r. Frozen rule:

    n_pb = clip[0, n_max] floor((T - T_compile - V*t_oracle - E*V*t_val)
                                / (G * (t_oracle + E * t_train)))

with T = 10800 s per r, E = 20, G = 10, V = 240; t_* are pooled medians
over all measured samples (fixed before results, no post-hoc selection).
n_max = min over bundles of eligible training-origin counts
(o in [max_i(first_active_idx)+56, tune_end-28]). n_pb = 0 aborts.

--print-origins emits the pre-fixed common measurement set: 8 origins
evenly spaced over [max_SKU(first_active+56), tune_end-28]; feed them to
scripts.build_b_warmup_states, then run the timing passes.

Run order:
    python -m scripts.measure_b_lp --print-origins
    python -m scripts.build_b_warmup_states --fold 1 --stage tune --r 3 5 \
        --origins <the 8 printed origins>
    python -m scripts.measure_b_lp --stage tune --fold 1 --r 3
    python -m scripts.measure_b_lp --stage tune --fold 1 --r 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core_pipeline.dfl.lp_layer import (
    SOLVER_ARGS,
    build_bundle_layer,
    downstream_cost,
    oracle_cost,
)
from core_pipeline.dfl.lp_window import LP_WINDOW_DAYS, to_lp_window
from core_pipeline.forecast.tsmixer.staging import load_bundle

torch.set_default_dtype(torch.float64)

K_G_COLUMN = "k_g_kappa_1_0"
STAGING_ROOT = Path("outputs/staging/tsmixer")
NOISE_SEED = 20261010
HISTORY_DAYS = 56
N_MEASURE_ORIGINS = 8


def wide_sales(panel, series_ids):
    # Same pivot as scripts/build_b_warmup_states (scripts is not a package).
    sub = panel[panel["series_id"].isin(series_ids)]
    wide = (sub.pivot(index="period_idx", columns="series_id", values="sales")
            .reindex(columns=series_ids).sort_index())
    return wide.to_numpy(dtype=np.float64)


def eligibility(bundle):
    """Per-bundle o_min and the tune-stage origin bounds (ST5-05 (1))."""
    meta = bundle["fold_meta"]
    o_max = int(meta["tune_train_end_idx"]) - 28
    static = bundle["series_static"]
    o_min_by_bundle = (static.groupby("group_id")["first_active_idx"].max()
                       + HISTORY_DAYS).astype(int)
    return o_min_by_bundle, o_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final"], default="tune")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--states-dir", default="outputs/b_warmup_states")
    parser.add_argument("--epochs-cap", type=int, default=20)
    parser.add_argument("--valid-samples", type=int, default=240)
    parser.add_argument("--budget-sec", type=float, default=10800.0)
    parser.add_argument("--print-origins", action="store_true")
    parser.add_argument("--out", default="outputs/b_lp_measure.json")
    args = parser.parse_args()

    bundle = load_bundle(STAGING_ROOT / f"fold_{args.fold:02d}")
    o_min_by_bundle, o_max = eligibility(bundle)

    if args.print_origins:
        o_min_common = int(o_min_by_bundle.max())
        origins = sorted({int(round(v)) for v in
                          np.linspace(o_min_common, o_max, N_MEASURE_ORIGINS)})
        print(json.dumps({"o_min_common": o_min_common, "o_max": o_max,
                          "measure_origins": origins}))
        return

    panel = bundle["panel_train"]
    static = bundle["series_static"]
    bundles_df = bundle["bundles"]
    econ = bundle["economic_params"].set_index("series_id")
    h_col = "h_tune" if args.stage == "tune" else "h"
    b_col = (f"b_tune_r{args.r}" if args.stage == "tune" else f"b_r{args.r}")

    state_dir = Path(args.states_dir) / f"{args.stage}_f{args.fold:02d}_r{args.r}"
    rng = np.random.default_rng(NOISE_SEED)
    n_periods = int(panel["period_idx"].max())

    per_bundle = []
    t_train_all, t_val_all, t_oracle_all = [], [], []
    t_compile_total = 0.0
    for row in bundles_df.itertuples(index=False):
        gid = row.group_id
        k_g = float(getattr(row, K_G_COLUMN))
        npz = np.load(state_dir / f"{gid}.npz", allow_pickle=False)
        sids = [str(s) for s in npz["series_ids"]]
        sales = wide_sales(panel, sids)
        h = econ.loc[sids, h_col].to_numpy(dtype=np.float64)
        b = econ.loc[sids, b_col].to_numpy(dtype=np.float64)
        w = np.ones(len(sids))
        lead = npz["pipeline"].shape[2]

        t0 = time.perf_counter()
        layer = build_bundle_layer(h, b, w, k_g, LP_WINDOW_DAYS, lead)
        t_compile = time.perf_counter() - t0
        t_compile_total += t_compile
        h_t = torch.from_numpy(h)
        b_t = torch.from_numpy(b)

        t_train, t_val, t_or = [], [], []
        for i, o in enumerate(npz["origins"]):
            o = int(o)
            if o + 28 > n_periods:
                continue
            future = sales[o:o + 28]
            mean = np.maximum(future.mean(axis=0), 1.0)
            d_hat28 = np.clip(future + rng.normal(0.0, 0.2 * mean, future.shape),
                              0.0, None)
            d29, inv0, pipe0 = to_lp_window(d_hat28, npz["inv"][i],
                                            npz["pipeline"][i])
            y29, _, _ = to_lp_window(future, npz["inv"][i], npz["pipeline"][i])
            inv_t = torch.from_numpy(inv0)
            pipe_t = torch.from_numpy(pipe0)
            y_t = torch.from_numpy(y29)

            d_var = torch.from_numpy(d29).clone().requires_grad_(True)
            t0 = time.perf_counter()
            (q,) = layer(d_var, inv_t, pipe_t, solver_args=SOLVER_ARGS)
            loss = downstream_cost(q, y_t, inv_t, pipe_t, h_t, b_t)
            loss.backward()
            t_train.append(time.perf_counter() - t0)
            if not torch.isfinite(d_var.grad).all():
                raise RuntimeError(f"non-finite gradient at {gid} o={o}")

            d_fixed = torch.from_numpy(d29)
            t0 = time.perf_counter()
            with torch.no_grad():
                (q_v,) = layer(d_fixed, inv_t, pipe_t, solver_args=SOLVER_ARGS)
                downstream_cost(q_v, y_t, inv_t, pipe_t, h_t, b_t)
            t_val.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            oracle_cost(y29, inv0, pipe0, h, b, w, k_g)
            t_or.append(time.perf_counter() - t0)

        rec = {"group_id": gid, "n_sku": len(sids), "n_samples": len(t_train),
               "sec_compile": round(t_compile, 3),
               "sec_train_median": float(np.median(t_train)),
               "sec_train_max": float(np.max(t_train)),
               "sec_val_median": float(np.median(t_val)),
               "sec_oracle_median": float(np.median(t_or))}
        per_bundle.append(rec)
        t_train_all.extend(t_train)
        t_val_all.extend(t_val)
        t_oracle_all.extend(t_or)
        print(json.dumps(rec))

    t_train_med = float(np.median(t_train_all))
    t_val_med = float(np.median(t_val_all))
    t_oracle_med = float(np.median(t_oracle_all))
    g = len(per_bundle)
    eligible = {gid: max(0, o_max - int(om) + 1)
                for gid, om in o_min_by_bundle.items()}
    n_max = min(eligible.values())

    numer = (args.budget_sec - t_compile_total
             - args.valid_samples * t_oracle_med
             - args.epochs_cap * args.valid_samples * t_val_med)
    denom = g * (t_oracle_med + args.epochs_cap * t_train_med)
    n_pb = int(np.clip(numer // denom if denom > 0 else 0, 0, n_max))

    budget = {"t_train_median_sec": t_train_med,
              "t_val_median_sec": t_val_med,
              "t_oracle_median_sec": t_oracle_med,
              "t_compile_total_sec": round(t_compile_total, 3),
              "epochs_cap": args.epochs_cap,
              "valid_samples_per_epoch": args.valid_samples,
              "budget_sec": args.budget_sec,
              "n_bundles": g,
              "eligible_per_bundle": eligible,
              "n_max": n_max,
              "n_per_bundle_rule_value": n_pb}
    for n in sorted({10, 20, 40, n_pb} - {0}):
        est = (t_compile_total
               + (g * n + args.valid_samples) * t_oracle_med
               + args.epochs_cap * (g * n * t_train_med
                                    + args.valid_samples * t_val_med))
        budget[f"est_hours_n{n}"] = round(est / 3600.0, 2)
    if n_pb == 0:
        budget["verdict"] = "ABORT: budget insufficient (ST5-05 stop rule)"
    print(json.dumps(budget))

    out_path = Path(args.out)
    merged = json.loads(out_path.read_text()) if out_path.exists() else {}
    merged[f"{args.stage}_f{args.fold:02d}_r{args.r}"] = {
        "stage": args.stage, "fold": args.fold, "r": args.r,
        "per_bundle": per_bundle, "budget": budget}
    out_path.write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
