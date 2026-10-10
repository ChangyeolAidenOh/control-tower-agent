"""Stage 5 (ST5-03 (4)): per-origin A4 warm-up state cache for B training.

For each (stage, r, bundle, origin o): initialize with the spec warm-up
start rule (7-day mean over days [o-21, o-15]; I = mean x 2, each
pipeline slot = mean x 1), roll the tuned Track A A4 policy through days
[o-14, o-1] with the NumPy engine, let the final midnight transition fold
in day o's morning arrival, realize day o's demand with realize_day, and
store the decision-time state (inv_eod(o), P_o) that
lp_window.to_lp_window consumes. Deterministic: no randomness anywhere.

Demand source: the staging bundle's panel_train sales (observed sales as
latent demand, core p = 0). Costs follow prepare_stage's stage split
(h/b_r{r} final, h_tune/b_tune_r{r} tune); K_g = bundles.k_g_kappa_1_0
for both stages (ST5-04 (5)). Outputs are git-excluded (ST5-02); this
script is the reproducible source.

Run: python -m scripts.build_b_warmup_states --fold 1 --stage tune \
         --r 3 5 --origins 1200 1400 1437 1500 1577
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core_pipeline.forecast.tsmixer.staging import load_bundle
from core_pipeline.policies.a4_basestock import A4BaseStock, A4Config
from core_pipeline.simulator.engine import (
    SkuParams,
    init_warmup_state,
    load_sim_config,
    realize_day,
    run_window,
)

WARMUP_DAYS = 14
INIT_LOOKBACK_DAYS = 7
MIN_ORIGIN = WARMUP_DAYS + INIT_LOOKBACK_DAYS + 1  # lookback must start at period 1
LEAD_TIME = 2  # core L = 2 (spec section 5.1); override with --lead for stress runs
K_G_COLUMN = "k_g_kappa_1_0"
STAGING_ROOT = Path("outputs/staging/tsmixer")
TUNING_PATH = Path("outputs/stage2_a4_tuning.json")
SIM_CONFIG_PATH = "configs/simulator.yaml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tuned_a4(r: int) -> tuple[A4BaseStock, dict]:
    entry = json.loads(TUNING_PATH.read_text())["selected"][str(r)]
    policy = A4BaseStock(A4Config(tau=float(entry["tau"]),
                                  allocation=entry["allocation_rule"]))
    return policy, {"tau": entry["tau"],
                    "allocation_rule": entry["allocation_rule"],
                    "kappa": entry["kappa"]}


def wide_sales(panel, series_ids: list[str]) -> np.ndarray:
    sub = panel[panel["series_id"].isin(series_ids)]
    wide = (sub.pivot(index="period_idx", columns="series_id", values="sales")
            .reindex(columns=series_ids).sort_index())
    idx = wide.index.to_numpy()
    if idx[0] != 1 or not np.all(np.diff(idx) == 1):
        raise ValueError("panel period_idx is not contiguous from 1")
    if wide.isna().any().any():
        raise ValueError("NaN sales in panel pivot")
    return wide.to_numpy(dtype=np.float64)


def parse_origins(args) -> list[int]:
    if args.origins_file:
        payload = json.loads(Path(args.origins_file).read_text())
        origins = payload["origins"] if isinstance(payload, dict) else payload
    else:
        origins = args.origins
    if not origins:
        raise SystemExit("provide --origins or --origins-file")
    return sorted(int(o) for o in set(origins))


def build_states(sales, origins, policy, params, k_g, sim_cfg, lead):
    n_sku = sales.shape[1]
    inv_out = np.empty((len(origins), n_sku))
    pipe_out = np.empty((len(origins), n_sku, lead))
    for i, o in enumerate(origins):
        lookback = sales[o - MIN_ORIGIN:o - WARMUP_DAYS - 1]      # days [o-21, o-15]
        state = init_warmup_state(lookback.mean(axis=0), lead, sim_cfg)
        warm_demand = sales[o - WARMUP_DAYS - 1:o - 1]            # days [o-14, o-1]
        hist = sales[:o - WARMUP_DAYS - 1]                        # days [1, o-15]
        final_state, _ = run_window(state, warm_demand, hist, policy,
                                    params, k_g, sim_cfg,
                                    t0_idx=o - WARMUP_DAYS)
        _, _, inv_eod, _, _ = realize_day(final_state, sales[o - 1], params)
        inv_out[i] = inv_eod
        pipe_out[i] = final_state.pipeline
    return inv_out, pipe_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    parser.add_argument("--r", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--origins", type=int, nargs="*", default=None)
    parser.add_argument("--origins-file", default=None)
    parser.add_argument("--lead", type=int, default=LEAD_TIME)
    parser.add_argument("--out", default="outputs/b_warmup_states")
    args = parser.parse_args()

    origins = parse_origins(args)
    bundle = load_bundle(STAGING_ROOT / f"fold_{args.fold:02d}")
    panel = bundle["panel_train"]
    static = bundle["series_static"]
    bundles_df = bundle["bundles"]
    econ = bundle["economic_params"].set_index("series_id")
    sim_cfg = load_sim_config(SIM_CONFIG_PATH)

    if args.stage == "tune":
        h_col, b_col_fmt = "h_tune", "b_tune_r{r}"
        if h_col not in econ.columns:
            raise SystemExit("tune stage requires the fold-1 bundle (h_tune missing)")
    else:
        h_col, b_col_fmt = "h", "b_r{r}"

    n_periods = int(panel["period_idx"].max())
    bad = [o for o in origins if o < MIN_ORIGIN or o > n_periods]
    if bad:
        raise SystemExit(f"origins out of range [{MIN_ORIGIN}, {n_periods}]: {bad}")

    for r in args.r:
        policy, a4_meta = load_tuned_a4(r)
        out_dir = Path(args.out) / f"{args.stage}_f{args.fold:02d}_r{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        b_col = b_col_fmt.format(r=r)
        manifest = {
            "stage": args.stage, "fold": args.fold, "r": r,
            "origins": origins, "warmup_days": WARMUP_DAYS,
            "init_lookback_days": INIT_LOOKBACK_DAYS, "lead": args.lead,
            "k_g_column": K_G_COLUMN, "h_col": h_col, "b_col": b_col,
            "a4": a4_meta, "a4_tuning_sha256": sha256_file(TUNING_PATH),
            "fold_meta_sha256": bundle["manifest"]["files"]["fold_meta.json"]["sha256"],
            "encoder_hash": bundle["encoder"]["hash"],
            "files": {},
        }
        for row in bundles_df.itertuples(index=False):
            gid = row.group_id
            k_g = float(getattr(row, K_G_COLUMN))
            sids = static.loc[static["group_id"] == gid, "series_id"].astype(str).tolist()
            sales = wide_sales(panel, sids)
            h = econ.loc[sids, h_col].to_numpy(dtype=np.float64)
            b = econ.loc[sids, b_col].to_numpy(dtype=np.float64)
            if not (np.isfinite(h).all() and np.isfinite(b).all()):
                raise ValueError(f"non-finite costs for {gid}")
            params = SkuParams(series_ids=tuple(sids), h=h, b=b,
                               w=np.ones(len(sids)))
            inv_out, pipe_out = build_states(sales, origins, policy, params,
                                             k_g, sim_cfg, args.lead)
            path = out_dir / f"{gid}.npz"
            np.savez(path, origins=np.asarray(origins, dtype=np.int64),
                     inv=inv_out, pipeline=pipe_out,
                     series_ids=np.asarray(sids))
            manifest["files"][path.name] = sha256_file(path)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{out_dir}: bundles={len(manifest['files'])} "
              f"origins={len(origins)}")


if __name__ == "__main__":
    main()
