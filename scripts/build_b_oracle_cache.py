"""Stage 5 (ST5-03 (1), ST5-02): precompute the oracle term of the regret
surrogate for every scheduled (bundle, origin) sample.

For each sample: rebuild the 29-day decision window from the cached A4
state and the realized future sales, solve_a5 the perfect-information
plan, and price it through downstream_cost (oracle_cost) so both regret
terms share one cost functional. Stored: the objective-value scalar plus
the invalidation key = sha256 over (inv, pipeline, realized path, h, b,
K_g) -- any change to state, demand path or cost setup changes the key
(ST5-02). Order arrays are NOT stored (parity tests regenerate them).

Hard invariant checked per sample: |priced cost - LP objective| rel
<= 1e-6 (the lost-sales split is exact at the LP optimum); violation
aborts. Output JSON files are small and kept in git (ST5-02).

Run: python -m scripts.build_b_oracle_cache --stage tune --fold 1 --r 3 5
"""

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from core_pipeline.dfl.lp_layer import oracle_cost
from core_pipeline.dfl.lp_window import to_lp_window
from core_pipeline.forecast.tsmixer.staging import load_bundle

K_G_COLUMN = "k_g_kappa_1_0"
STAGING_ROOT = Path("outputs/staging/tsmixer")
PARITY_TOL = 1e-6


def wide_sales(panel, series_ids):
    # Same pivot as scripts/build_b_warmup_states (scripts is not a package).
    sub = panel[panel["series_id"].isin(series_ids)]
    wide = (sub.pivot(index="period_idx", columns="series_id", values="sales")
            .reindex(columns=series_ids).sort_index())
    return wide.to_numpy(dtype=np.float64)


def sample_key(inv0, pipe0, future, h, b, k_g) -> str:
    m = hashlib.sha256()
    for arr in (inv0, pipe0, future, h, b):
        m.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    m.update(struct.pack("<d", float(k_g)))
    return m.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--r", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--states-dir", default="outputs/b_warmup_states")
    parser.add_argument("--out-dir", default="outputs/b_oracle")
    args = parser.parse_args()

    bundle = load_bundle(STAGING_ROOT / f"fold_{args.fold:02d}")
    panel = bundle["panel_train"]
    bundles_df = bundle["bundles"]
    econ = bundle["economic_params"].set_index("series_id")
    h_col = "h_tune" if args.stage == "tune" else "h"
    n_periods = int(panel["period_idx"].max())
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for r in args.r:
        b_col = (f"b_tune_r{r}" if args.stage == "tune" else f"b_r{r}")
        state_dir = (Path(args.states_dir)
                     / f"{args.stage}_f{args.fold:02d}_r{r}")
        state_manifest = json.loads((state_dir / "manifest.json").read_text())
        t0 = time.perf_counter()
        parity_max = 0.0
        cache = {"meta": {"stage": args.stage, "fold": args.fold, "r": r,
                          "h_col": h_col, "b_col": b_col,
                          "k_g_column": K_G_COLUMN,
                          "schedule_sha256": state_manifest.get("schedule_sha256"),
                          "state_files_sha256": state_manifest["files"]},
                 "bundles": {}}
        n_samples = 0
        for row in bundles_df.itertuples(index=False):
            gid = row.group_id
            k_g = float(getattr(row, K_G_COLUMN))
            npz = np.load(state_dir / f"{gid}.npz", allow_pickle=False)
            sids = [str(s) for s in npz["series_ids"]]
            sales = wide_sales(panel, sids)
            h = econ.loc[sids, h_col].to_numpy(dtype=np.float64)
            b = econ.loc[sids, b_col].to_numpy(dtype=np.float64)
            w = np.ones(len(sids))
            entry = {}
            for i, o in enumerate(npz["origins"]):
                o = int(o)
                if o + 28 > n_periods:
                    raise SystemExit(f"{gid} o={o}: realized path exceeds panel")
                future = sales[o:o + 28]
                y29, inv0, pipe0 = to_lp_window(future, npz["inv"][i],
                                                npz["pipeline"][i])
                cost, _, lp_obj = oracle_cost(y29, inv0, pipe0, h, b, w, k_g)
                rel = abs(cost - lp_obj) / max(abs(lp_obj), 1e-9)
                parity_max = max(parity_max, rel)
                if rel > PARITY_TOL:
                    raise SystemExit(f"oracle parity violated at {gid} o={o}: "
                                     f"rel {rel:.2e}")
                entry[str(o)] = {
                    "cost": cost,
                    "key": sample_key(npz["inv"][i], npz["pipeline"][i],
                                      future, h, b, k_g)}
            cache["bundles"][gid] = entry
            n_samples += len(entry)
        cache["meta"]["n_samples"] = n_samples
        cache["meta"]["parity_max_rel"] = parity_max
        out_path = out_root / f"{args.stage}_f{args.fold:02d}_r{r}.json"
        out_path.write_text(json.dumps(cache, indent=1))
        print(json.dumps({"out": str(out_path), "n_samples": n_samples,
                          "parity_max_rel": parity_max,
                          "sec": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
