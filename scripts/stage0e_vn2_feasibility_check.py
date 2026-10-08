"""Stage 0-E (parallel): VN2 feasibility check.

VN2 Inventory Planning Challenge (Vandeput / SupChains, 2025), Week 0 files:
Sales (weekly units, wide), In Stock (weekly boolean, wide), Master (static
hierarchy), Initial State (on-hand, in-transit W+1/W+2, costs).

Answers: date range and resolution, hierarchy sizes, in-stock flag semantics
(sales must be zero when not in stock), stockout frequency distribution for
the M5 simulator censoring sensitivity, and bundle candidates for a weekly
H1 replication.

Run:
    python -m scripts.stage0e_vn2_feasibility_check
Outputs:
    outputs/stage0e_vn2_feasibility.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/vn2")
OUT_JSON = Path("outputs/stage0e_vn2_feasibility.json")
KEY = ["Store", "Product"]


def find_file(keyword):
    hits = sorted(p for p in RAW_DIR.glob("*.csv") if keyword.lower() in p.name.lower())
    return hits[0] if hits else None


def quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


def load_wide(path):
    df = pd.read_csv(path).set_index(KEY)
    week_cols = [c for c in df.columns if str(c)[:2] == "20"]
    df = df[week_cols]
    df.columns = pd.to_datetime(df.columns)
    return df.sort_index(axis=1)


def run_lengths(mask_row):
    # lengths of consecutive True runs
    out, n = [], 0
    for v in mask_row:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def main():
    paths = {k: find_file(k) for k in ("Sales", "In Stock", "Master", "Initial State")}
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        print(f"missing files in {RAW_DIR}: {missing}; present: {[p.name for p in RAW_DIR.glob('*.csv')]}")
        sys.exit(1)
    report = {"files": {k: p.name for k, p in paths.items()}}

    sales = load_wide(paths["Sales"])
    instock = load_wide(paths["In Stock"])
    master = pd.read_csv(paths["Master"]).set_index(KEY)
    init = pd.read_csv(paths["Initial State"]).set_index(KEY)

    step = pd.Series(sales.columns).diff().dt.days.dropna()
    report["time"] = {
        "sales_weeks": int(sales.shape[1]),
        "sales_first": str(sales.columns[0].date()),
        "sales_last": str(sales.columns[-1].date()),
        "step_days_unique": sorted(int(v) for v in step.unique()),
        "instock_weeks": int(instock.shape[1]),
        "instock_first": str(instock.columns[0].date()),
        "instock_last": str(instock.columns[-1].date()),
        "instock_weeks_beyond_sales": int((instock.columns > sales.columns[-1]).sum()),
    }

    idx = sales.index.to_frame(index=False)
    report["hierarchy"] = {
        "n_pairs": int(len(sales)),
        "n_stores": int(idx["Store"].nunique()),
        "n_products": int(idx["Product"].nunique()),
        "stores_per_product": quantiles(idx.groupby("Product")["Store"].nunique()),
        "products_per_store": quantiles(idx.groupby("Store")["Product"].nunique()),
        "products_in_ge3_stores": int((idx.groupby("Product")["Store"].nunique() >= 3).sum()),
        "master_nunique": {c: int(master[c].nunique()) for c in master.columns},
        "master_index_matches_sales": bool(master.index.equals(sales.index)),
    }

    # Bundle candidates for a weekly replication: store x Department / DepartmentGroup.
    m = master.loc[sales.index]
    bundles = {}
    for col in ("Department", "DepartmentGroup", "Division"):
        sizes = m.groupby(["Store", col]).size() if col in m.columns else pd.Series(dtype=int)
        bundles[f"store_x_{col}"] = {
            "n_groups": int(len(sizes)),
            "groups_size_ge10": int((sizes >= 10).sum()),
            "groups_size_ge15": int((sizes >= 15).sum()),
            "size": quantiles(sizes),
        }
    report["bundle_candidates"] = bundles

    S = sales.to_numpy(dtype=float)
    K_raw = instock.reindex(index=sales.index, columns=sales.columns).to_numpy()
    K = np.isin(K_raw.astype(str), ["True", "true", "1", "1.0"])
    nan_k = pd.isna(K_raw)
    nz = np.nan_to_num(S) > 0
    ever = nz.any(axis=1)
    first_idx = np.where(ever, nz.argmax(axis=1), S.shape[1])
    active = np.arange(S.shape[1])[None, :] >= first_idx[:, None]
    n_active = active.sum(axis=1)

    zero_share_active = ((np.nan_to_num(S) == 0) & active).sum(axis=1) / np.maximum(n_active, 1)
    mean_weekly_active = np.where(n_active > 0, np.where(active, np.nan_to_num(S), 0).sum(axis=1) / np.maximum(n_active, 1), np.nan)
    report["sales"] = {
        "min": float(np.nanmin(S)),
        "max": float(np.nanmax(S)),
        "nan_share": float(np.isnan(S).mean()),
        "never_sold": int((~ever).sum()),
        "zero_share_all_weeks": float((np.nan_to_num(S) == 0).mean()),
        "active_weeks": quantiles(n_active[ever]),
        "zero_share_active": quantiles(zero_share_active[ever]),
        "mean_weekly_active": quantiles(mean_weekly_active[ever]),
        "share_zero_share_active_lt_0_4": float((zero_share_active[ever] < 0.4).mean()),
    }

    oos = ~K & ~nan_k
    oos_active = oos & active
    per_series_oos = oos_active.sum(axis=1) / np.maximum(n_active, 1)
    in_stock_active = K & active
    zero_given_instock = ((np.nan_to_num(S) == 0) & in_stock_active).sum() / max(in_stock_active.sum(), 1)
    runs = [r for i in range(S.shape[0]) for r in run_lengths(oos_active[i])]
    report["in_stock"] = {
        "flag_nan_share": float(nan_k.mean()),
        "oos_share_all_weeks": float(oos.mean()),
        "oos_share_active_weeks": float(oos_active.sum() / max(active.sum(), 1)),
        "sales_positive_when_oos_share": float((np.nan_to_num(S)[oos] > 0).mean()) if oos.any() else None,
        "zero_sales_share_given_in_stock_active": float(zero_given_instock),
        "per_series_oos_share_active": quantiles(per_series_oos[ever]),
        "share_series_never_oos": float((per_series_oos[ever] == 0).mean()),
        "oos_run_length_weeks": quantiles(runs) if runs else {},
        "oos_runs_total": int(len(runs)),
        "note": "flag is weekly on-shelf availability (True/False); not a stock level",
    }

    report["initial_state"] = {
        "columns": list(init.columns),
        "end_inventory": quantiles(init["End Inventory"]) if "End Inventory" in init.columns else {},
        "in_transit_w1": quantiles(init["In Transit W+1"]) if "In Transit W+1" in init.columns else {},
        "in_transit_w2": quantiles(init["In Transit W+2"]) if "In Transit W+2" in init.columns else {},
        "end_inventory_over_mean_weekly": quantiles(
            init["End Inventory"].to_numpy() / np.maximum(mean_weekly_active, 1e-9)
        ) if "End Inventory" in init.columns else {},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("files:", report["files"])
    print("time:", report["time"])
    print("hierarchy:", {k: v for k, v in report["hierarchy"].items() if k != "master_nunique"})
    print("master nunique:", report["hierarchy"]["master_nunique"])
    print("bundle candidates:")
    for k, v in bundles.items():
        print(f"  {k}: groups {v['n_groups']}, >=10: {v['groups_size_ge10']}, >=15: {v['groups_size_ge15']}, size {v['size']}")
    print("sales:", {k: v for k, v in report["sales"].items() if k in ("zero_share_all_weeks", "never_sold", "share_zero_share_active_lt_0_4")})
    print("sales zero_share_active:", report["sales"]["zero_share_active"])
    print("sales mean_weekly_active:", report["sales"]["mean_weekly_active"])
    ins = report["in_stock"]
    print(f"in-stock: oos all {ins['oos_share_all_weeks']:.3f}, oos active {ins['oos_share_active_weeks']:.3f}, "
          f"sales>0 when oos {ins['sales_positive_when_oos_share']}, zero|in-stock {ins['zero_sales_share_given_in_stock_active']:.3f}")
    print("per-series oos share (active):", ins["per_series_oos_share_active"], "never oos:", round(ins["share_series_never_oos"], 3))
    print("oos run length weeks:", ins["oos_run_length_weeks"], "runs:", ins["oos_runs_total"])
    print("initial state end inventory:", report["initial_state"]["end_inventory"])
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
