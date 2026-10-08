"""Stage 0-E: SKU subset selection (pre-registered rules).

Bundle = (state, dept). A bundle is the unit of the shared DC ordering
capacity/budget constraint: one DC (state) replenishing the same items into
all stores of that state. SKU = item x store.

Rules (fixed before running; changing them is a logged decision):
  R1 series filter: zero share in active window < 0.40,
     active days >= 701, last 28 days not all zero
  R2 item eligibility in a state: the item passes R1 in every store of that state
  R3 bundle quota per state (category balance): CA 4, TX 3, WI 3 bundles;
     FOODS 4, HOUSEHOLD 3, HOBBIES 3 overall (see STATE_CATEGORY_QUOTA)
  R4 dept choice within a category: the dept with the most eligible items
     (tie -> lexicographic); second dept for the second bundle of a category
  R5 items per bundle: round(16 / n_stores_in_state) -> CA 4, TX 5, WI 5
  R6 item draw: eligible items split into volume terciles (mean daily sales
     over passing stores); items drawn round-robin low/mid/high with a fixed seed

Run:
    python -m scripts.stage0e_select_sku_subset
Outputs:
    data/processed/sku_subset_v1.csv
    outputs/stage0e_subset_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/m5")
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_CSV = PROC_DIR / "sku_subset_v1.csv"
OUT_JSON = OUT_DIR / "stage0e_subset_summary.json"

SEED = 20261008
ZERO_SHARE_MAX = 0.40
MIN_ACTIVE_DAYS = 701
TRAILING_ZERO_WINDOW = 28
TARGET_SKU_PER_BUNDLE = 16

STATE_CATEGORY_QUOTA = {
    "CA": {"FOODS": 2, "HOUSEHOLD": 1, "HOBBIES": 1},
    "TX": {"FOODS": 1, "HOUSEHOLD": 1, "HOBBIES": 1},
    "WI": {"FOODS": 1, "HOUSEHOLD": 1, "HOBBIES": 1},
}


def load_sales():
    path = RAW_DIR / "sales_train_evaluation.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    d_cols = [c for c in header if c.startswith("d_")]
    df = pd.read_csv(path, dtype={c: "int16" for c in d_cols})
    return df, d_cols


def series_stats(sales, d_cols):
    mat = sales[d_cols].to_numpy(dtype=np.int16)
    n_series, n_days = mat.shape
    nonzero = mat > 0
    ever = nonzero.any(axis=1)
    first_idx = np.where(ever, nonzero.argmax(axis=1), n_days)
    last_idx = np.where(ever, n_days - 1 - nonzero[:, ::-1].argmax(axis=1), -1)
    active = n_days - first_idx
    zeros_active = np.array([int((mat[i, first_idx[i]:] == 0).sum()) for i in range(n_series)])
    mean_active = np.array([float(mat[i, first_idx[i]:].mean()) if ever[i] else np.nan for i in range(n_series)])
    out = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    out["active_days"] = active
    out["zero_share_active"] = zeros_active / np.maximum(active, 1)
    out["trailing_zero_days"] = n_days - 1 - last_idx
    out["mean_daily_active"] = mean_active
    out["pass_r1"] = (
        (out["zero_share_active"] < ZERO_SHARE_MAX)
        & (out["active_days"] >= MIN_ACTIVE_DAYS)
        & (out["trailing_zero_days"] < TRAILING_ZERO_WINDOW)
    )
    return out


def tercile_labels(values):
    ranks = values.rank(method="first")
    return pd.cut(ranks, bins=3, labels=["low", "mid", "high"]).astype(str)


def draw_round_robin(items_by_tercile, n_items, rng):
    order = ["low", "mid", "high"]
    pools = {k: list(rng.permutation(v)) for k, v in items_by_tercile.items()}
    picked = []
    i = 0
    while len(picked) < n_items and any(pools.values()):
        k = order[i % 3]
        if pools[k]:
            picked.append(pools[k].pop())
        i += 1
    return picked


def main():
    rng = np.random.default_rng(SEED)
    sales, d_cols = load_sales()
    stats = series_stats(sales, d_cols)

    stores_per_state = stats.groupby("state_id")["store_id"].nunique().to_dict()
    # R2: item eligible in a state iff it passes R1 in all stores of that state
    pass_cnt = stats[stats["pass_r1"]].groupby(["state_id", "dept_id", "cat_id", "item_id"]).size().rename("n_pass")
    pass_cnt = pass_cnt.reset_index()
    pass_cnt["n_stores"] = pass_cnt["state_id"].map(stores_per_state)
    eligible = pass_cnt[pass_cnt["n_pass"] == pass_cnt["n_stores"]].copy()

    elig_counts = eligible.groupby(["state_id", "cat_id", "dept_id"]).size().rename("n_eligible").reset_index()

    rows = []
    bundle_log = []
    bundle_no = 0
    for state, quota in STATE_CATEGORY_QUOTA.items():
        n_stores = stores_per_state[state]
        n_items = int(round(TARGET_SKU_PER_BUNDLE / n_stores))
        for cat, n_bundles in quota.items():
            cand = elig_counts[(elig_counts["state_id"] == state) & (elig_counts["cat_id"] == cat)]
            cand = cand.sort_values(["n_eligible", "dept_id"], ascending=[False, True])
            depts = cand["dept_id"].tolist()[:n_bundles]
            if len(depts) < n_bundles:
                print(f"warning: {state}/{cat} has {len(depts)} depts with eligible items, quota {n_bundles}")
            for dept in depts:
                bundle_no += 1
                bundle_id = f"B{bundle_no:02d}_{state}_{dept}"
                pool = eligible[(eligible["state_id"] == state) & (eligible["dept_id"] == dept)]["item_id"]
                pool_stats = (
                    stats[(stats["state_id"] == state) & (stats["item_id"].isin(pool))]
                    .groupby("item_id")["mean_daily_active"].mean()
                )
                terc = tercile_labels(pool_stats)
                by_terc = {k: pool_stats.index[terc == k].tolist() for k in ("low", "mid", "high")}
                picked = draw_round_robin(by_terc, n_items, rng)
                if len(picked) < n_items:
                    print(f"warning: {bundle_id} eligible {len(pool_stats)} < items needed {n_items}")
                sub = stats[(stats["state_id"] == state) & (stats["item_id"].isin(picked))].copy()
                sub["bundle_id"] = bundle_id
                sub["volume_tercile"] = sub["item_id"].map(dict(zip(pool_stats.index, terc)))
                rows.append(sub)
                bundle_log.append(
                    {
                        "bundle_id": bundle_id,
                        "state_id": state,
                        "dept_id": dept,
                        "cat_id": cat,
                        "n_stores": n_stores,
                        "n_eligible_items": int(len(pool_stats)),
                        "n_items": len(picked),
                        "n_sku": int(len(sub)),
                        "items": picked,
                        "mean_daily_active_by_tercile": {
                            k: float(pool_stats[pool_stats.index.isin(v)].mean()) if v else None
                            for k, v in by_terc.items()
                        },
                    }
                )

    subset = pd.concat(rows, ignore_index=True)
    cols = [
        "bundle_id", "state_id", "store_id", "dept_id", "cat_id", "item_id", "id",
        "volume_tercile", "active_days", "zero_share_active", "trailing_zero_days", "mean_daily_active",
    ]
    subset = subset[cols].sort_values(["bundle_id", "item_id", "store_id"]).reset_index(drop=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUT_CSV, index=False)

    summary = {
        "seed": SEED,
        "rules": {
            "zero_share_max": ZERO_SHARE_MAX,
            "min_active_days": MIN_ACTIVE_DAYS,
            "trailing_zero_window": TRAILING_ZERO_WINDOW,
            "target_sku_per_bundle": TARGET_SKU_PER_BUNDLE,
            "state_category_quota": STATE_CATEGORY_QUOTA,
            "item_eligibility": "passes R1 in all stores of the state",
        },
        "stores_per_state": stores_per_state,
        "eligible_items_by_state_cat_dept": elig_counts.to_dict(orient="records"),
        "bundles": bundle_log,
        "totals": {
            "n_bundles": len(bundle_log),
            "n_sku": int(len(subset)),
            "n_items": int(subset["item_id"].nunique()),
            "by_cat": subset["cat_id"].value_counts().to_dict(),
            "by_state": subset["state_id"].value_counts().to_dict(),
            "zero_share_active_q": {
                f"q{int(q*100):02d}": float(subset["zero_share_active"].quantile(q)) for q in (0.05, 0.5, 0.95)
            },
            "mean_daily_active_q": {
                f"q{int(q*100):02d}": float(subset["mean_daily_active"].quantile(q)) for q in (0.05, 0.5, 0.95)
            },
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("eligible items (state, cat, dept, n):")
    for r in elig_counts.itertuples(index=False):
        print(f"  {r.state_id} {r.cat_id} {r.dept_id} {r.n_eligible}")
    print("bundles:")
    for b in bundle_log:
        print(f"  {b['bundle_id']}: eligible {b['n_eligible_items']}, items {b['n_items']}, sku {b['n_sku']}")
    print(f"totals: {summary['totals']}")
    print(f"written: {OUT_CSV}, {OUT_JSON}")


if __name__ == "__main__":
    main()
