"""Stage 0-E: M5 raw structure check.

Reads the Kaggle M5 files from data/raw/m5/, reports structure facts needed
to pre-register the SKU subset rules (intermittency, store coverage,
bundle candidates, zero-sales handling, price/calendar covariate coverage)
and the 12-fold rolling-origin layout. No subset is selected here.

Run:
    python -m scripts.stage0e_m5_structure_check
Outputs:
    outputs/stage0e_m5_structure.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/m5")
OUT_DIR = Path("outputs")
OUT_JSON = OUT_DIR / "stage0e_m5_structure.json"

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

HORIZON = 28
N_FOLDS = 12

# Candidate thresholds to inspect. The final rule is fixed in preregistration.md
# after reading this output, not here.
INTERMITTENCY_CANDIDATES = [0.2, 0.3, 0.4, 0.5]
MIN_ACTIVE_DAYS = 365 + N_FOLDS * HORIZON  # one year of history before fold 1
TRAILING_ZERO_WINDOW = 28


def quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


def check_files():
    missing = [f for f in (SALES_FILE, CALENDAR_FILE, PRICES_FILE) if not (RAW_DIR / f).exists()]
    if missing:
        print(f"missing files in {RAW_DIR}: {missing}")
        sys.exit(1)
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in RAW_DIR.iterdir() if f.is_file()}
    return sizes


def load_sales(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    d_cols = [c for c in header if c.startswith("d_")]
    df = pd.read_csv(path, dtype={c: "int16" for c in d_cols})
    return df, d_cols


def main():
    sizes = check_files()
    report = {"files_mb": sizes}

    calendar = pd.read_csv(RAW_DIR / CALENDAR_FILE, parse_dates=["date"])
    sales, d_cols = load_sales(RAW_DIR / SALES_FILE)
    prices = pd.read_csv(RAW_DIR / PRICES_FILE)

    n_series, n_days = len(sales), len(d_cols)
    mat = sales[d_cols].to_numpy(dtype=np.int16)
    day_index = np.arange(n_days)

    cal_days = calendar.iloc[:n_days]
    report["sales"] = {
        "n_series": int(n_series),
        "n_days": int(n_days),
        "first_day": str(cal_days["date"].iloc[0].date()),
        "last_day": str(cal_days["date"].iloc[-1].date()),
        "min_value": int(mat.min()),
        "max_value": int(mat.max()),
        "hierarchy_nunique": {c: int(sales[c].nunique()) for c in ID_COLS[1:]},
        "stores_per_item": quantiles(sales.groupby("item_id")["store_id"].nunique()),
        "series_per_dept_store": quantiles(sales.groupby(["store_id", "dept_id"]).size()),
    }

    # Activity window: leading zeros before the first sale are treated as
    # "not yet on shelf", not as intermittent demand.
    nonzero = mat > 0
    ever_sold = nonzero.any(axis=1)
    first_idx = np.where(ever_sold, nonzero.argmax(axis=1), n_days)
    last_idx = np.where(ever_sold, n_days - 1 - nonzero[:, ::-1].argmax(axis=1), -1)
    active_len = np.where(ever_sold, n_days - first_idx, 0)
    zeros_active = np.array(
        [int((mat[i, first_idx[i]:] == 0).sum()) if ever_sold[i] else 0 for i in range(n_series)]
    )
    zero_share_active = np.where(active_len > 0, zeros_active / np.maximum(active_len, 1), np.nan)
    trailing_zero_days = n_days - 1 - last_idx
    mean_daily_active = np.array(
        [float(mat[i, first_idx[i]:].mean()) if ever_sold[i] else np.nan for i in range(n_series)]
    )
    # Approximate ADI (average demand interval) on the active window.
    nonzero_count_active = active_len - zeros_active
    adi = np.where(nonzero_count_active > 0, active_len / np.maximum(nonzero_count_active, 1), np.nan)

    series = sales[ID_COLS].copy()
    series["first_sale_idx"] = first_idx
    series["active_days"] = active_len
    series["zero_share_active"] = zero_share_active
    series["adi"] = adi
    series["trailing_zero_days"] = trailing_zero_days
    series["mean_daily_active"] = mean_daily_active

    report["activity"] = {
        "never_sold": int((~ever_sold).sum()),
        "first_sale_idx": quantiles(first_idx[ever_sold]),
        "active_days": quantiles(active_len[ever_sold]),
        "zero_share_active": quantiles(zero_share_active),
        "zero_share_raw_full_window": quantiles((mat == 0).mean(axis=1)),
        "adi_active": quantiles(adi),
        "mean_daily_active": quantiles(mean_daily_active),
        "trailing_zero_days": quantiles(trailing_zero_days[ever_sold]),
        "share_trailing_zero_ge_28d": float((trailing_zero_days[ever_sold] >= TRAILING_ZERO_WINDOW).mean()),
        "share_active_ge_min": float((active_len >= MIN_ACTIVE_DAYS).mean()),
        "min_active_days_rule": int(MIN_ACTIVE_DAYS),
    }
    report["zero_share_by_cat"] = (
        series.groupby("cat_id")["zero_share_active"].median().round(3).to_dict()
    )
    report["zero_share_by_dept"] = (
        series.groupby("dept_id")["zero_share_active"].median().round(3).to_dict()
    )

    # Filter sweep: how many series / items / bundles survive each candidate
    # intermittency threshold. Bundle = store x dept (shared-capacity group proxy).
    sweep = {}
    for thr in INTERMITTENCY_CANDIDATES:
        ok = (
            (series["zero_share_active"] < thr)
            & (series["active_days"] >= MIN_ACTIVE_DAYS)
            & (series["trailing_zero_days"] < TRAILING_ZERO_WINDOW)
        )
        sub = series[ok]
        stores_per_item = sub.groupby("item_id")["store_id"].nunique()
        items_ge3 = stores_per_item[stores_per_item >= 3].index
        sub3 = sub[sub["item_id"].isin(items_ge3)]
        bundle_sizes = sub3.groupby(["store_id", "dept_id"]).size()
        sweep[str(thr)] = {
            "series_pass": int(ok.sum()),
            "items_pass_ge3_stores": int(len(items_ge3)),
            "series_pass_ge3_stores": int(len(sub3)),
            "by_cat": sub3["cat_id"].value_counts().to_dict(),
            "bundles_store_dept_total": int(len(bundle_sizes)),
            "bundles_size_ge10": int((bundle_sizes >= 10).sum()),
            "bundles_size_ge20": int((bundle_sizes >= 20).sum()),
            "bundle_size": quantiles(bundle_sizes),
        }
    report["filter_sweep"] = sweep

    # Calendar covariates.
    report["calendar"] = {
        "rows": int(len(calendar)),
        "date_range": [str(calendar["date"].min().date()), str(calendar["date"].max().date())],
        "days_beyond_sales": int(len(calendar) - n_days),
        "wm_yr_wk_range": [int(calendar["wm_yr_wk"].min()), int(calendar["wm_yr_wk"].max())],
        "event_name_1_nonnull": int(calendar["event_name_1"].notna().sum()),
        "event_name_2_nonnull": int(calendar["event_name_2"].notna().sum()),
        "event_type_1_counts": calendar["event_type_1"].value_counts(dropna=True).to_dict(),
        "snap_days": {c: int(calendar[c].sum()) for c in ("snap_CA", "snap_TX", "snap_WI")},
        "null_counts": {c: int(v) for c, v in calendar.isna().sum().items() if v > 0},
    }

    # Price coverage. sell_prices has one row per (store, item, week) only when
    # the item is on sale; absent weeks mean "not on shelf" (censoring signal).
    d_to_wk = dict(zip(calendar["d"], calendar["wm_yr_wk"]))
    week_list = sorted(cal_days["wm_yr_wk"].unique())
    wk_pos = {w: i for i, w in enumerate(week_list)}
    last_wk_pos = len(week_list) - 1

    price_grp = prices.groupby(["store_id", "item_id"])
    price_stats = price_grp.agg(
        n_price_weeks=("wm_yr_wk", "nunique"),
        first_price_wk=("wm_yr_wk", "min"),
        last_price_wk=("wm_yr_wk", "max"),
        price_mean=("sell_price", "mean"),
        price_min=("sell_price", "min"),
        price_max=("sell_price", "max"),
    ).reset_index()
    series["first_sale_wk"] = [
        d_to_wk[d_cols[i]] if i < n_days else np.nan for i in first_idx
    ]
    series = series.merge(price_stats, on=["store_id", "item_id"], how="left")
    series["expected_weeks_from_first_sale"] = series["first_sale_wk"].map(
        lambda w: last_wk_pos - wk_pos[w] + 1 if w in wk_pos else np.nan
    )
    priced_in_window = (
        prices.merge(series[["store_id", "item_id", "first_sale_wk"]], on=["store_id", "item_id"], how="inner")
        .query("wm_yr_wk >= first_sale_wk and wm_yr_wk <= @week_list[-1]")
        .groupby(["store_id", "item_id"])["wm_yr_wk"]
        .nunique()
        .rename("n_price_weeks_active")
        .reset_index()
    )
    series = series.merge(priced_in_window, on=["store_id", "item_id"], how="left")
    series["price_week_coverage"] = series["n_price_weeks_active"] / series["expected_weeks_from_first_sale"]
    series["price_starts_before_first_sale"] = series["first_price_wk"] <= series["first_sale_wk"]
    series["price_range_ratio"] = series["price_max"] / series["price_min"]

    report["prices"] = {
        "rows": int(len(prices)),
        "store_item_pairs": int(len(price_stats)),
        "pairs_missing_price_entirely": int(series["n_price_weeks"].isna().sum()),
        "n_weeks_in_sales_window": int(len(week_list)),
        "price_week_coverage": quantiles(series["price_week_coverage"]),
        "share_coverage_lt_0_95": float((series["price_week_coverage"] < 0.95).mean()),
        "share_price_starts_before_first_sale": float(series["price_starts_before_first_sale"].mean()),
        "price_range_ratio": quantiles(series["price_range_ratio"]),
        "sell_price_quantiles": quantiles(prices["sell_price"]),
        "null_counts": {c: int(v) for c, v in prices.isna().sum().items() if v > 0},
    }

    # Rolling-origin fold layout: 12 folds x 28-day horizon, last fold ends on last day.
    folds = []
    for k in range(N_FOLDS):
        end = n_days - (N_FOLDS - 1 - k) * HORIZON
        start = end - HORIZON
        folds.append(
            {
                "fold": k + 1,
                "test_start_d": d_cols[start],
                "test_end_d": d_cols[end - 1],
                "test_start_date": str(cal_days["date"].iloc[start].date()),
                "test_end_date": str(cal_days["date"].iloc[end - 1].date()),
                "train_end_d": d_cols[start - 1],
            }
        )
    report["folds"] = {"horizon": HORIZON, "n_folds": N_FOLDS, "layout": folds}

    # Zero-sales handling evidence: share of active-window days that are zero
    # for the eventual MAPE auxiliary metric, by fold window.
    fold1_start = n_days - N_FOLDS * HORIZON
    eval_window = mat[:, fold1_start:]
    report["zero_share_in_eval_window"] = quantiles((eval_window == 0).mean(axis=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 160)
    print(f"sales: {n_series} series x {n_days} days, "
          f"{report['sales']['first_day']} to {report['sales']['last_day']}")
    print(f"hierarchy: {report['sales']['hierarchy_nunique']}")
    print(f"never sold: {report['activity']['never_sold']}, "
          f"trailing zero >= 28d: {report['activity']['share_trailing_zero_ge_28d']:.3f}, "
          f"active >= {MIN_ACTIVE_DAYS}d: {report['activity']['share_active_ge_min']:.3f}")
    print("zero_share_active quantiles:", report["activity"]["zero_share_active"])
    print("zero_share median by dept:", report["zero_share_by_dept"])
    print("filter sweep (threshold -> series_pass_ge3_stores, bundles>=10, bundles>=20):")
    for thr, r in sweep.items():
        print(f"  {thr}: {r['series_pass_ge3_stores']}, {r['bundles_size_ge10']}, {r['bundles_size_ge20']}, by_cat={r['by_cat']}")
    print("price coverage quantiles:", report["prices"]["price_week_coverage"])
    print(f"share price coverage < 0.95: {report['prices']['share_coverage_lt_0_95']:.3f}")
    print(f"calendar events: {report['calendar']['event_name_1_nonnull']} days, "
          f"snap: {report['calendar']['snap_days']}")
    print(f"fold 1 test: {folds[0]['test_start_date']} to {folds[0]['test_end_date']}, "
          f"fold 12 test: {folds[-1]['test_start_date']} to {folds[-1]['test_end_date']}")
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
