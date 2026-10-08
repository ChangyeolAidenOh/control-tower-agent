import json
from datetime import date, timedelta

from core_pipeline.data.db import PROJECT_ROOT, connect, load_config, run_sql_file

M5_DAY1 = date(2011, 1, 29)
SQL_ORDER = ["sku_daily.sql", "sku_cost.sql", "bundles.sql", "views.sql"]


def d_index(label):
    return int(label.split("_")[1])


def idx_to_date(idx):
    return M5_DAY1 + timedelta(days=idx - 1)


def load_folds(cur, cfg):
    win = cfg["datasets"]["m5"]["windows"]
    layout = json.load(open(PROJECT_ROOT / win["folds_json"]))[win["folds_key"]]
    horizon = layout["horizon"]
    warmup = win["warmup_days"]
    retrain = set(win["retrain_folds"])
    cur.execute("DELETE FROM mart.folds")
    for f in layout["layout"]:
        start, end, train_end = d_index(f["test_start_d"]), d_index(f["test_end_d"]), d_index(f["train_end_d"])
        cur.execute(
            """
            INSERT INTO mart.folds (fold, horizon_days, test_start_idx, test_end_idx, train_end_idx,
                                    warmup_start_idx, warmup_days, test_start_date, test_end_date, is_retrain)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (f["fold"], horizon, start, end, train_end, start - warmup, warmup,
             f["test_start_date"], f["test_end_date"], f["fold"] in retrain),
        )
    return len(layout["layout"]), layout["n_folds"]


def load_windows(cur, cfg):
    win = cfg["datasets"]["m5"]["windows"]
    sel_end = win["selection_end_idx"]
    val_days = 7 * win["validation_weeks"]
    rows = [
        ("selection", 1, sel_end, "all days up to the pre-fold-1 selection end (subset v2 statistics, p_bar, demand_base)"),
        ("validation", sel_end - val_days + 1, sel_end,
         f"{win['validation_weeks']} whole weeks ending at the selection end (A4-only kappa binding-share measurement)"),
    ]
    cur.execute("DELETE FROM mart.windows")
    for name, start, end, definition in rows:
        cur.execute(
            """
            INSERT INTO mart.windows (window_name, start_idx, end_idx, start_date, end_date, definition)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (name, start, end, idx_to_date(start), idx_to_date(end), definition),
        )
    return rows


def main():
    cfg = load_config()
    with connect(cfg) as conn, conn.cursor() as cur:
        run_sql_file(cur, PROJECT_ROOT / "sql" / "03_mart_tables.sql")
        n_loaded, n_expected = load_folds(cur, cfg)
        if n_loaded != n_expected:
            raise ValueError(f"folds layout has {n_loaded} entries, n_folds says {n_expected}")
        windows = load_windows(cur, cfg)
        for name in SQL_ORDER:
            run_sql_file(cur, PROJECT_ROOT / "sql" / "mart" / name)
        cur.execute("SELECT count(*), min(period_idx), max(period_idx) FROM mart.sku_daily")
        n_rows, p_min, p_max = cur.fetchone()
        cur.execute("SELECT rho, p_bar_median FROM mart.cost_meta WHERE cost_version = 'v1'")
        rho, p_med = cur.fetchone()
        cur.execute("SELECT count(*), max(abs(demand_base - demand_base_subset_csv)) FROM mart.bundles")
        n_bundles, max_diff = cur.fetchone()
        conn.commit()
    print(f"folds={n_loaded} windows={[(w[0], w[1], w[2]) for w in windows]}")
    print(f"sku_daily rows={n_rows} periods={p_min}..{p_max}")
    print(f"cost: rho={rho:.6f} p_bar_median={p_med:.4f}")
    print(f"bundles={n_bundles} max|demand_base - subset_csv|={max_diff:.6g}")


if __name__ == "__main__":
    main()
