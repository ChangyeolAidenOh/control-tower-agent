SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

CALENDAR_COLS = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                 "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                 "snap_ca", "snap_tx", "snap_wi"]
PRICE_COLS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
SUBSET_COLS = ["bundle_id", "state_id", "store_id", "dept_id", "cat_id", "item_id", "id",
               "volume_tercile", "active_days", "zero_share_active", "trailing_zero_days", "mean_daily_active"]
SALES_LONG_COLS = SALES_ID_COLS + ["d", "sales"]


def load_raw(cur, duck, ds_cfg, loader_version, root, record_manifest, copy_file, copy_query):
    raw_dir = root / ds_cfg["raw_dir"]
    sales_path = raw_dir / ds_cfg["files"]["sales"]
    calendar_path = raw_dir / ds_cfg["files"]["calendar"]
    prices_path = raw_dir / ds_cfg["files"]["prices"]
    subset_path = root / ds_cfg["subset_csv"]

    cur.execute("TRUNCATE raw.m5_calendar, raw.m5_sell_prices, raw.m5_sales_long, raw.sku_subset_v2")

    copy_file(cur, "raw.m5_calendar", CALENDAR_COLS, calendar_path, header=True)
    cur.execute("SELECT count(*) FROM raw.m5_calendar")
    n = cur.fetchone()[0]
    record_manifest(cur, "m5", calendar_path, n, n, None, loader_version)

    copy_file(cur, "raw.m5_sell_prices", PRICE_COLS, prices_path, header=True)
    cur.execute("SELECT count(*) FROM raw.m5_sell_prices")
    n = cur.fetchone()[0]
    record_manifest(cur, "m5", prices_path, n, n, None, loader_version)

    copy_file(cur, "raw.sku_subset_v2", SUBSET_COLS, subset_path, header=True)
    cur.execute("SELECT count(*) FROM raw.sku_subset_v2")
    n = cur.fetchone()[0]
    record_manifest(cur, "m5", subset_path, n, n, None, loader_version)

    duck.execute(
        """
        CREATE OR REPLACE TEMP TABLE m5_sales_sub AS
        SELECT s.*
        FROM read_csv_auto(?, header=true) s
        WHERE s.item_id IN (SELECT DISTINCT item_id FROM read_csv_auto(?, header=true))
        """,
        [str(sales_path), str(subset_path)],
    )
    source_rows = duck.execute(
        "SELECT count(*) FROM read_csv_auto(?, header=true, all_varchar=true)", [str(sales_path)]
    ).fetchone()[0]
    id_cols = ", ".join(SALES_ID_COLS)
    copy_query(
        cur, "raw.m5_sales_long", SALES_LONG_COLS, duck,
        f"""
        SELECT {id_cols}, d, CAST(sales AS BIGINT) AS sales
        FROM (UNPIVOT m5_sales_sub ON COLUMNS(* EXCLUDE ({id_cols})) INTO NAME d VALUE sales)
        """,
    )
    cur.execute("SELECT count(*) FROM raw.m5_sales_long")
    loaded_rows = cur.fetchone()[0]
    record_manifest(cur, "m5", sales_path, source_rows, loaded_rows,
                    "item_id in sku_subset_v2 (all stores); wide -> long melt", loader_version)
