MASTER_COLS = ["store", "product", "product_group", "division", "department",
               "department_group", "store_format", "format"]
INITIAL_STATE_COLS = ["store", "product", "start_inventory", "sales", "missed_sales", "end_inventory",
                      "in_transit_w1", "in_transit_w2", "holding_cost", "shortage_cost",
                      "cumulative_holding_cost", "cumulative_shortage_cost"]
LONG_COLS = ["store", "product", "week_start"]


def _melt_query(duck, value_col, path, cast):
    duck.execute(
        "CREATE OR REPLACE TEMP TABLE vn2_wide AS SELECT * FROM read_csv_auto(?, header=true)", [str(path)]
    )
    return (
        f"""
        SELECT Store, Product, CAST(week_start AS DATE) AS week_start, CAST({value_col} AS {cast}) AS {value_col}
        FROM (UNPIVOT vn2_wide ON COLUMNS(* EXCLUDE (Store, Product)) INTO NAME week_start VALUE {value_col})
        """
    )


def load_raw(cur, duck, ds_cfg, loader_version, root, record_manifest, copy_file, copy_query):
    raw_dir = root / ds_cfg["raw_dir"]
    files = ds_cfg["files"]
    master_path = raw_dir / files["master"]
    initial_path = raw_dir / files["initial_state"]
    sales_path = raw_dir / files["sales"]
    in_stock_path = raw_dir / files["in_stock"]

    cur.execute("TRUNCATE raw.vn2_master, raw.vn2_initial_state, raw.vn2_sales_long, raw.vn2_in_stock_long")

    copy_file(cur, "raw.vn2_master", MASTER_COLS, master_path, header=True)
    cur.execute("SELECT count(*) FROM raw.vn2_master")
    n = cur.fetchone()[0]
    record_manifest(cur, "vn2", master_path, n, n, None, loader_version)

    copy_file(cur, "raw.vn2_initial_state", INITIAL_STATE_COLS, initial_path, header=True)
    cur.execute("SELECT count(*) FROM raw.vn2_initial_state")
    n = cur.fetchone()[0]
    record_manifest(cur, "vn2", initial_path, n, n, None, loader_version)

    for table, value_col, path, cast in (
        ("raw.vn2_sales_long", "sales", sales_path, "DOUBLE"),
        ("raw.vn2_in_stock_long", "in_stock", in_stock_path, "BOOLEAN"),
    ):
        source_rows = duck.execute(
            "SELECT count(*) FROM read_csv_auto(?, header=true, all_varchar=true)", [str(path)]
        ).fetchone()[0]
        copy_query(cur, table, LONG_COLS + [value_col], duck, _melt_query(duck, value_col, path, cast))
        cur.execute(f"SELECT count(*) FROM {table}")
        loaded_rows = cur.fetchone()[0]
        record_manifest(cur, "vn2", path, source_rows, loaded_rows, "wide -> long melt (week columns)", loader_version)

    # Every sales cell must have an in_stock observation; in_stock may extend past the sales horizon
    # (competition forecast weeks), which raw keeps verbatim and staging excludes (preregistration section 9).
    cur.execute(
        """
        SELECT count(*) FROM raw.vn2_sales_long s
        LEFT JOIN raw.vn2_in_stock_long k USING (store, product, week_start)
        WHERE k.store IS NULL
        """
    )
    missing = cur.fetchone()[0]
    if missing:
        raise ValueError(f"vn2 sales cells without in_stock observation: {missing}")
    cur.execute(
        """
        SELECT count(DISTINCT week_start), min(week_start), max(week_start)
        FROM raw.vn2_in_stock_long
        WHERE week_start > (SELECT max(week_start) FROM raw.vn2_sales_long)
        """
    )
    extra_weeks, first_extra, last_extra = cur.fetchone()
    if extra_weeks:
        print(f"vn2: in_stock extends {extra_weeks} weeks past sales ({first_extra}..{last_extra}); kept in raw, excluded from staging")
