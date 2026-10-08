-- raw tier: column names and types follow the source files verbatim (snake_case only where the source has spaces).
-- Provenance of every loaded file (full-file hash even when only a subset of rows is loaded).
CREATE TABLE IF NOT EXISTS raw.file_manifest (
    dataset        TEXT        NOT NULL,
    file_name      TEXT        NOT NULL,
    file_path      TEXT        NOT NULL,
    sha256         CHAR(64)    NOT NULL,
    byte_size      BIGINT      NOT NULL,
    source_rows    BIGINT      NOT NULL,
    loaded_rows    BIGINT      NOT NULL,
    row_filter     TEXT,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    loader_version TEXT        NOT NULL,
    PRIMARY KEY (dataset, file_name)
);

-- M5 calendar.csv (1969 rows), verbatim.
CREATE TABLE IF NOT EXISTS raw.m5_calendar (
    date         DATE     NOT NULL,
    wm_yr_wk     INTEGER  NOT NULL,
    weekday      TEXT     NOT NULL,
    wday         SMALLINT NOT NULL,
    month        SMALLINT NOT NULL,
    year         SMALLINT NOT NULL,
    d            TEXT     NOT NULL,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_ca      SMALLINT NOT NULL,
    snap_tx      SMALLINT NOT NULL,
    snap_wi      SMALLINT NOT NULL,
    PRIMARY KEY (d)
);

-- M5 sell_prices.csv (~6.8M rows), verbatim, full.
CREATE TABLE IF NOT EXISTS raw.m5_sell_prices (
    store_id   TEXT           NOT NULL,
    item_id    TEXT           NOT NULL,
    wm_yr_wk   INTEGER        NOT NULL,
    sell_price NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (store_id, item_id, wm_yr_wk)
);

-- M5 sales_train_evaluation.csv melted to long form.
-- The wide file has 1947 columns (> PostgreSQL 1600-column limit), so melt is the only transformation.
-- Row filter: item_id in subset v2 items, all 10 stores (recorded in file_manifest.row_filter).
CREATE TABLE IF NOT EXISTS raw.m5_sales_long (
    id       TEXT     NOT NULL,
    item_id  TEXT     NOT NULL,
    dept_id  TEXT     NOT NULL,
    cat_id   TEXT     NOT NULL,
    store_id TEXT     NOT NULL,
    state_id TEXT     NOT NULL,
    d        TEXT     NOT NULL,
    sales    INTEGER  NOT NULL,
    PRIMARY KEY (id, d)
);

-- Pre-registered SKU subset v2 (data/processed/sku_subset_v2.csv), verbatim copy for traceability.
CREATE TABLE IF NOT EXISTS raw.sku_subset_v2 (
    bundle_id          TEXT             NOT NULL,
    state_id           TEXT             NOT NULL,
    store_id           TEXT             NOT NULL,
    dept_id            TEXT             NOT NULL,
    cat_id             TEXT             NOT NULL,
    item_id            TEXT             NOT NULL,
    id                 TEXT             NOT NULL,
    volume_tercile     TEXT             NOT NULL,
    active_days        INTEGER          NOT NULL,
    zero_share_active  DOUBLE PRECISION NOT NULL,
    trailing_zero_days INTEGER          NOT NULL,
    mean_daily_active  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (item_id, store_id)
);

-- VN2 Week 0 files (Store x Product, weekly). Sales / In Stock are wide (week start dates as columns) and are melted.
CREATE TABLE IF NOT EXISTS raw.vn2_master (
    store            INTEGER NOT NULL,
    product          INTEGER NOT NULL,
    product_group    INTEGER,
    division         INTEGER,
    department       INTEGER,
    department_group INTEGER,
    store_format     INTEGER,
    format           INTEGER,
    PRIMARY KEY (store, product)
);

CREATE TABLE IF NOT EXISTS raw.vn2_initial_state (
    store                    INTEGER NOT NULL,
    product                  INTEGER NOT NULL,
    start_inventory          NUMERIC(14, 4),
    sales                    NUMERIC(14, 4),
    missed_sales             NUMERIC(14, 4),
    end_inventory            NUMERIC(14, 4),
    in_transit_w1            NUMERIC(14, 4),
    in_transit_w2            NUMERIC(14, 4),
    holding_cost             NUMERIC(14, 4),
    shortage_cost            NUMERIC(14, 4),
    cumulative_holding_cost  NUMERIC(14, 4),
    cumulative_shortage_cost NUMERIC(14, 4),
    PRIMARY KEY (store, product)
);

CREATE TABLE IF NOT EXISTS raw.vn2_sales_long (
    store      INTEGER        NOT NULL,
    product    INTEGER        NOT NULL,
    week_start DATE           NOT NULL,
    sales      NUMERIC(14, 4),
    PRIMARY KEY (store, product, week_start)
);

CREATE TABLE IF NOT EXISTS raw.vn2_in_stock_long (
    store      INTEGER NOT NULL,
    product    INTEGER NOT NULL,
    week_start DATE    NOT NULL,
    in_stock   BOOLEAN,
    PRIMARY KEY (store, product, week_start)
);
