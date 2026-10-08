-- mart tier: pre-registered analysis objects for M5 (Stage 1-B..1-E). Built by scripts/build_mart.py.

-- 1-C: rolling-origin folds, d indices copied verbatim from outputs/stage0e_m5_structure.json["folds"]["layout"].
CREATE TABLE IF NOT EXISTS mart.folds (
    fold             SMALLINT PRIMARY KEY,
    horizon_days     SMALLINT NOT NULL,
    test_start_idx   INTEGER  NOT NULL,
    test_end_idx     INTEGER  NOT NULL,
    train_end_idx    INTEGER  NOT NULL,
    warmup_start_idx INTEGER  NOT NULL,
    warmup_days      SMALLINT NOT NULL,
    test_start_date  DATE     NOT NULL,
    test_end_date    DATE     NOT NULL,
    is_retrain       BOOLEAN  NOT NULL,
    CHECK (train_end_idx = test_start_idx - 1),
    CHECK (test_end_idx - test_start_idx + 1 = horizon_days),
    CHECK (warmup_start_idx = test_start_idx - warmup_days)
);

-- Window constants resolved from configs/data.yaml (selection window, kappa validation window).
CREATE TABLE IF NOT EXISTS mart.windows (
    window_name TEXT PRIMARY KEY,
    start_idx   INTEGER NOT NULL,
    end_idx     INTEGER NOT NULL,
    start_date  DATE    NOT NULL,
    end_date    DATE    NOT NULL,
    definition  TEXT    NOT NULL
);

-- 1-B: SKU x day panel for the pre-registered subset with calendar covariates and window flags.
CREATE TABLE IF NOT EXISTS mart.sku_daily (
    series_id            TEXT           NOT NULL,
    group_id             TEXT           NOT NULL,
    item_id              TEXT           NOT NULL,
    store_id             TEXT           NOT NULL,
    state_id             TEXT           NOT NULL,
    dept_id              TEXT           NOT NULL,
    cat_id               TEXT           NOT NULL,
    period_idx           INTEGER        NOT NULL,
    period_date          DATE           NOT NULL,
    sales                INTEGER        NOT NULL,
    price                NUMERIC(14, 4),
    price_ffill          NUMERIC(14, 4),
    is_listed            BOOLEAN        NOT NULL,
    is_active            BOOLEAN        NOT NULL,
    wday                 SMALLINT       NOT NULL,
    month                SMALLINT       NOT NULL,
    year                 SMALLINT       NOT NULL,
    event_name_1         TEXT,
    event_type_1         TEXT,
    event_name_2         TEXT,
    event_type_2         TEXT,
    snap                 SMALLINT       NOT NULL,
    in_selection_window  BOOLEAN        NOT NULL,
    in_validation_window BOOLEAN        NOT NULL,
    PRIMARY KEY (series_id, period_idx)
);

CREATE INDEX IF NOT EXISTS ix_sku_daily_period ON mart.sku_daily (period_idx);
CREATE INDEX IF NOT EXISTS ix_sku_daily_group_period ON mart.sku_daily (group_id, period_idx);

-- 1-D: cost parameters. h_i = rho * p_bar_i with rho fixed so median(h) = 1; b_i = r * h_i for the pre-registered r grid.
CREATE TABLE IF NOT EXISTS mart.cost_meta (
    cost_version   TEXT PRIMARY KEY,
    rho            DOUBLE PRECISION NOT NULL,
    p_bar_median   DOUBLE PRECISION NOT NULL,
    price_window   TEXT NOT NULL,
    r_grid         SMALLINT[] NOT NULL,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mart.sku_cost (
    series_id     TEXT PRIMARY KEY,
    group_id      TEXT NOT NULL,
    p_bar         DOUBLE PRECISION NOT NULL,
    listed_days   INTEGER NOT NULL,
    h             DOUBLE PRECISION NOT NULL,
    b_r3          DOUBLE PRECISION NOT NULL,
    b_r5          DOUBLE PRECISION NOT NULL,
    b_r10         DOUBLE PRECISION NOT NULL
);

-- 1-E: bundles (shared-constraint units). demand_base = sum over SKUs of selection-window mean daily sales on active days.
CREATE TABLE IF NOT EXISTS mart.bundles (
    group_id                 TEXT PRIMARY KEY,
    state_id                 TEXT NOT NULL,
    dept_id                  TEXT NOT NULL,
    n_sku                    INTEGER NOT NULL,
    demand_base              DOUBLE PRECISION NOT NULL,
    demand_base_subset_csv   DOUBLE PRECISION NOT NULL,
    k_g_kappa_1_0            DOUBLE PRECISION NOT NULL,
    k_g_kappa_1_2            DOUBLE PRECISION NOT NULL,
    k_g_kappa_1_5            DOUBLE PRECISION NOT NULL
);
