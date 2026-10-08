-- staging tier: the common loader schema. Every dataset lands here with the same shape.
-- Grain: one row per (dataset, series_id, period_idx). period_idx is the dataset's native integer index
-- (M5: d number 1..1941; VN2: week ordinal from the first week column), period_date is the period start date.
-- Folds are pre-registered on M5 d indices, so period_idx is the primary time key and period_date is derived.

CREATE TABLE IF NOT EXISTS staging.datasets (
    dataset         TEXT PRIMARY KEY,
    grain           TEXT NOT NULL CHECK (grain IN ('day', 'week')),
    period_idx_def  TEXT NOT NULL,
    period_date_def TEXT NOT NULL,
    price_def       TEXT NOT NULL,
    in_stock_def    TEXT NOT NULL,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Shared-constraint unit. M5: group_id = pre-registered bundle_id (state x dept); VN2: store x department (Stage 9 only).
CREATE TABLE IF NOT EXISTS staging.groups (
    dataset  TEXT NOT NULL REFERENCES staging.datasets (dataset),
    group_id TEXT NOT NULL,
    level1   TEXT NOT NULL,
    level2   TEXT NOT NULL,
    PRIMARY KEY (dataset, group_id)
);

-- Series dimension. M5: series = item x store (subset v2 only). VN2: series = store x product (all).
CREATE TABLE IF NOT EXISTS staging.series (
    dataset     TEXT NOT NULL REFERENCES staging.datasets (dataset),
    series_id   TEXT NOT NULL,
    group_id    TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    location_id TEXT NOT NULL,
    region_id   TEXT,
    dept_id     TEXT,
    cat_id      TEXT,
    in_subset   BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (dataset, series_id),
    FOREIGN KEY (dataset, group_id) REFERENCES staging.groups (dataset, group_id)
);

CREATE INDEX IF NOT EXISTS ix_series_group ON staging.series (dataset, group_id);

-- Period dimension with dataset-wide covariates (calendar, events). Region-level covariates (SNAP) are in period_region.
CREATE TABLE IF NOT EXISTS staging.periods (
    dataset      TEXT     NOT NULL REFERENCES staging.datasets (dataset),
    period_idx   INTEGER  NOT NULL,
    period_date  DATE     NOT NULL,
    price_period INTEGER,
    weekday      SMALLINT,
    month        SMALLINT,
    year         SMALLINT,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    PRIMARY KEY (dataset, period_idx),
    UNIQUE (dataset, period_date)
);

CREATE TABLE IF NOT EXISTS staging.period_region (
    dataset    TEXT     NOT NULL,
    period_idx INTEGER  NOT NULL,
    region_id  TEXT     NOT NULL,
    snap       SMALLINT NOT NULL,
    PRIMARY KEY (dataset, period_idx, region_id),
    FOREIGN KEY (dataset, period_idx) REFERENCES staging.periods (dataset, period_idx)
);

-- The common loader fact: (series_id, group_id, period, sales, price, in_stock optional).
-- group_id is denormalized on purpose so bundle-level queries need no join.
-- price NULL = not listed in that period (M5 before first availability; VN2 has no price file). in_stock NULL = not observed (M5).
CREATE TABLE IF NOT EXISTS staging.panel (
    dataset    TEXT           NOT NULL,
    series_id  TEXT           NOT NULL,
    group_id   TEXT           NOT NULL,
    period_idx INTEGER        NOT NULL,
    sales      NUMERIC(14, 4) NOT NULL CHECK (sales >= 0),
    price      NUMERIC(14, 4) CHECK (price IS NULL OR price > 0),
    in_stock   BOOLEAN,
    PRIMARY KEY (dataset, series_id, period_idx),
    FOREIGN KEY (dataset, series_id) REFERENCES staging.series (dataset, series_id),
    FOREIGN KEY (dataset, period_idx) REFERENCES staging.periods (dataset, period_idx)
);

CREATE INDEX IF NOT EXISTS ix_panel_period ON staging.panel (dataset, period_idx);
CREATE INDEX IF NOT EXISTS ix_panel_group_period ON staging.panel (dataset, group_id, period_idx);

-- Loader run log: one row per load, with config hash so a mart can be traced to its inputs.
CREATE TABLE IF NOT EXISTS staging.load_runs (
    run_id         BIGSERIAL PRIMARY KEY,
    dataset        TEXT        NOT NULL,
    loader_version TEXT        NOT NULL,
    config_sha256  CHAR(64)    NOT NULL,
    series_count   INTEGER     NOT NULL,
    period_min     INTEGER     NOT NULL,
    period_max     INTEGER     NOT NULL,
    panel_rows     BIGINT      NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL,
    finished_at    TIMESTAMPTZ NOT NULL
);
