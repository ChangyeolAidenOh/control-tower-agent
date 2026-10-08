-- Derive the common loader schema for M5 from raw. Idempotent: replaces all m5 rows.
-- Population = pre-registered subset v2 only (154 series, 10 bundles). Other stores stay in raw for ad hoc use.
DELETE FROM staging.panel         WHERE dataset = 'm5';
DELETE FROM staging.series        WHERE dataset = 'm5';
DELETE FROM staging.groups        WHERE dataset = 'm5';
DELETE FROM staging.period_region WHERE dataset = 'm5';
DELETE FROM staging.periods       WHERE dataset = 'm5';
DELETE FROM staging.datasets      WHERE dataset = 'm5';

INSERT INTO staging.datasets (dataset, grain, period_idx_def, period_date_def, price_def, in_stock_def)
VALUES ('m5', 'day',
        'integer suffix of calendar.d (d_k -> k); sales observed for 1..1941, calendar extends to 1969',
        'calendar.date',
        'sell_prices.sell_price joined on (store_id, item_id, calendar.wm_yr_wk); NULL before first listing',
        'not observed in M5; always NULL');

INSERT INTO staging.periods (dataset, period_idx, period_date, price_period, weekday, month, year,
                             event_name_1, event_type_1, event_name_2, event_type_2)
SELECT 'm5', substr(d, 3)::int, date, wm_yr_wk, wday, month, year,
       event_name_1, event_type_1, event_name_2, event_type_2
FROM raw.m5_calendar;

INSERT INTO staging.period_region (dataset, period_idx, region_id, snap)
SELECT 'm5', substr(d, 3)::int, r.region_id, r.snap
FROM raw.m5_calendar c
CROSS JOIN LATERAL (VALUES ('CA', c.snap_ca), ('TX', c.snap_tx), ('WI', c.snap_wi)) AS r (region_id, snap);

INSERT INTO staging.groups (dataset, group_id, level1, level2)
SELECT DISTINCT 'm5', bundle_id, state_id, dept_id
FROM raw.sku_subset_v2;

INSERT INTO staging.series (dataset, series_id, group_id, item_id, location_id, region_id, dept_id, cat_id, in_subset)
SELECT 'm5', item_id || '__' || store_id, bundle_id, item_id, store_id, state_id, dept_id, cat_id, TRUE
FROM raw.sku_subset_v2;

INSERT INTO staging.panel (dataset, series_id, group_id, period_idx, sales, price, in_stock)
SELECT 'm5',
       s.item_id || '__' || s.store_id,
       sub.bundle_id,
       substr(s.d, 3)::int,
       s.sales,
       p.sell_price,
       NULL
FROM raw.m5_sales_long s
JOIN raw.sku_subset_v2 sub ON sub.item_id = s.item_id AND sub.store_id = s.store_id
JOIN raw.m5_calendar c ON c.d = s.d
LEFT JOIN raw.m5_sell_prices p
       ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk;
