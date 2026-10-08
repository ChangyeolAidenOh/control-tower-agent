-- 1-B: build mart.sku_daily from staging (M5 only). Replaces all rows.
-- price_ffill = last observed price up to day t within the series (no backward fill: a NULL before the first listing stays NULL).
-- is_listed = price observed that day. is_active = on/after the series' first positive-sales day.
-- Window flags come from mart.windows (built by scripts/build_mart.py from configs/data.yaml).
TRUNCATE mart.sku_daily;

INSERT INTO mart.sku_daily
WITH base AS (
    SELECT p.series_id, p.group_id, s.item_id, s.location_id AS store_id, s.region_id AS state_id,
           s.dept_id, s.cat_id, p.period_idx, pe.period_date, p.sales::int AS sales, p.price,
           pe.weekday AS wday, pe.month, pe.year,
           pe.event_name_1, pe.event_type_1, pe.event_name_2, pe.event_type_2,
           pr.snap
    FROM staging.panel p
    JOIN staging.series s ON s.dataset = p.dataset AND s.series_id = p.series_id
    JOIN staging.periods pe ON pe.dataset = p.dataset AND pe.period_idx = p.period_idx
    JOIN staging.period_region pr ON pr.dataset = p.dataset AND pr.period_idx = p.period_idx AND pr.region_id = s.region_id
    WHERE p.dataset = 'm5'
),
filled AS (
    SELECT b.*,
           max(price) OVER (PARTITION BY series_id ORDER BY period_idx
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS price_ffill_raw,
           count(price) OVER (PARTITION BY series_id ORDER BY period_idx
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS price_grp,
           min(CASE WHEN sales > 0 THEN period_idx END) OVER (PARTITION BY series_id) AS first_sale_idx
    FROM base b
),
ffill AS (
    SELECT f.*,
           first_value(price) OVER (PARTITION BY series_id, price_grp ORDER BY period_idx) AS price_ffill
    FROM filled f
)
SELECT series_id, group_id, item_id, store_id, state_id, dept_id, cat_id,
       period_idx, period_date, sales, price, price_ffill,
       price IS NOT NULL AS is_listed,
       period_idx >= first_sale_idx AS is_active,
       wday, month, year, event_name_1, event_type_1, event_name_2, event_type_2, snap,
       period_idx BETWEEN ws.start_idx AND ws.end_idx AS in_selection_window,
       period_idx BETWEEN wv.start_idx AND wv.end_idx AS in_validation_window
FROM ffill
CROSS JOIN (SELECT start_idx, end_idx FROM mart.windows WHERE window_name = 'selection') ws
CROSS JOIN (SELECT start_idx, end_idx FROM mart.windows WHERE window_name = 'validation') wv;
