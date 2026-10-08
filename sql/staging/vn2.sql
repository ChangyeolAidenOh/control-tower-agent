-- Derive the common loader schema for VN2 from raw. Idempotent: replaces all vn2 rows.
-- Population = all Store x Product series; the Stage 9 subset (store x department groups) is chosen later via in_subset.
-- Periods come from the Sales file only. In Stock weeks past the sales horizon (competition forecast weeks)
-- stay in raw and are excluded here (preregistration section 9: not part of the information set).
DELETE FROM staging.panel         WHERE dataset = 'vn2';
DELETE FROM staging.series        WHERE dataset = 'vn2';
DELETE FROM staging.groups        WHERE dataset = 'vn2';
DELETE FROM staging.period_region WHERE dataset = 'vn2';
DELETE FROM staging.periods       WHERE dataset = 'vn2';
DELETE FROM staging.datasets      WHERE dataset = 'vn2';

INSERT INTO staging.datasets (dataset, grain, period_idx_def, period_date_def, price_def, in_stock_def)
VALUES ('vn2', 'week',
        'week ordinal: 0 = first week column of the Week 0 Sales file, +1 per following week',
        'week start date from the Sales file column header',
        'no price file in the VN2 Week 0 release; always NULL',
        'Week 0 In Stock file, boolean per store x product x week; weeks past the sales horizon excluded (raw keeps them)');

INSERT INTO staging.periods (dataset, period_idx, period_date, price_period, weekday, month, year,
                             event_name_1, event_type_1, event_name_2, event_type_2)
SELECT 'vn2',
       (row_number() OVER (ORDER BY week_start) - 1)::int,
       week_start,
       NULL, NULL,
       EXTRACT(MONTH FROM week_start)::smallint,
       EXTRACT(YEAR FROM week_start)::smallint,
       NULL, NULL, NULL, NULL
FROM (SELECT DISTINCT week_start FROM raw.vn2_sales_long) w;

INSERT INTO staging.groups (dataset, group_id, level1, level2)
SELECT DISTINCT 'vn2', 'S' || store || '_D' || department, store::text, department::text
FROM raw.vn2_master;

INSERT INTO staging.series (dataset, series_id, group_id, item_id, location_id, region_id, dept_id, cat_id, in_subset)
SELECT 'vn2',
       store || '__' || product,
       'S' || store || '_D' || department,
       product::text,
       store::text,
       NULL,
       department::text,
       division::text,
       FALSE
FROM raw.vn2_master;

INSERT INTO staging.panel (dataset, series_id, group_id, period_idx, sales, price, in_stock)
SELECT 'vn2',
       s.store || '__' || s.product,
       'S' || m.store || '_D' || m.department,
       p.period_idx,
       s.sales,
       NULL,
       k.in_stock
FROM raw.vn2_sales_long s
JOIN raw.vn2_master m ON m.store = s.store AND m.product = s.product
JOIN staging.periods p ON p.dataset = 'vn2' AND p.period_date = s.week_start
LEFT JOIN raw.vn2_in_stock_long k
       ON k.store = s.store AND k.product = s.product AND k.week_start = s.week_start;
