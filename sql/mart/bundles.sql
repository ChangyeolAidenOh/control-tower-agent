-- 1-E: bundle table. demand_base = sum_i mean(sales_i on active days in the selection window),
-- the same statistic as mean_daily_active in sku_subset_v2.csv (recomputed from the mart and cross-checked).
-- K_g = kappa * demand_base for the pre-registered kappa candidates {1.0, 1.2, 1.5}.
DELETE FROM mart.bundles;

INSERT INTO mart.bundles
WITH per_sku AS (
    SELECT series_id, group_id, state_id, dept_id, avg(sales)::float8 AS mean_daily_active
    FROM mart.sku_daily
    WHERE in_selection_window AND is_active
    GROUP BY series_id, group_id, state_id, dept_id
),
per_bundle AS (
    SELECT group_id, state_id, dept_id, count(*) AS n_sku, sum(mean_daily_active) AS demand_base
    FROM per_sku
    GROUP BY group_id, state_id, dept_id
),
from_csv AS (
    SELECT bundle_id AS group_id, sum(mean_daily_active)::float8 AS demand_base_subset_csv
    FROM raw.sku_subset_v2
    GROUP BY bundle_id
)
SELECT b.group_id, b.state_id, b.dept_id, b.n_sku, b.demand_base, c.demand_base_subset_csv,
       1.0 * b.demand_base, 1.2 * b.demand_base, 1.5 * b.demand_base
FROM per_bundle b
JOIN from_csv c USING (group_id);
