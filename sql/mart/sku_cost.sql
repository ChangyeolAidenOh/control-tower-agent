-- 1-D: p_bar_i = mean observed price over listed days in the selection window; rho = 1 / median(p_bar) so median(h) = 1.
DELETE FROM mart.sku_cost;
DELETE FROM mart.cost_meta WHERE cost_version = 'v1';

WITH pbar AS (
    SELECT series_id, group_id, avg(price)::float8 AS p_bar, count(price) AS listed_days
    FROM mart.sku_daily
    WHERE in_selection_window AND price IS NOT NULL
    GROUP BY series_id, group_id
),
med AS (
    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY p_bar) AS p_bar_median FROM pbar
),
meta AS (
    INSERT INTO mart.cost_meta (cost_version, rho, p_bar_median, price_window, r_grid)
    SELECT 'v1', 1.0 / p_bar_median, p_bar_median,
           'mean of observed sell_price over listed days with period_idx <= selection end',
           ARRAY[3, 5, 10]::smallint[]
    FROM med
    RETURNING rho
)
INSERT INTO mart.sku_cost (series_id, group_id, p_bar, listed_days, h, b_r3, b_r5, b_r10)
SELECT p.series_id, p.group_id, p.p_bar, p.listed_days,
       m.rho * p.p_bar, 3 * m.rho * p.p_bar, 5 * m.rho * p.p_bar, 10 * m.rho * p.p_bar
FROM pbar p CROSS JOIN meta m;
