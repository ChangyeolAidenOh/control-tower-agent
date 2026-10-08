SELECT * FROM mart.windows;
SELECT fold, test_start_idx, test_end_idx, train_end_idx, warmup_start_idx, test_start_date, test_end_date, is_retrain FROM mart.folds ORDER BY fold;
SELECT count(*) AS rows, count(DISTINCT series_id) AS series,
       sum(is_listed::int) AS listed, sum(is_active::int) AS active,
       sum((price_ffill IS NULL)::int) AS price_ffill_null,
       sum(in_selection_window::int) AS sel_rows, sum(in_validation_window::int) AS val_rows
FROM mart.sku_daily;
SELECT min(p_bar), percentile_cont(0.5) WITHIN GROUP (ORDER BY p_bar) AS p_bar_med, max(p_bar),
       min(h), percentile_cont(0.5) WITHIN GROUP (ORDER BY h) AS h_med, max(h), min(listed_days)
FROM mart.sku_cost;
SELECT group_id, n_sku, round(demand_base::numeric, 3) AS demand_base, round(demand_base_subset_csv::numeric, 3) AS from_csv,
       round(k_g_kappa_1_2::numeric, 2) AS k_g_1_2
FROM mart.bundles ORDER BY group_id;
SELECT fold, sum(is_train::int) AS train_days, sum(is_warmup::int) AS warmup_days, sum(is_test::int) AS test_days,
       max(CASE WHEN is_train THEN period_idx END) AS max_train_idx, min(CASE WHEN is_test THEN period_idx END) AS min_test_idx
FROM mart.v_fold_periods GROUP BY fold ORDER BY fold;
