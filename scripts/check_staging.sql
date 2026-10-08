SELECT dataset, grain, loaded_at FROM staging.datasets ORDER BY dataset;
SELECT dataset, count(*) AS groups FROM staging.groups GROUP BY dataset;
SELECT dataset, count(*) AS series, sum(in_subset::int) AS in_subset FROM staging.series GROUP BY dataset;
SELECT dataset, count(*) AS rows, count(DISTINCT series_id) AS series, min(period_idx), max(period_idx),
       round(avg((price IS NULL)::int), 4) AS price_null_share,
       round(avg((in_stock IS NULL)::int), 4) AS in_stock_null_share
FROM staging.panel GROUP BY dataset;
SELECT group_id, count(*) AS sku FROM staging.series WHERE dataset = 'm5' GROUP BY group_id ORDER BY group_id;
SELECT dataset, file_name, source_rows, loaded_rows, row_filter FROM raw.file_manifest ORDER BY dataset, file_name;
SELECT * FROM staging.load_runs ORDER BY run_id DESC LIMIT 2;
