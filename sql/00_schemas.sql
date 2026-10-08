-- 3-tier layout: raw (verbatim source) -> staging (dataset-agnostic loader schema) -> mart (analysis tables/views)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

COMMENT ON SCHEMA raw IS 'Source files as delivered (M5: calendar, sell_prices full; sales melted long for subset items only). No transformation beyond melt.';
COMMENT ON SCHEMA staging IS 'Common loader schema shared by M5 and VN2: series x period panel with optional in_stock.';
COMMENT ON SCHEMA mart IS 'Pre-registered analysis objects: sku_daily, folds, sku_cost, bundles, agent traces (Stage 1-B..1-E).';
