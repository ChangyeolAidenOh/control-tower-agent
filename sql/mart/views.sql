-- 1-C: fold views. Information set for fold k = periods with period_idx <= train_end_idx (is_train).
-- Warm-up = 14 days before the test window (is_warmup, a subset of train days). Test = the 28-day window.
CREATE OR REPLACE VIEW mart.v_fold_periods AS
SELECT f.fold,
       pe.period_idx,
       pe.period_date,
       pe.period_idx <= f.train_end_idx                                      AS is_train,
       pe.period_idx BETWEEN f.warmup_start_idx AND f.train_end_idx         AS is_warmup,
       pe.period_idx BETWEEN f.test_start_idx AND f.test_end_idx            AS is_test,
       f.is_retrain
FROM mart.folds f
CROSS JOIN staging.periods pe
WHERE pe.dataset = 'm5' AND pe.period_idx <= f.test_end_idx;

CREATE OR REPLACE VIEW mart.v_sku_fold AS
SELECT fp.fold, fp.is_train, fp.is_warmup, fp.is_test, fp.is_retrain, d.*
FROM mart.v_fold_periods fp
JOIN mart.sku_daily d ON d.period_idx = fp.period_idx;

CREATE OR REPLACE VIEW mart.v_fold_train AS
SELECT * FROM mart.v_sku_fold WHERE is_train;

CREATE OR REPLACE VIEW mart.v_fold_test AS
SELECT * FROM mart.v_sku_fold WHERE is_test;

CREATE OR REPLACE VIEW mart.v_fold_warmup AS
SELECT * FROM mart.v_sku_fold WHERE is_warmup;

-- kappa validation window (A4-only binding-share measurement, preregistration section 5.5).
CREATE OR REPLACE VIEW mart.v_validation_window AS
SELECT * FROM mart.sku_daily WHERE in_validation_window;
