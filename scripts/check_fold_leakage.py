import sys

from core_pipeline.data.db import connect, load_config

CHECKS = [
    ("folds: contiguous, non-overlapping, horizon 28",
     """
     SELECT count(*) FROM (
         SELECT fold, test_start_idx, test_end_idx,
                lag(test_end_idx) OVER (ORDER BY fold) AS prev_end
         FROM mart.folds
     ) f
     WHERE (prev_end IS NOT NULL AND test_start_idx <> prev_end + 1)
        OR test_end_idx - test_start_idx + 1 <> 28
     """),
    ("folds: fold 1 train end = selection window end",
     """
     SELECT count(*) FROM mart.folds f, mart.windows w
     WHERE f.fold = 1 AND w.window_name = 'selection' AND f.train_end_idx <> w.end_idx
     """),
    ("folds: last test day = last day in mart.sku_daily",
     """
     SELECT count(*) FROM (SELECT max(test_end_idx) AS e FROM mart.folds) f,
                          (SELECT max(period_idx) AS m FROM mart.sku_daily) d
     WHERE f.e <> d.m
     """),
    ("windows: validation inside selection and 7-day multiple",
     """
     SELECT count(*) FROM mart.windows v, mart.windows s
     WHERE v.window_name = 'validation' AND s.window_name = 'selection'
       AND (v.start_idx < s.start_idx OR v.end_idx <> s.end_idx OR (v.end_idx - v.start_idx + 1) % 7 <> 0)
     """),
    ("train view: no period after train_end in any fold",
     """
     SELECT count(*) FROM (
         SELECT t.fold, max(t.period_idx) AS max_train, f.train_end_idx, f.test_start_idx
         FROM mart.v_fold_train t JOIN mart.folds f USING (fold)
         GROUP BY t.fold, f.train_end_idx, f.test_start_idx
     ) x
     WHERE max_train > train_end_idx OR max_train >= test_start_idx
     """),
    ("test view: exact 28-day window per fold and per series",
     """
     SELECT count(*) FROM (
         SELECT t.fold, t.series_id, min(t.period_idx) AS mn, max(t.period_idx) AS mx, count(*) AS n,
                f.test_start_idx, f.test_end_idx
         FROM mart.v_fold_test t JOIN mart.folds f USING (fold)
         GROUP BY t.fold, t.series_id, f.test_start_idx, f.test_end_idx
     ) x
     WHERE mn <> test_start_idx OR mx <> test_end_idx OR n <> 28
     """),
    ("warmup view: 14 days, all inside train",
     """
     SELECT count(*) FROM (
         SELECT fold, series_id, count(*) AS n, bool_and(is_train) AS all_train
         FROM mart.v_sku_fold WHERE is_warmup GROUP BY fold, series_id
     ) x
     WHERE n <> 14 OR NOT all_train
     """),
    ("selection flag: no day after selection end flagged",
     """
     SELECT count(*) FROM mart.sku_daily d, mart.windows w
     WHERE w.window_name = 'selection' AND d.in_selection_window AND d.period_idx > w.end_idx
     """),
    ("sku_cost: p_bar reproducible from selection-window rows only",
     """
     SELECT count(*) FROM mart.sku_cost c
     JOIN (SELECT series_id, avg(price)::float8 AS p FROM mart.sku_daily
           WHERE in_selection_window AND price IS NOT NULL GROUP BY series_id) r USING (series_id)
     WHERE abs(c.p_bar - r.p) > 1e-9
     """),
    ("bundles: demand_base reproducible from selection-window active rows only",
     """
     SELECT count(*) FROM mart.bundles b
     JOIN (SELECT group_id, sum(m) AS d FROM (
               SELECT group_id, series_id, avg(sales)::float8 AS m FROM mart.sku_daily
               WHERE in_selection_window AND is_active GROUP BY group_id, series_id) s
           GROUP BY group_id) r USING (group_id)
     WHERE abs(b.demand_base - r.d) > 1e-9
     """),
    ("sku_daily: complete series x day grid, no NULL sales",
     """
     SELECT count(*) FROM (
         SELECT series_id, count(*) AS n, count(sales) AS ns FROM mart.sku_daily GROUP BY series_id
     ) x, (SELECT max(period_idx) AS m FROM mart.sku_daily) d
     WHERE n <> d.m OR ns <> d.m
     """),
    ("price_ffill: never fills across a day the series was not yet listed",
     """
     SELECT count(*) FROM mart.sku_daily d
     WHERE price_ffill IS NOT NULL
       AND period_idx < (SELECT min(period_idx) FROM mart.sku_daily e WHERE e.series_id = d.series_id AND e.price IS NOT NULL)
     """),
]


def main():
    cfg = load_config()
    retrain = set(cfg["datasets"]["m5"]["windows"]["retrain_folds"])
    failures = 0
    with connect(cfg) as conn, conn.cursor() as cur:
        for name, sql in CHECKS:
            cur.execute(sql)
            bad = cur.fetchone()[0]
            status = "PASS" if bad == 0 else f"FAIL ({bad})"
            if bad:
                failures += 1
            print(f"{status:10s} {name}")
        cur.execute("SELECT fold FROM mart.folds WHERE is_retrain ORDER BY fold")
        got = {r[0] for r in cur.fetchall()}
        status = "PASS" if got == retrain else f"FAIL ({sorted(got)})"
        if got != retrain:
            failures += 1
        print(f"{status:10s} folds: retrain flags match config {sorted(retrain)}")
    print(f"checks failed: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
