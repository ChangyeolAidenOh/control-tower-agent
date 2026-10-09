"""Stage 3-B tuning runner (decision ST3-03). Run ONCE.

Steps 1-2 of the fixed protocol: budgeted prefix of a pre-generated
random-search order scored on representative quantiles {0.1, 0.5, 0.9},
then the winning params train all 11 quantiles once to fix per-quantile
best_iteration on the pre-fold-1 split (valid targets d_1438..d_1605).
Fold retraining (steps 3-4) lives in a separate runner that only LOADS
the JSON this script writes -- commit outputs/stage3_lgbm_tuning.json
before any fold training.

Trial-count rule (logged, never performance-based): measure trial 1,
then n_run = min(max_trials, max(1, floor(budget_seconds / t_trial1))).
--measure-only stops after trial 1 and prints the rule inputs.

Run: python -m scripts.tune_lgbm [--measure-only] [--budget-seconds 10800]
"""

import argparse
import resource
import sys
import time

from core_pipeline.data import db
from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)
from core_pipeline.forecast.metrics import wql

OUT_PATH = "outputs/stage3_lgbm_tuning.json"


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def run_trial(params, train_ds, valid_frame, valid_ds, args):
    scores_preds = []
    iters = {}
    for tau in lm.REP_QUANTILES:
        booster, used = lm.train_quantile(
            train_ds, tau, params, args.num_boost_round_max,
            num_threads=args.num_threads, valid_ds=valid_ds,
            early_stopping_rounds=args.early_stopping,
        )
        preds = booster.predict(valid_frame.X, num_iteration=used)
        scores_preds.append(preds)
        iters[str(tau)] = used
    import numpy as np
    score = wql(valid_frame.y, np.column_stack(scores_preds),
                lm.REP_QUANTILES)
    return score, iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trials", type=int, default=30)
    ap.add_argument("--budget-seconds", type=int, default=10800)
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--num-boost-round-max", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=100)
    args = ap.parse_args()

    db.load_dotenv()
    conn = db.connect(db.load_config())
    cfg = DatasetConfig()
    builder = FoldDatasetBuilder(cfg, MartPanelSource(conn))

    t0 = time.time()
    train_frame, valid_frame = builder.tuning_split()
    print(f"frames train={len(train_frame.X)} valid={len(valid_frame.X)} "
          f"build_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")

    t0 = time.time()
    train_ds = lm.make_dataset(train_frame)
    valid_ds = lm.make_dataset(valid_frame, reference=train_ds)
    train_ds.construct()
    valid_ds.construct()
    print(f"lgb datasets construct_s={time.time() - t0:.0f} "
          f"rss_mb={rss_mb():.0f}")

    trials = lm.sample_trials(args.max_trials, lm.SEARCH_SEED)
    results = []

    t0 = time.time()
    score, iters = run_trial(trials[0], train_ds, valid_frame, valid_ds, args)
    t_trial1 = time.time() - t0
    results.append({"trial": 1, "params": trials[0], "score": score,
                    "rep_iters": iters})
    n_run = min(args.max_trials,
                max(1, int(args.budget_seconds // max(t_trial1, 1e-9))))
    print(f"trial 1 score={score:.5f} t_s={t_trial1:.0f} "
          f"rss_mb={rss_mb():.0f}")
    print(f"budget rule: n_run = min({args.max_trials}, "
          f"floor({args.budget_seconds}/{t_trial1:.0f})) = {n_run}")
    if args.measure_only:
        print("measure-only: stopping after trial 1")
        return

    for i in range(1, n_run):
        t0 = time.time()
        score, iters = run_trial(trials[i], train_ds, valid_frame,
                                 valid_ds, args)
        results.append({"trial": i + 1, "params": trials[i], "score": score,
                        "rep_iters": iters})
        print(f"trial {i + 1}/{n_run} score={score:.5f} "
              f"t_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")

    best = lm.select_best_trial(results)
    print(f"best trial={best['trial']} score={best['score']:.5f}")

    boosters = {}
    best_iterations = {}
    for tau in lm.QUANTILES:
        booster, used = lm.train_quantile(
            train_ds, tau, best["params"], args.num_boost_round_max,
            num_threads=args.num_threads, valid_ds=valid_ds,
            early_stopping_rounds=args.early_stopping,
        )
        boosters[tau] = booster
        best_iterations[str(tau)] = used
    final_wql, crossing = lm.valid_wql(valid_frame, boosters, lm.QUANTILES)
    print(f"final 11q valid_wql={final_wql:.5f} "
          f"crossing_rate={crossing:.4f} rss_mb={rss_mb():.0f}")

    lm.write_json(OUT_PATH, {
        "decision": "ST3-03",
        "search_seed": lm.SEARCH_SEED,
        "train_seed": lm.TRAIN_SEED,
        "quantiles": list(lm.QUANTILES),
        "rep_quantiles": list(lm.REP_QUANTILES),
        "max_trials": args.max_trials,
        "budget_seconds": args.budget_seconds,
        "t_trial1_seconds": t_trial1,
        "n_run": n_run,
        "num_boost_round_max": args.num_boost_round_max,
        "early_stopping_rounds": args.early_stopping,
        "num_threads": args.num_threads,
        "trials": results,
        "best_trial": best["trial"],
        "best_params": best["params"],
        "best_iterations": best_iterations,
        "valid_wql_11q_rearranged": final_wql,
        "valid_crossing_rate_raw": crossing,
        "rows": {"train": len(train_frame.X), "valid": len(valid_frame.X)},
        "spec_hash": train_frame.spec_hash,
        "encoder_hash": train_frame.encoder_hash,
        "runtime": lm.runtime_manifest(),
    })
    print(f"wrote {OUT_PATH} -- commit this file before fold training")
    conn.close()


if __name__ == "__main__":
    main()
