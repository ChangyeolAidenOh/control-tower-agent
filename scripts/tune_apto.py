"""Stage 6 A-PTO tuning runner (decision ST6-11). Run ONCE.

CatBoost MultiRMSE (28-horizon multi-output) on the Stage 3 pre-fold-1
tuning split (train targets <= d_1437, valid targets d_1438..d_1605),
series-scaled targets, series ID excluded. Fixed 27-combination grid
(depth x learning_rate x l2_leaf_reg), seed-20261011 permutation; the
trial-count rule is mechanical and never performance-based:

  measure trial 1, then n_run = min(27, max(1, floor(T_budget / t1)))

T_budget = 1866.3 s, the measured wall-clock sum of the 24 TSMixer
tuning trials (outputs/stage4_tsmixer_tuning.json grid_run sec_total) --
the preregistration SS4 "same tuning budget" fixed numerically.
--measure-only stops after trial 1 and prints the rule inputs.

Fold retraining (a separate runner) only LOADS the JSON this script
writes; commit outputs/stage6_apto_tuning.json before any fold training.

Run: python -m scripts.tune_apto [--measure-only]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import resource
import sys
import time

import numpy as np
from catboost import CatBoostRegressor

from core_pipeline.data import db
from core_pipeline.forecast import apto
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)

OUT_PATH = "outputs/stage6_apto_tuning.json"
T_BUDGET_S = 1866.3
GRID_SEED = 20261011
MODEL_SEED = 20261011
GRID = {
    "depth": (6, 8, 10),
    "learning_rate": (0.03, 0.06, 0.1),
    "l2_leaf_reg": (1.0, 3.0, 9.0),
}


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def grid_order() -> list[dict]:
    combos = [
        {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
        for d, lr, l2 in itertools.product(
            GRID["depth"], GRID["learning_rate"], GRID["l2_leaf_reg"])
    ]
    rng = np.random.default_rng(GRID_SEED)
    perm = rng.permutation(len(combos))
    return [{"trial_id": int(i), **combos[int(i)]} for i in perm]


def run_trial(params: dict, Xtr, ytr, Xva, yva, cat_cols, args):
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=args.max_iters,
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_seed=MODEL_SEED,
        thread_count=args.num_threads,
        early_stopping_rounds=args.early_stopping,
        use_best_model=True,
        verbose=0,
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, cat_features=list(cat_cols),
              eval_set=(Xva, yva))
    sec = time.perf_counter() - t0
    best_iter = int(model.get_best_iteration())
    score = float(model.get_best_score()["validation"]["MultiRMSE"])
    return score, best_iter, sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--num-threads", type=int, default=6)
    args = ap.parse_args()

    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    cfg = DatasetConfig(include_series_id=False)
    builder = FoldDatasetBuilder(cfg, source)

    t0 = time.perf_counter()
    tr_long, va_long = builder.tuning_split()
    tr = apto.to_wide(tr_long, cfg)
    va = apto.to_wide(va_long, cfg)
    scales = apto.series_scales(source, cfg.tune_train_target_end)
    ytr = apto.scale_targets(tr.y, tr.meta, scales)
    yva = apto.scale_targets(va.y, va.meta, scales)
    prep_s = time.perf_counter() - t0
    print(f"frames: train={len(tr.X)} valid={len(va.X)} "
          f"features={len(tr.X.columns)} prep_s={prep_s:.1f} "
          f"dropped_rows tr={tr.n_dropped_incomplete} "
          f"va={va.n_dropped_incomplete}")
    omax_tr = int(tr.meta["origin_idx"].max())
    omax_va = int(va.meta["origin_idx"].max())
    assert omax_tr + 28 <= cfg.tune_train_target_end, omax_tr
    assert omax_va + 28 <= cfg.tune_valid_target_end, omax_va

    order = grid_order()
    results = []
    score1, it1, t1 = run_trial(order[0], tr.X, ytr, va.X, yva,
                                tr.categorical, args)
    results.append({"trial": order[0]["trial_id"], "order_pos": 0,
                    **order[0], "score": score1, "best_iteration": it1,
                    "sec_total": round(t1, 1)})
    n_run = min(len(order), max(1, int(T_BUDGET_S // t1)))
    print(f"trial 1: score={score1:.6f} best_iter={it1} t1={t1:.1f}s "
          f"-> n_run={n_run} (T_budget={T_BUDGET_S}s)")
    if args.measure_only:
        print("measure-only: stopping before the remaining trials")
        return

    for pos in range(1, n_run):
        p = order[pos]
        score, bi, sec = run_trial(p, tr.X, ytr, va.X, yva,
                                   tr.categorical, args)
        results.append({"trial": p["trial_id"], "order_pos": pos, **p,
                        "score": score, "best_iteration": bi,
                        "sec_total": round(sec, 1)})
        print(f"trial {pos + 1}/{n_run}: id={p['trial_id']} "
              f"score={score:.6f} best_iter={bi} sec={sec:.1f}")

    finite = [r for r in results if np.isfinite(r["score"])]
    if not finite:
        raise RuntimeError("no trial produced a finite score")
    best = min(finite, key=lambda r: (r["score"], r["order_pos"]))
    print(f"selected trial_id={best['trial_id']} score={best['score']:.6f} "
          f"best_iteration={best['best_iteration']}")

    out = {
        "protocol": "ST6-11",
        "model": "catboost_multirmse_h28",
        "grid": {k: list(v) for k, v in GRID.items()},
        "grid_seed": GRID_SEED,
        "model_seed": MODEL_SEED,
        "execution_order": [o["trial_id"] for o in order],
        "budget_rule": {
            "T_budget_s": T_BUDGET_S,
            "budget_source": ("stage4_tsmixer_tuning.json grid_run "
                              "sec_total sum of 24 trials"),
            "t_trial1_s": round(t1, 1),
            "n_run": n_run,
        },
        "max_iters": args.max_iters,
        "early_stopping": args.early_stopping,
        "num_threads": args.num_threads,
        "scaling": {
            "rule": "per-series max(1, mean demand first_active..cutoff)",
            "cutoff": cfg.tune_train_target_end,
            "values": {k: round(v, 6) for k, v in sorted(scales.items())},
        },
        "frames": {
            "train_rows": len(tr.X), "valid_rows": len(va.X),
            "n_features": len(tr.X.columns),
            "feature_columns": list(tr.X.columns),
            "categorical": list(tr.categorical),
            "train_origin_max": omax_tr, "valid_origin_max": omax_va,
        },
        "spec_hash": tr.spec_hash,
        "encoder_hash": tr.encoder_hash,
        "trials": results,
        "selected": {k: best[k] for k in
                     ("trial_id", "depth", "learning_rate", "l2_leaf_reg",
                      "score", "best_iteration")},
        "environment": {
            "python": platform.python_version(),
            "peak_rss_mb": round(rss_mb(), 1),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()