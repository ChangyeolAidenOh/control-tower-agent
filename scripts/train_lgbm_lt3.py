"""Train the direct 3-day cumulative-demand quantile model (Stage 3-E
method A, ST3-08).

No hyperparameter search: the daily model's frozen HP (ST3-03 tuning
JSON) is reused as-is. Because the target differs, the per-quantile
boosting iteration is chosen ONCE on the same pre-fold-1 tuning split
(train targets <= d_1437, valid d_1438..d_1605) -- this is a one-shot
model-selection step on the pre-evaluation window, recorded as such.
Fold retraining ({1,4,7,10}) then uses those fixed iterations with no
validation set and no re-selection, exactly like the daily runner.

Run: python -m scripts.train_lgbm_lt3
"""

import json
import resource
import sys
import time
from pathlib import Path

from core_pipeline.data import db
from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast.dataset import (
    TARGET_LT_CUM, DatasetConfig, FoldDatasetBuilder, MartPanelSource,
    RETRAIN_FOLDS,
)
from core_pipeline.forecast.metrics import wql

TUNING_PATH = Path("outputs/stage3_lgbm_tuning.json")
MODELS_DIR = Path("outputs/models")
MANIFEST_PATH = Path("outputs/stage3_lgbm_lt3_manifest.json")
RELOAD_TOL = 1e-9


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> None:
    tuning = json.loads(TUNING_PATH.read_text())
    params = tuning["best_params"]
    num_threads = int(tuning["num_threads"])
    nbr_max = int(tuning["num_boost_round_max"])
    es_rounds = int(tuning["early_stopping_rounds"])

    db.load_dotenv()
    conn = db.connect(db.load_config())
    builder = FoldDatasetBuilder(DatasetConfig(), MartPanelSource(conn))

    # --- one-shot iteration selection on the pre-fold-1 split ----------
    t0 = time.time()
    tr, va = builder.tuning_split(target_mode=TARGET_LT_CUM)
    if (tr.spec_hash != tuning["spec_hash"]
            or tr.encoder_hash != tuning["encoder_hash"]):
        raise SystemExit("spec/encoder hash mismatch vs tuning JSON")
    print(f"lt split train={len(tr.X)} valid={len(va.X)} "
          f"build_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")
    train_ds = lm.make_dataset(tr)
    valid_ds = lm.make_dataset(va, reference=train_ds)

    iterations, select_boosters = {}, {}
    for tau in lm.QUANTILES:
        booster, used = lm.train_quantile(
            train_ds, tau, params, num_boost_round=nbr_max,
            num_threads=num_threads, valid_ds=valid_ds,
            early_stopping_rounds=es_rounds,
        )
        iterations[tau] = used
        select_boosters[tau] = booster
        print(f"  select q{tau} iters={used}")
    _, rearranged, crossing = lm.predict_matrix(select_boosters, va.X)
    split_wql = wql(va.y, rearranged, lm.QUANTILES)
    print(f"lt split 11q wql={split_wql:.5f} crossing={crossing:.4f} "
          f"rss_mb={rss_mb():.0f}")
    del select_boosters, train_ds, valid_ds, tr, va

    # --- fold retraining with fixed iterations --------------------------
    folds_meta, model_files = {}, {}
    worst_reload = 0.0
    for fold in RETRAIN_FOLDS:
        t0 = time.time()
        frame = builder.training_frame(fold, target_mode=TARGET_LT_CUM)
        x_sample = frame.X.head(500).copy()
        folds_meta[str(fold)] = {
            "rows": len(frame.X),
            "train_end": int(frame.meta["target_idx"].max()),
        }
        ds = lm.make_dataset(frame)
        ds.construct()
        del frame
        print(f"fold {fold} rows={folds_meta[str(fold)]['rows']} "
              f"dataset_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")
        for tau in lm.QUANTILES:
            t1 = time.time()
            booster, used = lm.train_quantile(
                ds, tau, params, num_boost_round=iterations[tau],
                num_threads=num_threads,
            )
            assert used == iterations[tau]
            path = MODELS_DIR / f"lgbm_lt3_f{fold}_q{tau}.txt"
            sha = lm.save_booster(booster, path)
            diff = lm.reload_max_abs_diff(booster, path, x_sample)
            worst_reload = max(worst_reload, diff)
            if diff > RELOAD_TOL:
                raise SystemExit(f"reload equality failed: {path}")
            model_files[path.name] = {"sha256": sha,
                                      "num_boost_round": used}
            print(f"  q{tau} iters={used} t_s={time.time() - t1:.0f} "
                  f"sha={sha} reload_diff={diff:.1e}")
        del ds

    lm.write_json(MANIFEST_PATH, {
        "decision": "ST3-08",
        "target_mode": TARGET_LT_CUM,
        "iteration_selection": (
            "no hyperparameter search; per-quantile iteration chosen "
            "once on the pre-fold-1 tuning split (train<=d_1437, valid "
            "d_1438-d_1605); fold retraining reuses fixed iterations"),
        "tuning_json_sha256": lm.file_sha256(TUNING_PATH),
        "best_params": params,
        "iterations": {str(k): v for k, v in iterations.items()},
        "select_split_wql_11q": split_wql,
        "spec_hash": tuning["spec_hash"],
        "encoder_hash": tuning["encoder_hash"],
        "folds": folds_meta,
        "models": model_files,
        **lm.runtime_manifest(),
    })
    print(f"wrote {MANIFEST_PATH} (models={len(model_files)}, "
          f"worst_reload={worst_reload:.1e})")
    conn.close()


if __name__ == "__main__":
    main()
