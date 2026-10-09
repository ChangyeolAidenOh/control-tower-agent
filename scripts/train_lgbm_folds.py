"""Stage 3-B fold retraining runner (ST3-03 protocol steps 3-4).

Trains 11 quantile models at each retrain fold {1, 4, 7, 10} with the
hyperparameters and PER-QUANTILE FIXED iteration counts loaded from
outputs/stage3_lgbm_tuning.json -- the only tuning artifact, committed
before this script runs. By construction there is no validation set, no
early stopping, and no parameter argument here: evaluation data cannot
influence any training choice (preregistration SS2: retrain yes,
re-tune no).

Guards:
- aborts if the tuning JSON's spec_hash/encoder_hash differ from the
  current builder (feature or encoder drift);
- every saved model is reloaded and checked for prediction equality on a
  sample (max abs diff > 1e-9 fails the run);
- writes outputs/stage3_lgbm_manifest.json with runtime versions, fold
  cutoffs, row counts, model sha256 hashes and the tuning-JSON hash.

Run: python -m scripts.train_lgbm_folds
"""

import json
import resource
import sys
import time
from pathlib import Path

from core_pipeline.data import db
from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast.dataset import (
    RETRAIN_FOLDS, DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)

TUNING_PATH = Path("outputs/stage3_lgbm_tuning.json")
MODELS_DIR = Path("outputs/models")
MANIFEST_PATH = Path("outputs/stage3_lgbm_manifest.json")
RELOAD_TOL = 1e-9


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> None:
    if not TUNING_PATH.exists():
        raise SystemExit(f"{TUNING_PATH} not found -- run scripts.tune_lgbm "
                         "first and commit its output")
    tuning = json.loads(TUNING_PATH.read_text())
    params = tuning["best_params"]
    best_iterations = {float(k): int(v)
                       for k, v in tuning["best_iterations"].items()}
    num_threads = int(tuning["num_threads"])
    quantiles = tuple(float(q) for q in tuning["quantiles"])
    assert quantiles == lm.QUANTILES, "quantile set drift vs lgbm_model"

    db.load_dotenv()
    conn = db.connect(db.load_config())
    builder = FoldDatasetBuilder(DatasetConfig(), MartPanelSource(conn))

    frame_probe = None  # hashes come from the first fold's frame
    folds_meta = {}
    model_files = {}
    worst_reload = 0.0

    for fold in RETRAIN_FOLDS:
        t0 = time.time()
        frame = builder.training_frame(fold)
        if frame_probe is None:
            if (frame.spec_hash != tuning["spec_hash"]
                    or frame.encoder_hash != tuning["encoder_hash"]):
                raise SystemExit(
                    "spec/encoder hash mismatch vs tuning JSON -- "
                    f"frame=({frame.spec_hash},{frame.encoder_hash}) "
                    f"tuning=({tuning['spec_hash']},{tuning['encoder_hash']})")
            frame_probe = (frame.spec_hash, frame.encoder_hash)
        x_sample = frame.X.head(500).copy()
        cutoff = int(frame.meta["target_idx"].max())
        folds_meta[str(fold)] = {"rows": len(frame.X), "train_end": cutoff}
        ds = lm.make_dataset(frame)
        ds.construct()
        del frame
        print(f"fold {fold} rows={folds_meta[str(fold)]['rows']} "
              f"train_end={cutoff} dataset_s={time.time() - t0:.0f} "
              f"rss_mb={rss_mb():.0f}")

        for tau in quantiles:
            t1 = time.time()
            booster, used = lm.train_quantile(
                ds, tau, params, num_boost_round=best_iterations[tau],
                num_threads=num_threads,
            )
            assert used == best_iterations[tau]
            path = MODELS_DIR / f"lgbm_f{fold}_q{tau}.txt"
            sha = lm.save_booster(booster, path)
            diff = lm.reload_max_abs_diff(booster, path, x_sample)
            worst_reload = max(worst_reload, diff)
            if diff > RELOAD_TOL:
                raise SystemExit(
                    f"reload equality failed: {path} max_abs_diff={diff:.3e}")
            model_files[path.name] = {"sha256": sha,
                                      "num_boost_round": used}
            print(f"  q{tau} iters={used} t_s={time.time() - t1:.0f} "
                  f"sha={sha} reload_diff={diff:.1e}")
        del ds
        print(f"fold {fold} done rss_mb={rss_mb():.0f}")

    lm.write_json(MANIFEST_PATH, {
        "decision": "ST3-03",
        "tuning_json_sha256": lm.file_sha256(TUNING_PATH),
        "spec_hash": frame_probe[0],
        "encoder_hash": frame_probe[1],
        "train_seed": lm.TRAIN_SEED,
        "num_threads": num_threads,
        "params": params,
        "best_iterations": tuning["best_iterations"],
        "folds": folds_meta,
        "models": model_files,
        "worst_reload_max_abs_diff": worst_reload,
        "runtime": lm.runtime_manifest(),
    })
    print(f"wrote {MANIFEST_PATH} "
          f"(models={len(model_files)}, worst_reload={worst_reload:.1e})")
    conn.close()


if __name__ == "__main__":
    main()
