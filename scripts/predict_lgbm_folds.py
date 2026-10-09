"""Generate LightGBM fold predictions into the shared store (ST3-05).

For every fold 1..12: load the retrain-fold models (sha-verified against
stage3_lgbm_manifest.json), forecast all 11 quantiles at every origin of
the fold's test window (full h=1..28 stored, fold 12 included), apply
the fixed monotone rearrangement, record pre-correction crossing stats,
and write outputs/predictions/lgbm_f{fold}.parquet + manifest.

Scoring is NOT done here; 3-F applies the shared evaluation mask
(targets inside each fold's test window) to these files.

Run: python -m scripts.predict_lgbm_folds
"""

import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core_pipeline.data import db
from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.dataset import (
    DatasetConfig, DemandHistory, FoldDatasetBuilder, MartPanelSource,
    retrain_fold_for,
)

MODELS_DIR = Path("outputs/models")
TRAIN_MANIFEST = Path("outputs/stage3_lgbm_manifest.json")


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def load_boosters(retrain_fold: int, model_hashes: dict) -> dict:
    boosters = {}
    for tau in lm.QUANTILES:
        name = f"lgbm_f{retrain_fold}_q{tau}.txt"
        path = MODELS_DIR / name
        sha = lm.file_sha256(path)
        if sha != model_hashes[name]["sha256"]:
            raise SystemExit(f"model hash mismatch: {name}")
        boosters[tau] = lgb.Booster(model_file=str(path))
    return boosters


def main() -> None:
    manifest = json.loads(TRAIN_MANIFEST.read_text())
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    builder = FoldDatasetBuilder(DatasetConfig(), source)

    entries = []
    cached_rf, boosters = None, None
    for fold in range(1, 13):
        rf = retrain_fold_for(fold)
        if rf != cached_rf:
            boosters = load_boosters(rf, manifest["models"])
            cached_rf = rf
        t0 = time.time()
        sids, origins, horizons, raws, fixed = [], [], [], [], []
        spec_hash = encoder_hash = None
        for origin in builder.iter_test_origins(fold):
            hist = DemandHistory.from_source(source, max_period_idx=origin)
            frame = builder.inference_frame(hist, origin)
            raw, rearranged, _ = lm.predict_matrix(
                boosters, frame.X, lm.QUANTILES)
            sids.append(frame.meta["series_id"].to_numpy())
            origins.append(frame.meta["origin_idx"].to_numpy())
            horizons.append(frame.meta["horizon"].to_numpy())
            raws.append(raw)
            fixed.append(rearranged)
            spec_hash, encoder_hash = frame.spec_hash, frame.encoder_hash
        raw_all = np.concatenate(raws)
        df = store.build_frame(
            np.concatenate(sids), np.concatenate(origins),
            np.concatenate(horizons), np.concatenate(fixed),
        )
        entry = store.write_fold("lgbm", fold, df, {
            "retrain_fold": rf,
            "spec_hash": spec_hash,
            "encoder_hash": encoder_hash,
            **store.crossing_stats(raw_all),
        })
        entries.append(entry)
        print(f"fold {fold} rows={entry['rows']} rf={rf} "
              f"crossing={entry['crossing_rate_before']:.4f} "
              f"t_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")

    path = store.write_manifest("lgbm", entries, lm.runtime_manifest(
        train_manifest_sha256=lm.file_sha256(TRAIN_MANIFEST),
    ))
    print(f"wrote {path}")
    conn.close()


if __name__ == "__main__":
    main()
