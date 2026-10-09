"""Generate the three Stage 3-E cumulative-demand prediction sets
(ST3-08) into the shared store, one file set per method:

  lgbm_lt3   (A) direct lt_cum model, sha-verified against its manifest
  lt3_como   (B) comonotonic-coupling quantile summation from the daily
                 lgbm parquet (assumption, NOT a universal bound)
  lt3_indep  (C) independence + truncated inverse-CDF Monte Carlo
                 (exploratory; conditional on the tail-truncation rule)

All files carry target_kind="cumulative_demand" so they can never be
joined with daily per-horizon files by accident. Storage covers every
test origin; the evaluation mask (target_end_idx <= 1941) is applied by
scripts.eval_lt3, identically for all three methods.

Run: python -m scripts.predict_lt3
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core_pipeline.data import db
from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast import lt3
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.dataset import (
    DatasetConfig, DemandHistory, FoldDatasetBuilder, MartPanelSource,
    retrain_fold_for,
)

MODELS_DIR = Path("outputs/models")
LT_MANIFEST = Path("outputs/stage3_lgbm_lt3_manifest.json")


def load_boosters(retrain_fold: int, model_hashes: dict) -> dict:
    boosters = {}
    for tau in lm.QUANTILES:
        name = f"lgbm_lt3_f{retrain_fold}_q{tau}.txt"
        path = MODELS_DIR / name
        if lm.file_sha256(path) != model_hashes[name]["sha256"]:
            raise SystemExit(f"model hash mismatch: {name}")
        boosters[tau] = lgb.Booster(model_file=str(path))
    return boosters


def main() -> None:
    manifest = json.loads(LT_MANIFEST.read_text())
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    builder = FoldDatasetBuilder(DatasetConfig(), source)

    entries = {"lgbm_lt3": [], "lt3_como": [], "lt3_indep": []}
    cached_rf, boosters = None, None
    for fold in range(1, 13):
        t0 = time.time()
        rf = retrain_fold_for(fold)
        if rf != cached_rf:
            boosters = load_boosters(rf, manifest["models"])
            cached_rf = rf
        sids, orgs, raws, fixed = [], [], [], []
        spec_hash = encoder_hash = None
        for origin in builder.iter_test_origins(fold):
            hist = DemandHistory.from_source(source, max_period_idx=origin)
            frame = builder.inference_frame_lt(hist, origin)
            raw, rearranged, _ = lm.predict_matrix(boosters, frame.X)
            sids.append(frame.meta["series_id"].to_numpy())
            orgs.append(frame.meta["origin_idx"].to_numpy())
            raws.append(raw)
            fixed.append(rearranged)
            spec_hash, encoder_hash = frame.spec_hash, frame.encoder_hash
        df_a = store.build_lt_frame(
            np.concatenate(sids), np.concatenate(orgs),
            np.concatenate(fixed))
        entries["lgbm_lt3"].append(store.write_lt_fold(
            "lgbm_lt3", fold, df_a, {
                "retrain_fold": rf,
                "spec_hash": spec_hash, "encoder_hash": encoder_hash,
                **store.crossing_stats(np.concatenate(raws)),
            }))

        daily = store.load_fold("lgbm", fold)
        entries["lt3_como"].append(store.write_lt_fold(
            "lt3_como", fold, lt3.comonotone_from_daily(daily), {
                "source": f"lgbm_f{fold}.parquet (rearranged daily "
                          "marginals, h=1..3)",
                "coupling": "comonotonic assumption; not a universal "
                            "pointwise quantile bound",
            }))
        entries["lt3_indep"].append(store.write_lt_fold(
            "lt3_indep", fold, lt3.independent_mc_from_daily(daily), {
                "source": f"lgbm_f{fold}.parquet (rearranged daily "
                          "marginals, h=1..3)",
                "coupling": "conditional independence (exploratory)",
                "mc_n": lt3.MC_N, "mc_seed": lt3.MC_SEED,
                "tail_rule": "inverse CDF truncated at q0.05/q0.95; "
                             "results conditional on this rule",
            }))
        print(f"fold {fold} rf={rf} rows_each={len(df_a)} "
              f"t_s={time.time() - t0:.0f}")

    for model, ents in entries.items():
        extra = {"lt_manifest_sha256": lm.file_sha256(LT_MANIFEST)} \
            if model == "lgbm_lt3" else {}
        path = store.write_manifest(model, ents, {
            "target_kind": store.TARGET_KIND_CUM,
            "aggregation_days": store.LT_AGG_DAYS, **extra,
        })
        print(f"wrote {path}")
    conn.close()


if __name__ == "__main__":
    main()
