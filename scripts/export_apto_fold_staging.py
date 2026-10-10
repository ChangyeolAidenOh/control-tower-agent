"""Export A-PTO fold-retraining staging (folds 1/4/7/10) for Colab GPU
(ST6-12 (6)): wide training frames + per-fold series scales.

Writes outputs/apto_staging/fold_train_f{K}.parquet,
scales_f{K}.json, fold_staging_manifest.json.

Run: python -m scripts.export_apto_fold_staging
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import apto
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource, RETRAIN_FOLDS,
)

OUT_DIR = Path("outputs/apto_staging")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(frame: apto.APTOFrame) -> pd.DataFrame:
    df = frame.X.copy()
    for j in range(apto.N_HORIZONS):
        df[f"y_raw_h{j + 1}"] = frame.y[:, j]
    df["series_id"] = frame.meta["series_id"].to_numpy()
    df["origin_idx"] = frame.meta["origin_idx"].to_numpy()
    return df


def main() -> None:
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    cfg = DatasetConfig(include_series_id=False)
    builder = FoldDatasetBuilder(cfg, source)
    folds_tbl = source.load_folds().set_index("fold")

    manifest: dict = {"protocol": "ST6-12 (6)", "folds": {},
                      "files_sha256": {}}
    for k in RETRAIN_FOLDS:
        train_end = int(folds_tbl.loc[k, "train_end_idx"])
        fr = apto.to_wide(builder.training_frame(k), cfg)
        omax = int(fr.meta["origin_idx"].max())
        assert omax + 28 <= train_end, (k, omax, train_end)
        scales = apto.series_scales(source, train_end)
        p = OUT_DIR / f"fold_train_f{k}.parquet"
        pack(fr).to_parquet(p, index=False)
        sp = OUT_DIR / f"scales_f{k}.json"
        sp.write_text(json.dumps(
            {s: round(v, 6) for s, v in sorted(scales.items())}, indent=1))
        manifest["folds"][str(k)] = {
            "train_end_idx": train_end, "rows": len(fr.X),
            "origin_max": omax, "spec_hash": fr.spec_hash,
            "encoder_hash": fr.encoder_hash,
        }
        manifest["files_sha256"][p.name] = file_sha256(p)
        manifest["files_sha256"][sp.name] = file_sha256(sp)
        print(f"fold {k}: rows={len(fr.X)} train_end={train_end} "
              f"origin_max={omax}")

    (OUT_DIR / "fold_staging_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    print("written fold_staging_manifest.json")


if __name__ == "__main__":
    main()