"""Export A-PTO tuning staging to parquet for Colab GPU training
(ST6-12: compute relocation only -- same frames, same contract).

Writes outputs/apto_staging/:
  tune_train.parquet / tune_valid.parquet  (wide frames; feature columns
    + y_raw_h1..h28 + series_id, origin_idx)
  scales_tune.json                         (per-series divisor, cutoff 1437)
  staging_manifest.json                    (spec/encoder hash, grid, order,
    budget, sha256 of each parquet)

Run: python -m scripts.export_apto_staging
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import apto
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)

OUT_DIR = Path("outputs/apto_staging")
T_BUDGET_S = 1866.3
GRID_SEED = 20261011
MODEL_SEED = 20261011
GRID = {
    "depth": (6, 8, 10),
    "learning_rate": (0.03, 0.06, 0.1),
    "l2_leaf_reg": (1.0, 3.0, 9.0),
}


def grid_order() -> list[dict]:
    combos = [
        {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
        for d, lr, l2 in itertools.product(
            GRID["depth"], GRID["learning_rate"], GRID["l2_leaf_reg"])
    ]
    rng = np.random.default_rng(GRID_SEED)
    perm = rng.permutation(len(combos))
    return [{"trial_id": int(i), **combos[int(i)]} for i in perm]


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    cfg = DatasetConfig(include_series_id=False)
    builder = FoldDatasetBuilder(cfg, source)

    tr_long, va_long = builder.tuning_split()
    tr = apto.to_wide(tr_long, cfg)
    va = apto.to_wide(va_long, cfg)
    scales = apto.series_scales(source, cfg.tune_train_target_end)

    omax_tr = int(tr.meta["origin_idx"].max())
    omax_va = int(va.meta["origin_idx"].max())
    assert omax_tr + 28 <= cfg.tune_train_target_end, omax_tr
    assert omax_va + 28 <= cfg.tune_valid_target_end, omax_va
    assert len(va.X) == 141 * 154, len(va.X)

    paths = {}
    for name, fr in (("tune_train", tr), ("tune_valid", va)):
        p = OUT_DIR / f"{name}.parquet"
        pack(fr).to_parquet(p, index=False)
        back = pd.read_parquet(p)
        if len(back) != len(fr.X):
            raise RuntimeError(f"reload row mismatch for {p}")
        paths[name] = p

    sp = OUT_DIR / "scales_tune.json"
    sp.write_text(json.dumps(
        {k: round(v, 6) for k, v in sorted(scales.items())}, indent=1))

    manifest = {
        "protocol": "ST6-12",
        "model": "catboost_multirmse_h28",
        "grid": {k: list(v) for k, v in GRID.items()},
        "grid_seed": GRID_SEED,
        "model_seed": MODEL_SEED,
        "execution_order": [o["trial_id"] for o in grid_order()],
        "grid_order_full": grid_order(),
        "T_budget_s": T_BUDGET_S,
        "budget_source": ("stage4_tsmixer_tuning.json grid_run sec_total "
                          "sum of 24 trials (ST6-11)"),
        "feature_columns": list(tr.X.columns),
        "categorical": list(tr.categorical),
        "n_features": len(tr.X.columns),
        "rows": {"tune_train": len(tr.X), "tune_valid": len(va.X)},
        "scaling": {
            "rule": "per-series max(1, mean demand first_active..cutoff)",
            "cutoff": cfg.tune_train_target_end,
            "file": "scales_tune.json",
        },
        "spec_hash": tr.spec_hash,
        "encoder_hash": tr.encoder_hash,
        "files_sha256": {p.name: file_sha256(p) for p in
                         [*paths.values(), sp]},
        "cpu_measurement_record": {
            "trial1_completed": False,
            "wall_clock_s_at_abort": 5700,
            "note": ("M2 CPU 6-thread trial 1 (1000 iters, ES 50) aborted "
                     "~95 min; 3x over T_budget -> compute relocated to "
                     "Colab GPU (ST6-12), grid/order/budget unchanged"),
        },
    }
    (OUT_DIR / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    print(f"written {OUT_DIR}: train={len(tr.X)} valid={len(va.X)} "
          f"features={len(tr.X.columns)}")
    for n, s in manifest["files_sha256"].items():
        print(f"  {n} sha256={s[:16]}...")


if __name__ == "__main__":
    main()