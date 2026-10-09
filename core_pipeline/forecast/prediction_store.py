"""Shared prediction storage contract (Stage 3, decision ST3-05).

Both forecast runners (LightGBM, Chronos-2) write per-fold parquet files
with one schema so 3-D (ACI), 3-E (cumulative intervals) and 3-F
(reporting) read predictions without touching model code.

Path: outputs/predictions/{model}_f{fold}.parquet
Columns: series_id, origin_idx, horizon, target_idx,
         q0.05, q0.10, ..., q0.90, q0.95   (rearranged, non-decreasing)
Point forecast = q0.50. No `mean` column: Chronos-2's `predictions`
field is the 0.5 quantile under an internal name, so storing it as a
mean would assign a wrong statistical meaning.

Storage vs scoring are separate contracts:
- storage keeps ALL h = 1..28 for every origin (fold 12 included;
  targets past d_1941 have no truth and are stored for planning use);
- scoring (3-F) applies the shared evaluation mask: per fold, rows with
  target_idx inside that fold's test window. The mask is identical for
  every model. Frozen before 3-D/3-E (ST3-05).
- 3-D ACI never reads future truth out of these files ahead of time;
  the 28-day observation lag governs when an error may update state.

Validations on write (all raise on violation):
  1. (series_id, origin_idx, horizon) unique
  2. target_idx == origin_idx + horizon
  3. every (origin, series) carries exactly horizons 1..28
  4. quantile columns non-decreasing left to right
  5. no NULL / non-finite prediction values
  6. post-write reload equality (values + sha256)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core_pipeline.forecast.common import QUANTILES, file_sha256, write_json

SCHEMA_VERSION = "pred_v1"
PRED_DIR = Path("outputs/predictions")


def quantile_columns(quantiles: Sequence[float] = QUANTILES) -> list[str]:
    return [f"q{q:.2f}" for q in quantiles]


def crossing_stats(raw: np.ndarray) -> dict:
    """Pre-rearrangement adjacency violations over 11 quantile columns:
    share of rows with any violation and the maximum violation gap."""
    diffs = np.diff(np.asarray(raw, dtype=np.float64), axis=1)
    viol = diffs < 0
    rate = float(viol.any(axis=1).mean()) if len(raw) else 0.0
    max_gap = float(np.maximum(0.0, -diffs).max()) if len(raw) else 0.0
    return {"crossing_rate_before": rate, "crossing_max_gap_before": max_gap}


def build_frame(
    series_id: np.ndarray,
    origin_idx: np.ndarray,
    horizon: np.ndarray,
    rearranged: np.ndarray,
    quantiles: Sequence[float] = QUANTILES,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "series_id": series_id,
        "origin_idx": np.asarray(origin_idx, dtype=np.int64),
        "horizon": np.asarray(horizon, dtype=np.int64),
    })
    df["target_idx"] = df["origin_idx"] + df["horizon"]
    for j, col in enumerate(quantile_columns(quantiles)):
        df[col] = np.asarray(rearranged[:, j], dtype=np.float32)
    return df


def validate_frame(df: pd.DataFrame,
                   quantiles: Sequence[float] = QUANTILES) -> dict:
    qcols = quantile_columns(quantiles)
    required = ["series_id", "origin_idx", "horizon", "target_idx"] + qcols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df.duplicated(["series_id", "origin_idx", "horizon"]).any():
        raise ValueError("duplicate (series_id, origin_idx, horizon) keys")
    if not (df["target_idx"] == df["origin_idx"] + df["horizon"]).all():
        raise ValueError("target_idx != origin_idx + horizon")
    per_pair = df.groupby(["origin_idx", "series_id"])["horizon"]
    if not ((per_pair.nunique() == 28) & (per_pair.count() == 28)).all():
        raise ValueError("some (origin, series) lacks the full h=1..28 set")
    qm = df[qcols].to_numpy(dtype=np.float64)
    if not np.isfinite(qm).all():
        raise ValueError("non-finite prediction values")
    if (np.diff(qm, axis=1) < 0).any():
        raise ValueError("quantile columns are not non-decreasing")
    return {
        "rows": int(len(df)),
        "series": int(df["series_id"].nunique()),
        "origins": int(df["origin_idx"].nunique()),
    }


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for col in a.columns:
        va, vb = a[col].to_numpy(), b[col].to_numpy()
        ok = (np.array_equal(va, vb) if va.dtype.kind in "OUi"
              else np.array_equal(va, vb, equal_nan=True))
        if not ok:
            return False
    return True


def write_fold(model: str, fold: int, df: pd.DataFrame,
               extra_meta: dict,
               quantiles: Sequence[float] = QUANTILES) -> dict:
    """Validate, write parquet, reload-check; returns the manifest entry."""
    stats = validate_frame(df, quantiles)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{model}_f{fold}.parquet"
    df.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    if not _frames_equal(df.reset_index(drop=True), back):
        raise ValueError(f"reload mismatch for {path}")
    return {
        "fold": fold,
        "file": path.name,
        "prediction_file_sha256": file_sha256(path),
        "schema_version": SCHEMA_VERSION,
        "quantile_rearranged": True,
        **stats,
        **extra_meta,
    }


def write_manifest(model: str, entries: list[dict], runtime: dict) -> Path:
    path = PRED_DIR / f"{model}_manifest.json"
    write_json(path, {
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "quantiles": list(QUANTILES),
        "point_forecast": "q0.50",
        "folds": entries,
        "runtime": runtime,
    })
    return path


def load_fold(model: str, fold: int) -> pd.DataFrame:
    return pd.read_parquet(PRED_DIR / f"{model}_f{fold}.parquet")


def covariate_schema_hash(columns: Sequence[str], dtypes: Sequence[str]) -> str:
    payload = json.dumps(sorted(zip(map(str, columns), map(str, dtypes))))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
