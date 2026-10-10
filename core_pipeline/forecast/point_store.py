"""Single-output prediction store (ST4-03, P0 output semantics).

A1/A2/B predictions are one number per (series, origin, horizon) whose
statistical meaning differs by training objective (mse_point /
cost_weighted_point / decision_focused_point). They are stored as a `yhat`
column with an explicit prediction_kind and are never written into quantile
columns; only A3's q0.50 is a median claim.

The Stage 3 quantile store (prediction_store) is left untouched and its
frozen 11-quantile parquets are unchanged. The two schemas reject each
other's frames by construction: validate_point_frame refuses any quantile
column, and prediction_store.validate_frame requires the full quantile set,
so a cross-load fails validation on either side.

Torch-free; the tests for this module run inside the main suite.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.forecast import prediction_store as quantile_store
from core_pipeline.forecast.common import file_sha256, write_json
from core_pipeline.forecast.tsmixer.config import HORIZON, POINT_KINDS

POINT_SCHEMA_VERSION = "pred_point_v1"
_QUANTILE_COL_RE = re.compile(r"^q0\.\d+$")


def build_point_frame(
    series_id: np.ndarray,
    origin_idx: np.ndarray,
    horizon: np.ndarray,
    yhat: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "series_id": series_id,
        "origin_idx": np.asarray(origin_idx, dtype=np.int64),
        "horizon": np.asarray(horizon, dtype=np.int64),
    })
    df["target_idx"] = df["origin_idx"] + df["horizon"]
    df["yhat"] = np.asarray(yhat, dtype=np.float32)
    return df


def validate_point_frame(df: pd.DataFrame, prediction_kind: str) -> dict:
    if prediction_kind not in POINT_KINDS:
        raise ValueError(f"unknown prediction_kind {prediction_kind!r}")
    qcols = [c for c in df.columns if _QUANTILE_COL_RE.match(str(c))]
    if qcols:
        raise ValueError(f"point frame carries quantile columns: {qcols}")
    required = ["series_id", "origin_idx", "horizon", "target_idx", "yhat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df.duplicated(["series_id", "origin_idx", "horizon"]).any():
        raise ValueError("duplicate (series_id, origin_idx, horizon) keys")
    if not (df["target_idx"] == df["origin_idx"] + df["horizon"]).all():
        raise ValueError("target_idx != origin_idx + horizon")
    per_pair = df.groupby(["origin_idx", "series_id"])["horizon"]
    if not ((per_pair.nunique() == HORIZON) & (per_pair.count() == HORIZON)).all():
        raise ValueError(f"some (origin, series) lacks the full h=1..{HORIZON} set")
    y = df["yhat"].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("non-finite prediction values")
    if (y < 0).any():
        # shared softplus output transform guarantees this; a violation
        # means a pipeline bug upstream (ST4-02)
        raise ValueError("negative yhat values")
    return {
        "rows": int(len(df)),
        "series": int(df["series_id"].nunique()),
        "origins": int(df["origin_idx"].nunique()),
    }


def write_point_fold(model: str, fold: int, df: pd.DataFrame,
                     prediction_kind: str, extra_meta: dict) -> dict:
    """Validate, write parquet, reload-check; returns the manifest entry."""
    stats = validate_point_frame(df, prediction_kind)
    quantile_store.PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = quantile_store.PRED_DIR / f"{model}_f{fold}.parquet"
    df.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    if not quantile_store._frames_equal(df.reset_index(drop=True), back):
        raise ValueError(f"reload mismatch for {path}")
    return {
        "fold": fold,
        "file": path.name,
        "prediction_file_sha256": file_sha256(path),
        "schema_version": POINT_SCHEMA_VERSION,
        "prediction_kind": prediction_kind,
        **stats,
        **extra_meta,
    }


def write_point_manifest(model: str, entries: list[dict], runtime: dict,
                         prediction_kind: str) -> Path:
    if prediction_kind not in POINT_KINDS:
        raise ValueError(f"unknown prediction_kind {prediction_kind!r}")
    path = quantile_store.PRED_DIR / f"{model}_manifest.json"
    write_json(path, {
        "model": model,
        "schema_version": POINT_SCHEMA_VERSION,
        "prediction_kind": prediction_kind,
        "point_forecast": "yhat",
        "folds": entries,
        "runtime": runtime,
    })
    return path


def load_point_fold(model: str, fold: int) -> pd.DataFrame:
    return pd.read_parquet(quantile_store.PRED_DIR / f"{model}_f{fold}.parquet")


def write_quantile_manifest(model: str, entries: list[dict], runtime: dict,
                            quantiles: tuple[float, ...],
                            quantile_set_id: str) -> Path:
    """Parameterized counterpart of prediction_store.write_manifest for the
    A3 9-quantile model: records the actual grid and a quantile_set_id
    (ST4-01 section 8, "decile9") instead of the hardcoded 11-quantile
    metadata. For a quantile_grid model q0.50 is a genuine median claim, so
    point_forecast stays "q0.50". Stage 3 manifests are not rewritten."""
    path = quantile_store.PRED_DIR / f"{model}_manifest.json"
    write_json(path, {
        "model": model,
        "schema_version": quantile_store.SCHEMA_VERSION,
        "quantiles": list(quantiles),
        "quantile_set_id": quantile_set_id,
        "point_forecast": "q0.50",
        "folds": entries,
        "runtime": runtime,
    })
    return path
