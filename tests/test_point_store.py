"""ST4-03 point-store tests. Torch-free: these run in the main suite
(the 2-step execution rule applies only to torch-importing test files)."""

import json

import numpy as np
import pandas as pd
import pytest

from core_pipeline.forecast import point_store, prediction_store
from core_pipeline.forecast.common import QUANTILES
from core_pipeline.forecast.tsmixer.config import HORIZON, QUANTILES_9


def _frame(n_series: int = 2, n_origins: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sids, origins, hs = [], [], []
    for s in range(n_series):
        for o in range(n_origins):
            sids += [f"s{s}"] * HORIZON
            origins += [1600 + o * 28] * HORIZON
            hs += list(range(1, HORIZON + 1))
    yhat = rng.uniform(0.0, 5.0, size=len(sids))
    return point_store.build_point_frame(
        np.array(sids), np.array(origins), np.array(hs), yhat)


def test_point_roundtrip_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PRED_DIR", tmp_path)
    df = _frame()
    entry = point_store.write_point_fold(
        "tsmixer_a1", 1, df, "mse_point", {"spec_hash": "x"})
    assert entry["schema_version"] == "pred_point_v1"
    assert entry["prediction_kind"] == "mse_point"
    assert entry["rows"] == len(df) and entry["spec_hash"] == "x"
    back = point_store.load_point_fold("tsmixer_a1", 1)
    assert list(back.columns) == list(df.columns) and len(back) == len(df)
    path = point_store.write_point_manifest(
        "tsmixer_a1", [entry], {"device": "cpu"}, "mse_point")
    assert path.exists()


def test_point_rejects_quantile_columns():
    df = _frame()
    df["q0.50"] = df["yhat"]
    with pytest.raises(ValueError, match="quantile columns"):
        point_store.validate_point_frame(df, "cost_weighted_point")


def test_point_rejects_bad_keys_and_kind():
    df = _frame()
    with pytest.raises(ValueError, match="prediction_kind"):
        point_store.validate_point_frame(df, "median")
    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        point_store.validate_point_frame(dup, "mse_point")
    bad = df.copy()
    bad.loc[0, "target_idx"] = bad.loc[0, "target_idx"] + 1
    with pytest.raises(ValueError, match="target_idx"):
        point_store.validate_point_frame(bad, "mse_point")


def test_point_rejects_incomplete_horizons_and_values():
    df = _frame()
    with pytest.raises(ValueError, match="full h=1"):
        point_store.validate_point_frame(df.iloc[:-1], "mse_point")
    neg = df.copy()
    neg.loc[0, "yhat"] = -0.1
    with pytest.raises(ValueError, match="negative"):
        point_store.validate_point_frame(neg, "mse_point")
    nan = df.copy()
    nan.loc[0, "yhat"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        point_store.validate_point_frame(nan, "mse_point")


def test_mutual_schema_rejection():
    # a point frame fails the quantile validator (quantile columns missing)
    df = _frame()
    with pytest.raises(ValueError, match="missing columns"):
        prediction_store.validate_frame(df, QUANTILES)
    # a quantile frame fails the point validator (quantile columns present)
    n = HORIZON
    q = np.sort(np.random.default_rng(1).uniform(0, 5, size=(n, len(QUANTILES))), axis=1)
    qdf = prediction_store.build_frame(
        np.array(["s0"] * n), np.full(n, 1600), np.arange(1, n + 1), q, QUANTILES)
    with pytest.raises(ValueError, match="quantile columns"):
        point_store.validate_point_frame(qdf, "mse_point")


def test_quantile_manifest_decile9(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PRED_DIR", tmp_path)
    path = point_store.write_quantile_manifest(
        "tsmixer_a3", [], {"device": "cpu"}, QUANTILES_9, "decile9")
    meta = json.loads(path.read_text())
    assert meta["quantiles"] == list(QUANTILES_9)
    assert meta["quantile_set_id"] == "decile9"
    assert meta["point_forecast"] == "q0.50"  # genuine median claim for A3
