"""Tests for prediction_store.py and chronos_support.py (ST3-05). No DB,
no chronos package required."""

import numpy as np
import pandas as pd
import pytest

from core_pipeline.forecast import chronos_support as cs
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.lgbm_model import QUANTILES

N_Q = len(QUANTILES)


def _toy_frame(n_series=3, origins=(100, 101)):
    rng = np.random.default_rng(0)
    rows = []
    for o in origins:
        for s in range(n_series):
            base = np.sort(rng.uniform(0, 5, (28, N_Q)), axis=1)
            rows.append((np.full(28, f"s{s}"), np.full(28, o),
                         np.arange(1, 29), base))
    sid = np.concatenate([r[0] for r in rows])
    org = np.concatenate([r[1] for r in rows])
    hor = np.concatenate([r[2] for r in rows])
    mat = np.concatenate([r[3] for r in rows])
    return store.build_frame(sid, org, hor, mat)


def test_build_and_validate_ok():
    df = _toy_frame()
    stats = store.validate_frame(df)
    assert stats == {"rows": 168, "series": 3, "origins": 2}
    assert (df["target_idx"] == df["origin_idx"] + df["horizon"]).all()


@pytest.mark.parametrize("breaker", [
    lambda d: d.drop(d.index[5]),                                 # missing h
    lambda d: pd.concat([d, d.iloc[[0]]]),                        # dup key
    lambda d: d.assign(target_idx=d["target_idx"] + 1),           # misalign
    lambda d: d.assign(**{"q0.95": d["q0.05"] - 1.0}),            # crossing
    lambda d: d.assign(**{"q0.50": np.nan}),                      # null
])
def test_validate_rejects(breaker):
    with pytest.raises(ValueError):
        store.validate_frame(breaker(_toy_frame()))


def test_write_fold_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PRED_DIR", tmp_path)
    df = _toy_frame()
    entry = store.write_fold("toy", 1, df, {"spec_hash": "x"})
    assert entry["rows"] == len(df) and entry["fold"] == 1
    assert len(entry["prediction_file_sha256"]) == 16
    back = store.load_fold("toy", 1)
    assert len(back) == len(df)


def test_crossing_stats():
    raw = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0]])
    s = store.crossing_stats(raw)
    assert s["crossing_rate_before"] == pytest.approx(0.5)
    assert s["crossing_max_gap_before"] == pytest.approx(1.0)


def test_idx_date_roundtrip():
    idx = np.array([1, 2, 1941, 1969])
    dates = cs.idx_to_date(idx)
    assert str(dates[0]) == "2011-01-29"
    back = cs.date_to_idx(pd.Series(pd.to_datetime(dates)))
    np.testing.assert_array_equal(back, idx)


def test_find_quantile_column():
    assert cs.find_quantile_column(["0.05", "predictions"], 0.05) == "0.05"
    assert cs.find_quantile_column(["0.5"], 0.5) == "0.5"
    with pytest.raises(KeyError):
        cs.find_quantile_column(["foo"], 0.1)


def _toy_panel(origin=50, horizon=5, n_series=2):
    rows = []
    for s in range(n_series):
        for d in range(1, origin + horizon + 1):
            rows.append({
                "series_id": f"s{s}", "period_idx": d,
                "target": float(d % 4) if d <= origin else np.nan,
                "wday": (d - 1) % 7 + 1, "month": 1, "snap": 0.0,
                "is_event": 0.0, "event_type": "none",
            })
    return pd.DataFrame(rows)


def test_split_frames_contract():
    origin, horizon = 50, 5
    panel = _toy_panel(origin, horizon)
    hist, fut = cs.split_frames(panel, origin, horizon)
    assert "target" in hist.columns and "target" not in fut.columns
    assert len(fut) == 2 * horizon
    assert cs.date_to_idx(hist["timestamp"]).max() == origin
    assert cs.date_to_idx(fut["timestamp"]).min() == origin + 1
    shared = set(hist.columns) & set(fut.columns)
    assert set(cs.COVARIATE_COLUMNS) <= shared


def test_extract_quantile_matrix():
    origin = 50
    out = pd.DataFrame({
        "series_id": ["a"] * 3 + ["b"] * 3,
        "timestamp": list(pd.to_datetime(cs.idx_to_date(
            np.array([51, 52, 53])))) * 2,
        "predictions": np.ones(6),
        "0.1": np.arange(6, dtype=float),
        "0.5": np.arange(6, dtype=float) + 1,
        "0.9": np.arange(6, dtype=float) + 2,
    })
    sid, hor, raw = cs.extract_quantile_matrix(out, origin, (0.1, 0.5, 0.9))
    assert list(hor) == [1, 2, 3, 1, 2, 3]
    assert raw.shape == (6, 3)
    assert (np.diff(raw, axis=1) > 0).all()
