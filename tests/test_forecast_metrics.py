"""Tests for metrics.py and lgbm_model.py (ST3-03 contract). No DB."""

import numpy as np
import pytest

from core_pipeline.forecast import lgbm_model as lm
from core_pipeline.forecast.metrics import (
    macro_mean,
    mase_table,
    monotone_rearrange,
    nonzero_mape,
    pinball,
    seasonal_naive_mae,
    wape,
    wql,
)


def test_pinball_hand_values():
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    assert pinball(np.array([8.0]), np.array([10.0]), 0.9)[0] == pytest.approx(0.2)


def test_wql_coefficient_two_hand_example():
    # rho_0.9(10 - 8) = 1.8 -> WQL = 2 * 1.8 / (1 * 10) = 0.36
    v = wql(np.array([10.0]), np.array([[8.0]]), (0.9,))
    assert v == pytest.approx(0.36)


def test_wql_median_equals_wape():
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, 500).astype(float)
    med = y + rng.normal(0, 1, 500)
    assert wql(y, med[:, None], (0.5,)) == pytest.approx(wape(y, med))


def test_wql_zero_denominator_nan():
    assert np.isnan(wql(np.zeros(5), np.ones((5, 1)), (0.5,)))
    assert np.isnan(wape(np.zeros(5), np.ones(5)))


def test_monotone_rearrange():
    preds = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0]])
    fixed, rate = monotone_rearrange(preds)
    assert (np.diff(fixed, axis=1) >= 0).all()
    assert rate == pytest.approx(0.5)
    np.testing.assert_array_equal(fixed[1], preds[1])


def test_mase_undefined_counted_no_epsilon():
    denoms = {"a": 2.0, "b": 0.0, "c": float("nan")}
    out = mase_table(
        y=np.array([3.0, 3.0, 5.0, 5.0, 1.0]),
        yhat=np.array([1.0, 1.0, 5.0, 5.0, 1.0]),
        series_ids=["a", "a", "b", "b", "c"],
        denominators=denoms,
    )
    row_a = out[out["series_id"] == "a"].iloc[0]
    assert row_a["mase"] == pytest.approx(1.0)
    assert not out[out["series_id"] == "b"]["defined"].iloc[0]
    assert not out[out["series_id"] == "c"]["defined"].iloc[0]
    assert int((~out["defined"]).sum()) == 2
    mean, undef = macro_mean(out["mase"].to_numpy())
    assert mean == pytest.approx(1.0) and undef == 2


def test_seasonal_naive_mae():
    v = np.arange(15, dtype=float)  # |y_t - y_{t-7}| = 7
    assert seasonal_naive_mae(v, m=7) == pytest.approx(7.0)
    assert np.isnan(seasonal_naive_mae(np.ones(7), m=7))
    assert seasonal_naive_mae(np.ones(20), m=7) == 0.0  # caller marks undefined


def test_nonzero_mape_counts():
    val, n = nonzero_mape(np.array([0.0, 2.0, 4.0]), np.array([1.0, 1.0, 2.0]))
    assert n == 2 and val == pytest.approx(0.5)
    val, n = nonzero_mape(np.zeros(3), np.ones(3))
    assert n == 0 and np.isnan(val)


def test_sample_trials_deterministic_and_bounded():
    a = lm.sample_trials(30)
    b = lm.sample_trials(30)
    assert a == b
    for t in a:
        assert 32 <= t["num_leaves"] <= 256
        assert 0.02 <= t["learning_rate"] <= 0.2
        assert t["min_data_in_leaf"] in (50, 100, 200, 500, 1000)
        assert 0.6 <= t["feature_fraction"] <= 1.0
        assert 0.6 <= t["bagging_fraction"] <= 1.0
        assert 1e-3 <= t["lambda_l2"] <= 10.0
    assert lm.base_params(0.5)["bagging_freq"] == 1


def test_select_best_trial_tie_first():
    results = [
        {"trial": 1, "score": 0.5}, {"trial": 2, "score": 0.4},
        {"trial": 3, "score": 0.4}, {"trial": 4, "score": float("nan")},
    ]
    assert lm.select_best_trial(results)["trial"] == 2


def test_lgbm_train_save_reload_roundtrip(tmp_path):
    import pandas as pd
    from core_pipeline.forecast.dataset import TrainFrame

    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({
        "x1": rng.normal(0, 1, n).astype(np.float32),
        "x2": rng.uniform(0, 1, n).astype(np.float32),
        "cat": rng.integers(0, 4, n).astype(np.int32),
    })
    y = (2 * X["x1"].to_numpy() + X["cat"].to_numpy()
         + rng.normal(0, 0.5, n)).astype(np.float32)
    frame = TrainFrame(X=X, y=y, meta=pd.DataFrame(index=range(n)),
                       categorical=("cat",), spec_hash="t", encoder_hash="t",
                       target_mode="per_horizon")
    ds = lm.make_dataset(frame)
    booster, used = lm.train_quantile(ds, 0.5, {"num_leaves": 15},
                                      num_boost_round=60, num_threads=2)
    assert used == 60
    path = tmp_path / "m.txt"
    sha = lm.save_booster(booster, path)
    assert len(sha) == 16
    assert lm.reload_max_abs_diff(booster, path, X.head(200)) < 1e-9
    raw, fixed, rate = lm.predict_matrix({0.5: booster}, X.head(50), (0.5,))
    assert raw.shape == (50, 1) and rate == 0.0
    np.testing.assert_array_equal(raw, fixed)
