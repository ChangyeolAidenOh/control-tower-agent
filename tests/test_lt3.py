"""Tests for Stage 3-E methods and the cumulative storage contract
(ST3-08): inverse-CDF sampler, comonotone summation, MC determinism,
and the daily/cumulative mixing guards."""

import numpy as np
import pandas as pd
import pytest

from core_pipeline.forecast import lt3
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES

QCOLS = store.quantile_columns()
N_Q = len(QUANTILES)


def daily_frame(n_series=2, origins=(10, 11), h_max=3, base=1.0):
    rows = []
    for o in origins:
        for s in range(n_series):
            for h in range(1, h_max + 1):
                q = base * (s + 1) + 0.1 * np.arange(N_Q) + 0.01 * h
                rows.append({"series_id": f"s{s}", "origin_idx": o,
                             "horizon": h, "target_idx": o + h,
                             **{c: v for c, v in zip(QCOLS, q)}})
    return pd.DataFrame(rows)


def test_sampler_grid_truncation_monotone():
    qv = np.array([[0.0, 1.0, 2.0]], dtype=float)
    lv = [0.1, 0.5, 0.9]
    u = np.array([[0.1, 0.5, 0.9, 0.01, 0.99, 0.7]])
    out = lt3.sample_from_quantiles(qv, u, lv)
    np.testing.assert_allclose(out[0, :3], [0.0, 1.0, 2.0])  # grid exact
    assert out[0, 3] == 0.0 and out[0, 4] == 2.0             # truncation
    assert out[0, 5] == pytest.approx(1.5)                   # linear
    u_sorted = np.sort(np.random.default_rng(0).random((1, 200)))
    s = lt3.sample_from_quantiles(qv, u_sorted, lv)
    assert (np.diff(s[0]) >= 0).all()                        # monotone


def test_comonotone_sums_marginals():
    df = lt3.comonotone_from_daily(daily_frame())
    store.validate_lt_frame(df)
    row = df[(df["series_id"] == "s0") & (df["origin_idx"] == 10)]
    # sum over h of q-col: 3*base + 3*0.1*j + (0.01+0.02+0.03)
    expect = 3 * 1.0 + 0.3 * np.arange(N_Q) + 0.06
    np.testing.assert_allclose(row[QCOLS].to_numpy()[0], expect, rtol=1e-6)


def test_pivot_requires_full_h():
    bad = daily_frame().query("~(horizon == 2 and series_id == 's0')")
    with pytest.raises(ValueError):
        lt3.comonotone_from_daily(bad)


def test_mixing_guards_both_ways():
    lt = lt3.comonotone_from_daily(daily_frame())
    with pytest.raises(ValueError):
        store.assert_daily_frame(lt)            # cumulative where daily
    with pytest.raises(ValueError):
        store.validate_lt_frame(daily_frame())  # daily where cumulative


def test_validate_lt_rejects():
    good = lt3.comonotone_from_daily(daily_frame())
    with pytest.raises(ValueError):             # duplicate key
        store.validate_lt_frame(pd.concat([good, good.iloc[[0]]]))
    with pytest.raises(ValueError):             # broken span
        store.validate_lt_frame(good.assign(
            target_end_idx=good["target_end_idx"] + 1))
    with pytest.raises(ValueError):             # wrong kind
        store.validate_lt_frame(good.assign(target_kind="daily"))
    with pytest.raises(ValueError):             # crossing
        store.validate_lt_frame(good.assign(**{"q0.95": -1.0}))


def test_independent_mc_deterministic_and_valid():
    df1 = lt3.independent_mc_from_daily(daily_frame(), n=512)
    df2 = lt3.independent_mc_from_daily(daily_frame(), n=512)
    pd.testing.assert_frame_equal(df1, df2)     # fixed streams
    store.validate_lt_frame(df1)
    qm = df1[QCOLS].to_numpy()
    daily = daily_frame()
    lo3 = daily.groupby(["series_id", "origin_idx"])["q0.05"].sum().min()
    hi3 = daily.groupby(["series_id", "origin_idx"])["q0.95"].sum().max()
    assert qm.min() >= lo3 - 1e-9 and qm.max() <= hi3 + 1e-9  # truncation


def test_independent_mc_differs_from_comonotone():
    daily = daily_frame(base=2.0)
    como = lt3.comonotone_from_daily(daily)[QCOLS].to_numpy()
    indep = lt3.independent_mc_from_daily(daily, n=2048)[QCOLS].to_numpy()
    # independence concentrates the sum: narrower 90% band than the
    # comonotonic coupling on identical marginals
    assert (indep[:, -1] - indep[:, 0]).mean() < \
           (como[:, -1] - como[:, 0]).mean()
