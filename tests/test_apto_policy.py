"""ST6-11 (3)-(4): A-PTO base-stock policy contract tests."""

import json

import numpy as np
import pytest

from core_pipeline.policies.apto_policy import (
    APTOPolicy, load_safety_quantiles,
)
from core_pipeline.simulator.engine import Observation

SERIES = ("SKU_A", "SKU_B")


class StubReplayer:
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def get(self, origin_idx, series_ids):
        return np.asarray(self.forecasts[int(origin_idx)], dtype=np.float64)


def fc(rows_by_day):
    """(28, 2) forecast with given {day(1-based): [a, b]} rows."""
    f = np.zeros((28, 2))
    for d, vals in rows_by_day.items():
        f[d - 1, :] = vals
    return f


def make_obs(inv, pipeline, k_g, t_idx=100):
    n = len(inv)
    return Observation(
        t_idx=t_idx,
        demand_hist=np.zeros((40, n)),
        inv=np.asarray(inv, dtype=np.float64),
        pipeline=np.asarray(pipeline, dtype=np.float64),
        k_g=float(k_g),
        h=np.ones(n), b=np.ones(n) * 3.0, w=np.ones(n),
    )


def make_policy(forecasts, q=(0.0, 0.0)):
    return APTOPolicy("apto", StubReplayer(forecasts), SERIES,
                      {"SKU_A": q[0], "SKU_B": q[1]})


def test_empty_stock_covers_3day_demand_only():
    # demand inside the window (d2) is ordered; d5 is beyond protection
    pol = make_policy({100: fc({2: [5.0, 0.0], 5: [0.0, 9.0]})})
    q = pol.decide(make_obs([0, 0], [[0, 0], [0, 0]], k_g=100.0))
    assert q == pytest.approx([5.0, 0.0], abs=1e-9)
    d = pol.last_diagnostics()
    assert d["scaling_active"] == 0.0 and d["scale_ratio"] == 1.0


def test_inventory_position_subtracts_stock_and_pipeline():
    pol = make_policy({100: fc({1: [4.0, 4.0], 3: [4.0, 4.0]})})
    # S = 8 each; IP_A = 2+1+5 = 8 -> 0, IP_B = 0+3+0 = 3 -> 5
    q = pol.decide(make_obs([2, 0], [[1, 5], [3, 0]], k_g=100.0))
    assert q == pytest.approx([0.0, 5.0], abs=1e-9)


def test_safety_quantile_adds_and_negative_floors_at_zero():
    pol = make_policy({100: fc({2: [5.0, 5.0]})}, q=(2.5, -50.0))
    q = pol.decide(make_obs([0, 0], [[0, 0], [0, 0]], k_g=100.0))
    # A: 5 + 2.5; B: max(0, 5 - 50) = 0
    assert q == pytest.approx([7.5, 0.0], abs=1e-9)


def test_proportional_scaling_numbers_and_diagnostics():
    pol = make_policy({100: fc({2: [6.0, 6.0]})})
    q = pol.decide(make_obs([0, 0], [[0, 0], [0, 0]], k_g=6.0))
    assert q == pytest.approx([3.0, 3.0], abs=1e-9)
    d = pol.last_diagnostics()
    assert d["scaling_active"] == 1.0
    assert d["scale_ratio"] == pytest.approx(0.5, abs=1e-12)
    assert d["requested_weighted_sum"] == pytest.approx(12.0, abs=1e-9)
    assert d["weighted_order_sum"] == pytest.approx(6.0, rel=1e-9)
    assert d["capacity_violation"] == 0.0


def test_below_capacity_submitted_unmodified():
    pol = make_policy({100: fc({2: [2.0, 3.0]})})
    q = pol.decide(make_obs([0, 0], [[0, 0], [0, 0]], k_g=6.0))
    assert q == pytest.approx([2.0, 3.0], abs=1e-12)
    d = pol.last_diagnostics()
    assert d["scaling_active"] == 0.0 and d["scale_ratio"] == 1.0


def test_quantile_loader_rejects_missing_series(tmp_path):
    p = tmp_path / "sq.json"
    p.write_text(json.dumps({
        "protection_days": 3,
        "quantiles": {"r3": {"SKU_A": 1.0}},
    }))
    got = load_safety_quantiles(3, ("SKU_A",), path=p)
    assert got == {"SKU_A": 1.0}
    with pytest.raises(ValueError, match="missing quantiles"):
        load_safety_quantiles(3, SERIES, path=p)
    with pytest.raises(ValueError, match="no quantile table"):
        load_safety_quantiles(5, ("SKU_A",), path=p)