"""lp_policy: strict replayer lineage/completeness + LP policy contract.

TestEngineRolling drives the policy through the real NumPy engine
(run_window, frozen configs/simulator.yaml, strict ViolationHandling.ERROR):
daily re-solve, first-day-only execution, and the D+3 arrival of the
first order are asserted on engine DayResults, closing ST5-03 (8)'s
rolling-execution item end to end.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from core_pipeline.data.db import PROJECT_ROOT
from core_pipeline.dfl.lp_problem import A5Solution, solve_a5
from core_pipeline.dfl.lp_window import to_lp_window
from core_pipeline.forecast import point_store
from core_pipeline.forecast.common import file_sha256
from core_pipeline.policies import lp_policy
from core_pipeline.policies.lp_policy import (
    EvalPredictionReplayer, LPForecastPolicy, PolicyError, ReplayError,
)
from core_pipeline.simulator.engine import (
    BundleState, Observation, SkuParams, load_sim_config, run_window,
)

MODEL = "tsmixer_b_r3"
SCENARIO = "core"
FOLD = 1
SERIES = ("SKU_A", "SKU_B")
ORIGINS = (100, 101, 102, 103, 104)


def forecast_value(origin, series_rank, horizon):
    return float(origin * 1000 + series_rank * 100 + horizon)


def build_frame(origins=ORIGINS, series=SERIES):
    sid, org, hor, yhat = [], [], [], []
    for o in origins:
        for rank, s in enumerate(series):
            for h in range(1, 29):
                sid.append(s)
                org.append(o)
                hor.append(h)
                yhat.append(forecast_value(o, rank, h))
    return point_store.build_point_frame(
        np.asarray(sid, dtype=object), np.asarray(org), np.asarray(hor),
        np.asarray(yhat))


def write_eval(tmp_path, df, path_sha="scenariosha", fold=FOLD):
    fname = f"{MODEL}_{SCENARIO}_f{fold}.parquet"
    fpath = tmp_path / fname
    df.to_parquet(fpath, index=False)
    manifest = {
        "model": MODEL,
        "prediction_kind": "decision_focused_point",
        "prediction_purpose": "rolling_decision_replay",
        "folds": [{
            "fold": fold,
            "file": fname,
            "prediction_file_sha256": file_sha256(fpath),
            "demand_path_sha256": path_sha,
        }],
    }
    (tmp_path / f"{MODEL}_{SCENARIO}_manifest.json").write_text(
        json.dumps(manifest))
    return fpath


def make_replayer(tmp_path, **kwargs):
    defaults = dict(
        model=MODEL, scenario=SCENARIO, fold=FOLD,
        expected_origins=ORIGINS, expected_series=SERIES,
        expected_path_sha="scenariosha", base_dir=tmp_path,
    )
    defaults.update(kwargs)
    return EvalPredictionReplayer(**defaults)


def make_obs(inv, pipeline, h, b, k_g, t_idx=100, n_hist=40):
    n = len(inv)
    return Observation(
        t_idx=t_idx,
        demand_hist=np.zeros((n_hist, n)),
        inv=np.asarray(inv, dtype=np.float64),
        pipeline=np.asarray(pipeline, dtype=np.float64),
        k_g=float(k_g),
        h=np.asarray(h, dtype=np.float64),
        b=np.asarray(b, dtype=np.float64),
        w=np.ones(n, dtype=np.float64),
    )


class TestReplayer:
    def test_roundtrip_and_column_order(self, tmp_path):
        write_eval(tmp_path, build_frame())
        rep = make_replayer(tmp_path)
        block = rep.get(101, ("SKU_B", "SKU_A"))
        assert block.shape == (28, 2)
        assert block[0, 0] == forecast_value(101, 1, 1)
        assert block[27, 1] == forecast_value(101, 0, 28)

    def test_missing_file_is_hard_error(self, tmp_path):
        with pytest.raises(ReplayError, match="missing eval predictions"):
            make_replayer(tmp_path)

    def test_sha_tamper_rejected(self, tmp_path):
        fpath = write_eval(tmp_path, build_frame())
        df = pd.read_parquet(fpath)
        df.loc[0, "yhat"] = df.loc[0, "yhat"] + 1.0
        df.to_parquet(fpath, index=False)
        with pytest.raises(ReplayError, match="sha256 mismatch"):
            make_replayer(tmp_path)

    def test_path_sha_lineage_rejected(self, tmp_path):
        write_eval(tmp_path, build_frame(), path_sha="otherscenario")
        with pytest.raises(ReplayError, match="demand_path_sha256"):
            make_replayer(tmp_path)

    def test_origin_set_mismatch_rejected(self, tmp_path):
        write_eval(tmp_path, build_frame(origins=ORIGINS[:-1]))
        with pytest.raises(ReplayError, match="origin set mismatch"):
            make_replayer(tmp_path)

    def test_incomplete_horizon_rejected(self, tmp_path):
        df = build_frame()
        df = df[~((df["origin_idx"] == 102) & (df["series_id"] == "SKU_A")
                  & (df["horizon"] == 28))]
        write_eval(tmp_path, df)
        with pytest.raises(ValueError, match="full h=1"):
            make_replayer(tmp_path)

    def test_unknown_origin_no_fallback(self, tmp_path):
        write_eval(tmp_path, build_frame())
        rep = make_replayer(tmp_path)
        with pytest.raises(ReplayError, match="origin 999"):
            rep.get(999, SERIES)


class StubReplayer:
    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.requested = []

    def get(self, origin_idx, series_ids):
        self.requested.append(int(origin_idx))
        if int(origin_idx) not in self.forecasts:
            raise ReplayError(
                f"stub has no forecast for origin {origin_idx}; "
                f"known: {sorted(self.forecasts)}")
        return np.asarray(self.forecasts[int(origin_idx)], dtype=np.float64)


def spike_forecast(n_sku, day, amount):
    f = np.zeros((28, n_sku))
    f[day - 1, :] = amount
    return f


class TestPolicyLP:
    def test_orders_cover_h3_spike_exactly(self):
        pol = LPForecastPolicy("b", StubReplayer(
            {100: spike_forecast(2, 3, 5.0)}), SERIES)
        obs = make_obs(inv=[0, 0], pipeline=[[0, 0], [0, 0]],
                       h=[1.0, 1.0], b=[3.0, 3.0], k_g=100.0)
        q = pol.decide(obs)
        assert q == pytest.approx([5.0, 5.0], abs=1e-6)
        d = pol.last_diagnostics()
        assert d["capacity_violation"] == 0.0
        assert d["weighted_order_sum"] == pytest.approx(10.0, abs=1e-5)

    def test_existing_stock_suppresses_order(self):
        pol = LPForecastPolicy("b", StubReplayer(
            {100: spike_forecast(2, 3, 5.0)}), SERIES)
        obs = make_obs(inv=[5, 5], pipeline=[[0, 0], [0, 0]],
                       h=[1.0, 1.0], b=[3.0, 3.0], k_g=100.0)
        q = pol.decide(obs)
        assert q == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_capacity_constraint_respected(self):
        pol = LPForecastPolicy("b", StubReplayer(
            {100: spike_forecast(2, 3, 50.0)}), SERIES)
        obs = make_obs(inv=[0, 0], pipeline=[[0, 0], [0, 0]],
                       h=[1.0, 1.0], b=[3.0, 3.0], k_g=60.0)
        q = pol.decide(obs)
        total = float(q.sum())
        assert total <= 60.0 * (1.0 + 1e-6)
        assert total == pytest.approx(60.0, rel=1e-4)

    def test_rolling_daily_resolve_and_d3_arrival(self):
        forecasts = {t: spike_forecast(1, 3, 5.0) for t in range(100, 106)}
        pol = LPForecastPolicy("b", StubReplayer(forecasts), ("SKU_A",))
        inv = np.zeros(1)
        pipe = np.zeros((1, 2))
        orders = []
        for t in range(100, 106):
            obs = make_obs(inv=inv, pipeline=pipe, h=[1.0], b=[3.0],
                           k_g=100.0, t_idx=t)
            q = pol.decide(obs)
            orders.append(float(q[0]))
            inv = inv + pipe[:, 0]
            pipe = np.column_stack([pipe[:, 1], q])
        assert pol.n_solves == 6
        assert orders[0] == pytest.approx(5.0, abs=1e-6)
        assert orders[1] == pytest.approx(0.0, abs=1e-6)
        assert sum(orders) == pytest.approx(5.0, abs=1e-5)
        assert float(inv[0]) == pytest.approx(5.0, abs=1e-6)


class TestPolicyHygiene:
    def _patched(self, monkeypatch, orders_row, status="optimal"):
        orders = np.zeros((29, 2))
        orders[0] = orders_row
        sol = A5Solution(orders=orders, objective=1.0, status=status)
        monkeypatch.setattr(lp_policy, "solve_a5",
                            lambda *a, **k: sol)
        pol = LPForecastPolicy("b", StubReplayer(
            {100: np.zeros((28, 2))}), SERIES)
        obs = make_obs(inv=[0, 0], pipeline=[[0, 0], [0, 0]],
                       h=[1.0, 1.0], b=[3.0, 3.0], k_g=10.0)
        return pol, obs

    def test_tiny_negative_clipped_and_recorded(self, monkeypatch):
        pol, obs = self._patched(monkeypatch, [-5e-10, 2.0])
        q = pol.decide(obs)
        assert q[0] == 0.0 and q[1] == 2.0
        assert pol.last_diagnostics()["n_neg_clipped"] == 1.0

    def test_large_negative_is_policy_error(self, monkeypatch):
        pol, obs = self._patched(monkeypatch, [-1e-8, 2.0])
        with pytest.raises(PolicyError, match="order below"):
            pol.decide(obs)

    def test_capacity_overflow_is_policy_error(self, monkeypatch):
        pol, obs = self._patched(monkeypatch, [6.0, 6.0])
        with pytest.raises(PolicyError, match="capacity exceeded"):
            pol.decide(obs)

    def test_within_tolerance_submitted_unmodified(self, monkeypatch):
        pol, obs = self._patched(monkeypatch, [5.0, 5.0 + 4e-6])
        q = pol.decide(obs)
        assert q[1] == pytest.approx(5.0 + 4e-6, abs=1e-12)
        assert pol.last_diagnostics()["capacity_violation"] > 0.0

    def test_nonoptimal_status_hard_stop(self, monkeypatch):
        pol, obs = self._patched(monkeypatch, [0.0, 0.0],
                                 status="infeasible")
        with pytest.raises(PolicyError, match="LP status"):
            pol.decide(obs)


class TestEngineRolling:
    """ST5-03 (8) rolling execution through the real engine (strict mode)."""

    def _cfg(self):
        return load_sim_config(PROJECT_ROOT / "configs" / "simulator.yaml")

    def _params(self):
        return SkuParams(
            series_ids=("SKU_A",),
            h=np.array([1.0]),
            b=np.array([3.0]),
            w=np.array([1.0]),
        )

    def test_run_window_first_order_arrives_d3(self):
        cfg = self._cfg()
        t0 = 100
        n_days = 6
        forecasts = {t: spike_forecast(1, 3, 5.0)
                     for t in range(t0, t0 + n_days)}
        stub = StubReplayer(forecasts)
        pol = LPForecastPolicy("b_eval", stub, ("SKU_A",))
        state = BundleState(inv=np.zeros(1),
                            pipeline=np.zeros((1, cfg.lead_time_core)))
        demand = np.zeros((n_days, 1))
        hist = np.zeros((40, 1))
        final, results = run_window(
            state, demand, hist, pol, self._params(),
            k_g=100.0, cfg=cfg, t0_idx=t0,
        )
        assert stub.requested == list(range(t0, t0 + n_days))
        assert pol.n_solves == n_days
        orders = [float(r.order[0]) for r in results]
        assert orders[0] == pytest.approx(5.0, abs=1e-6)
        assert sum(orders) == pytest.approx(5.0, abs=1e-5)
        assert all(o == pytest.approx(0.0, abs=1e-6) for o in orders[1:])
        assert float(results[3].inv_start[0]) == pytest.approx(5.0, abs=1e-6)
        assert float(results[2].inv_start[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(final.inv[0]) == pytest.approx(5.0, abs=1e-6)
        assert float(final.pipeline.sum()) == pytest.approx(0.0, abs=1e-6)
        for r in results:
            assert not r.projection_applied
            assert r.violation_amount == 0.0

    def test_run_window_first_day_only_plan_discarded(self):
        cfg = self._cfg()
        t0 = 200
        day0 = spike_forecast(1, 3, 5.0)
        day0[3, 0] = 4.0
        forecasts = {t0: day0, t0 + 1: np.zeros((28, 1))}
        stub = StubReplayer(forecasts)
        pol = LPForecastPolicy("b_eval", stub, ("SKU_A",))
        state = BundleState(inv=np.zeros(1),
                            pipeline=np.zeros((1, cfg.lead_time_core)))
        demand = np.zeros((2, 1))
        hist = np.zeros((40, 1))
        demand_29, inv0, pipeline0 = to_lp_window(
            day0, state.inv, state.pipeline)
        plan0 = solve_a5(demand_29, inv0, pipeline0,
                         np.array([1.0]), np.array([3.0]),
                         np.array([1.0]), 100.0)
        assert float(plan0.orders[0, 0]) == pytest.approx(5.0, abs=1e-6)
        assert float(plan0.orders[1, 0]) == pytest.approx(4.0, abs=1e-6)
        _, results = run_window(
            state, demand, hist, pol, self._params(),
            k_g=100.0, cfg=cfg, t0_idx=t0,
        )
        assert float(results[0].order[0]) == pytest.approx(5.0, abs=1e-6)
        assert float(results[1].order[0]) == pytest.approx(0.0, abs=1e-6)

class TestA5Boundary:
    """ST6-09 (2): data-end boundary behavior of the perfect-info policy."""

    def _policy(self, path, start_idx, last_idx):
        from core_pipeline.policies.lp_policy import A5PerfectInfoPolicy
        return A5PerfectInfoPolicy("a5", path, start_idx, last_idx, ("S",))

    def test_full_horizon_matches_direct_solve(self):
        rng = np.random.default_rng(3)
        path = rng.gamma(2.0, 2.0, (60, 1))
        pol = self._policy(path, 100, 100 + 60)
        obs = make_obs(inv=[4.0], pipeline=[[1.0, 0.0]], h=[1.0], b=[3.0],
                       k_g=50.0, t_idx=100)
        q = pol.decide(obs)
        demand_29, inv0, pipe0 = to_lp_window(
            path[1:29], obs.inv, obs.pipeline)
        ref = solve_a5(demand_29, inv0, pipe0, np.array([1.0]),
                       np.array([3.0]), np.array([1.0]), 50.0)
        assert q[0] == pytest.approx(float(ref.orders[0, 0]), abs=1e-6)
        assert pol.last_truncated is False

    def test_future_under_lead_orders_zero(self):
        path = np.full((60, 1), 5.0)
        last_idx = 160
        pol = self._policy(path, 110, last_idx)
        for t, fd in ((158, 2), (159, 1)):
            obs = make_obs(inv=[0.0], pipeline=[[0.0, 0.0]], h=[1.0],
                           b=[3.0], k_g=50.0, t_idx=t)
            q = pol.decide(obs)
            assert q[0] == pytest.approx(0.0, abs=1e-6)
            assert pol.last_diagnostics()["future_days"] == fd
            assert pol.last_truncated is True

    def test_future_zero_no_solve(self):
        path = np.full((60, 1), 5.0)
        pol = self._policy(path, 110, 160)
        obs = make_obs(inv=[7.0], pipeline=[[0.0, 0.0]], h=[1.0], b=[3.0],
                       k_g=50.0, t_idx=160)
        q = pol.decide(obs)
        assert q[0] == 0.0
        assert pol.n_solves == 0
        assert math.isnan(pol.last_diagnostics()["lp_objective"])