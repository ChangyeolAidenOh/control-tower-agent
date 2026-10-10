"""Stage 6: shared LP forecast policies (A1/A2/B, A3 SAA) + strict replayers.

Contract (ST5-09 (a), ST6-01, ST6-02 (5), ST6-03 (5), ST6-08):
  daily   replayed forecast -> to_lp_window (29-day zero-pad axis, ST5-04)
          -> solve (point LP solve_a5, or compiled SAA problem for A3)
          -> execute orders[0] only, re-optimize next day;
  inputs  the engine Observation exclusively -- no training-state injection
          (ST5-03 (7)); costs h/b and K_g come from the Observation / the
          compiled problem keyed by the environment r, so the r=10 stress
          prices the LP at environment costs (ST6-01 (4));
  replay  pre-generated evaluation parquets only (outputs/predictions_eval);
          manifest lineage and completeness are verified at load, and a
          missing (origin, series) is a hard error -- never a live forward;
  hygiene q < -1e-9 policy error; [-1e-9, 0) clipped to zero and recorded;
          fsum(w q) > K_g (1 + 1e-6) policy error, no radial rescale; values
          within tolerance are submitted unmodified (ST6-01 (3));
  failure any non-optimal LP status is a hard stop with diagnostics -- no
          fallback order (ST6-01 (6)).
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data.db import PROJECT_ROOT
from core_pipeline.dfl.lp_problem import A3SaaProblem, solve_a5
from core_pipeline.dfl.lp_window import to_lp_window
from core_pipeline.forecast import point_store
from core_pipeline.forecast import prediction_store as quantile_store
from core_pipeline.forecast.common import file_sha256
from core_pipeline.forecast.tsmixer.config import QUANTILES_9
from core_pipeline.simulator.engine import Observation

EVAL_DIR = PROJECT_ROOT / "outputs" / "predictions_eval"
FORECAST_DAYS = 28
N_SCEN = 9
NEG_TOL = 1e-9
CAP_REL_TOL = 1e-6


class PolicyError(RuntimeError):
    """LP policy violated its execution contract (hard stop, ST6-01 (6))."""


class ReplayError(RuntimeError):
    """Evaluation prediction replay is missing or fails lineage checks."""


def _verify_replay_files(base: Path, model: str, scenario: str, fold: int,
                         expected_path_sha: str | None):
    """Shared lineage gate (ST6-03 (5)): file + manifest exist, entry
    matches the file sha, purpose is rolling replay, demand-path lineage
    matches the runner's scenario. Returns (fpath, manifest, entry)."""
    fname = f"{model}_{scenario}_f{fold}.parquet"
    fpath = base / fname
    if not fpath.exists():
        raise ReplayError(
            f"missing eval predictions {fpath}; live model forward "
            "fallback is not allowed (ST6-01 (2))")
    man_path = base / f"{model}_{scenario}_manifest.json"
    if not man_path.exists():
        raise ReplayError(f"missing manifest {man_path}")
    manifest = json.loads(man_path.read_text())
    if manifest.get("prediction_purpose") != "rolling_decision_replay":
        raise ReplayError(
            f"{man_path.name}: prediction_purpose "
            f"{manifest.get('prediction_purpose')!r} is not a rolling "
            "decision replay artifact")
    entry = next(
        (e for e in manifest.get("folds", []) if e["fold"] == fold), None)
    if entry is None:
        raise ReplayError(f"fold {fold} absent from {man_path.name}")
    if entry["file"] != fname:
        raise ReplayError(
            f"manifest entry file {entry['file']!r} != {fname!r}")
    sha = file_sha256(fpath)
    if sha != entry["prediction_file_sha256"]:
        raise ReplayError(
            f"{fname}: sha256 mismatch vs manifest "
            f"({sha[:12]} != {entry['prediction_file_sha256'][:12]})")
    if (expected_path_sha is not None
            and entry.get("demand_path_sha256") != expected_path_sha):
        raise ReplayError(
            f"{fname}: demand_path_sha256 does not match the runner's "
            "scenario path (ST6-03 (5))")
    return fpath, manifest, entry


def _check_keys(df: pd.DataFrame, fname: str,
                expected_origins, expected_series):
    exp_origins = sorted(int(o) for o in expected_origins)
    got_origins = sorted(int(o) for o in df["origin_idx"].unique())
    if got_origins != exp_origins:
        raise ReplayError(
            f"{fname}: origin set mismatch (got {got_origins[0]}.."
            f"{got_origins[-1]} n={len(got_origins)}, expected "
            f"{exp_origins[0]}..{exp_origins[-1]} n={len(exp_origins)})")
    exp_series = sorted(str(s) for s in expected_series)
    got_series = sorted(str(s) for s in df["series_id"].unique())
    if got_series != exp_series:
        raise ReplayError(
            f"{fname}: series set mismatch "
            f"(got {len(got_series)}, expected {len(exp_series)})")
    return got_origins, exp_series


def _finalize_orders(name: str, q_raw: np.ndarray, w: np.ndarray,
                     k_g: float, t_idx: int, fhash: str):
    """ST6-01 (3) numeric hygiene, shared by every LP policy."""
    q = np.asarray(q_raw, dtype=np.float64).copy()
    min_raw = float(q.min()) if q.size else 0.0
    if (q < -NEG_TOL).any():
        raise PolicyError(
            f"{name}: order below -{NEG_TOL} at t={t_idx} "
            f"(min {min_raw:.3e}) forecast_hash={fhash}")
    n_clip = int((q < 0.0).sum())
    np.clip(q, 0.0, None, out=q)
    wsum = math.fsum((np.asarray(w, dtype=np.float64) * q).tolist())
    cap = float(k_g)
    violation = max(0.0, wsum - cap)
    if wsum > cap * (1.0 + CAP_REL_TOL):
        raise PolicyError(
            f"{name}: capacity exceeded at t={t_idx} "
            f"(sum {wsum:.6f} > K_g {cap:.6f}, violation "
            f"{violation:.3e}); radial rescale is not allowed "
            f"forecast_hash={fhash}")
    diag = {
        "n_neg_clipped": float(n_clip),
        "min_raw_order": min_raw,
        "weighted_order_sum": float(wsum),
        "capacity_slack": float(max(0.0, cap - wsum)),
        "capacity_violation": float(violation),
    }
    return q, diag


class EvalPredictionReplayer:
    """Strict replay of one (model, scenario, fold) point-forecast parquet."""

    def __init__(
        self,
        model: str,
        scenario: str,
        fold: int,
        expected_origins,
        expected_series,
        expected_path_sha: str | None = None,
        base_dir: Path | None = None,
    ):
        base = Path(base_dir) if base_dir is not None else EVAL_DIR
        fpath, manifest, _ = _verify_replay_files(
            base, model, scenario, fold, expected_path_sha)
        df = pd.read_parquet(fpath)
        kind = manifest.get("prediction_kind", "")
        point_store.validate_point_frame(df, kind)
        got_origins, exp_series = _check_keys(
            df, fpath.name, expected_origins, expected_series)

        self.model = model
        self.scenario = scenario
        self.fold = fold
        self.prediction_kind = kind
        self._origin_pos = {o: i for i, o in enumerate(got_origins)}
        self._series_pos = {s: i for i, s in enumerate(exp_series)}
        arr = np.empty((len(got_origins), len(exp_series), FORECAST_DAYS),
                       dtype=np.float64)
        o_idx = df["origin_idx"].map(self._origin_pos).to_numpy(np.int64)
        s_idx = df["series_id"].astype(str).map(self._series_pos).to_numpy(
            np.int64)
        h_idx = df["horizon"].to_numpy(np.int64) - 1
        arr[o_idx, s_idx, h_idx] = df["yhat"].to_numpy(np.float64)
        self._arr = arr

    def get(self, origin_idx: int, series_ids) -> np.ndarray:
        """(28, n_sku) forecast for days origin+1..origin+28."""
        pos = self._origin_pos.get(int(origin_idx))
        if pos is None:
            raise ReplayError(
                f"{self.model} fold {self.fold}: origin {origin_idx} not in "
                "the replay parquet; live forward fallback is not allowed")
        try:
            cols = [self._series_pos[str(s)] for s in series_ids]
        except KeyError as exc:
            raise ReplayError(
                f"{self.model} fold {self.fold}: unknown series "
                f"{exc.args[0]!r}") from None
        return self._arr[pos, cols, :].T.copy()


class QuantileEvalReplayer:
    """Strict replay of the A3 9-quantile evaluation parquet (ST6-08)."""

    def __init__(
        self,
        model: str,
        scenario: str,
        fold: int,
        expected_origins,
        expected_series,
        expected_path_sha: str | None = None,
        base_dir: Path | None = None,
    ):
        base = Path(base_dir) if base_dir is not None else EVAL_DIR
        fpath, manifest, _ = _verify_replay_files(
            base, model, scenario, fold, expected_path_sha)
        df = pd.read_parquet(fpath)
        quantile_store.validate_frame(df, QUANTILES_9)
        got_origins, exp_series = _check_keys(
            df, fpath.name, expected_origins, expected_series)

        self.model = model
        self.scenario = scenario
        self.fold = fold
        self.prediction_kind = manifest.get("prediction_kind", "")
        self._origin_pos = {o: i for i, o in enumerate(got_origins)}
        self._series_pos = {s: i for i, s in enumerate(exp_series)}
        qcols = quantile_store.quantile_columns(QUANTILES_9)
        arr = np.empty((len(got_origins), len(exp_series), FORECAST_DAYS,
                        len(qcols)), dtype=np.float64)
        o_idx = df["origin_idx"].map(self._origin_pos).to_numpy(np.int64)
        s_idx = df["series_id"].astype(str).map(self._series_pos).to_numpy(
            np.int64)
        h_idx = df["horizon"].to_numpy(np.int64) - 1
        arr[o_idx, s_idx, h_idx, :] = df[qcols].to_numpy(np.float64)
        self._arr = arr

    def get(self, origin_idx: int, series_ids) -> np.ndarray:
        """(28, n_sku, 9) quantile forecasts, scenario axis last."""
        pos = self._origin_pos.get(int(origin_idx))
        if pos is None:
            raise ReplayError(
                f"{self.model} fold {self.fold}: origin {origin_idx} not in "
                "the replay parquet; live forward fallback is not allowed")
        try:
            cols = [self._series_pos[str(s)] for s in series_ids]
        except KeyError as exc:
            raise ReplayError(
                f"{self.model} fold {self.fold}: unknown series "
                f"{exc.args[0]!r}") from None
        return np.transpose(self._arr[pos, cols, :, :], (1, 0, 2)).copy()


class LPForecastPolicy:
    """Rolling point-forecast LP policy (A1/A2/B)."""

    def __init__(self, name: str, replayer, series_ids):
        self.name = name
        self._replayer = replayer
        self._series_ids = tuple(str(s) for s in series_ids)
        self._diag: dict[str, float] = {}
        self.n_solves = 0
        self.last_forecast_hash = ""

    def decide(self, obs: Observation) -> np.ndarray:
        n_sku = len(self._series_ids)
        forecast = np.asarray(
            self._replayer.get(obs.t_idx, self._series_ids),
            dtype=np.float64)
        if forecast.shape != (FORECAST_DAYS, n_sku):
            raise PolicyError(
                f"{self.name}: forecast shape {forecast.shape} != "
                f"({FORECAST_DAYS}, {n_sku})")
        fhash = hashlib.sha256(
            np.ascontiguousarray(forecast).tobytes()).hexdigest()[:16]
        self.last_forecast_hash = fhash

        demand_29, inv0, pipeline0 = to_lp_window(
            forecast, obs.inv, obs.pipeline)
        t0 = time.perf_counter()
        sol = solve_a5(demand_29, inv0, pipeline0,
                       np.asarray(obs.h, dtype=np.float64),
                       np.asarray(obs.b, dtype=np.float64),
                       np.asarray(obs.w, dtype=np.float64),
                       float(obs.k_g))
        solve_s = time.perf_counter() - t0
        self.n_solves += 1
        if sol.status != "optimal":
            raise PolicyError(
                f"{self.name}: LP status {sol.status!r} at t={obs.t_idx} "
                f"forecast_hash={fhash}; no fallback order (ST6-01 (6))")

        q, diag = _finalize_orders(
            self.name, sol.orders[0], obs.w, obs.k_g, obs.t_idx, fhash)
        self._diag = {"lp_objective": float(sol.objective),
                      "lp_solve_s": float(solve_s), **diag}
        return q

    def last_diagnostics(self) -> dict[str, float]:
        return dict(self._diag)


class A3SAAPolicy:
    """Rolling 9-scenario comonotone SAA policy (ST6-08): shared-q open-loop
    plan, first day executed, re-optimized daily. The compiled problem is
    keyed by (bundle, environment r) outside this class, so its baked costs
    already follow ST6-01 (4)."""

    def __init__(self, name: str, replayer: QuantileEvalReplayer,
                 series_ids, problem: A3SaaProblem):
        self.name = name
        self._replayer = replayer
        self._series_ids = tuple(str(s) for s in series_ids)
        self._problem = problem
        self._diag: dict[str, float] = {}
        self.n_solves = 0
        self.last_forecast_hash = ""

    def decide(self, obs: Observation) -> np.ndarray:
        n_sku = len(self._series_ids)
        fc = np.asarray(
            self._replayer.get(obs.t_idx, self._series_ids),
            dtype=np.float64)
        if fc.shape != (FORECAST_DAYS, n_sku, N_SCEN):
            raise PolicyError(
                f"{self.name}: forecast shape {fc.shape} != "
                f"({FORECAST_DAYS}, {n_sku}, {N_SCEN})")
        fhash = hashlib.sha256(
            np.ascontiguousarray(fc).tobytes()).hexdigest()[:16]
        self.last_forecast_hash = fhash

        scen = []
        inv0 = pipeline0 = None
        for s in range(N_SCEN):
            d29, inv0, pipeline0 = to_lp_window(
                fc[:, :, s], obs.inv, obs.pipeline)
            scen.append(d29)
        t0 = time.perf_counter()
        sol = self._problem.solve(np.stack(scen), inv0, pipeline0)
        solve_s = time.perf_counter() - t0
        self.n_solves += 1
        if sol.status != "optimal":
            raise PolicyError(
                f"{self.name}: SAA LP status {sol.status!r} at t={obs.t_idx} "
                f"forecast_hash={fhash}; no fallback order (ST6-01 (6))")

        q, diag = _finalize_orders(
            self.name, sol.orders[0], obs.w, obs.k_g, obs.t_idx, fhash)
        self._diag = {"lp_objective": float(sol.objective),
                      "lp_solve_s": float(solve_s), **diag}
        return q

    def last_diagnostics(self) -> dict[str, float]:
        return dict(self._diag)