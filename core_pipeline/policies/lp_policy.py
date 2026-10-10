"""Stage 6 task 1: shared LP forecast policy (A1/A2/B) + strict replayer.

Contract (ST5-09 (a), ST6-01, ST6-02 (5), ST6-03 (5)):
  daily   replayed 28-day forecast -> to_lp_window (29-day zero-pad axis,
          ST5-04) -> solve_a5 -> execute orders[0] only, re-optimize next day;
  inputs  the engine Observation exclusively -- no training-state injection
          (ST5-03 (7)); costs h/b and K_g come from the Observation, so the
          r=10 stress automatically prices the LP at environment costs
          (ST6-01 (4));
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
from core_pipeline.dfl.lp_problem import solve_a5
from core_pipeline.dfl.lp_window import to_lp_window
from core_pipeline.forecast import point_store
from core_pipeline.forecast.common import file_sha256
from core_pipeline.simulator.engine import Observation

EVAL_DIR = PROJECT_ROOT / "outputs" / "predictions_eval"
FORECAST_DAYS = 28
NEG_TOL = 1e-9
CAP_REL_TOL = 1e-6


class PolicyError(RuntimeError):
    """LP policy violated its execution contract (hard stop, ST6-01 (6))."""


class ReplayError(RuntimeError):
    """Evaluation prediction replay is missing or fails lineage checks."""


class EvalPredictionReplayer:
    """Strict replay of one (model, scenario, fold) evaluation parquet.

    Verifies at construction: file + manifest exist, manifest entry matches
    the file (sha256), prediction_purpose is rolling_decision_replay, the
    demand-path lineage matches the runner's scenario (when given), the
    origin set equals the expected decision days, the series set equals the
    expected panel, and every (origin, series) carries the full h=1..28
    block (point_store validator). get() never recomputes anything.
    """

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

        df = pd.read_parquet(fpath)
        kind = manifest.get("prediction_kind", "")
        point_store.validate_point_frame(df, kind)

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
        """(28, n_sku) forecast for days origin+1..origin+28, column order
        follows series_ids. Unknown origin or series is a hard error."""
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


class LPForecastPolicy:
    """Rolling LP policy over replayed point forecasts (engine Policy
    protocol: name attribute + decide(obs))."""

    def __init__(self, name: str, replayer: EvalPredictionReplayer,
                 series_ids):
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

        q = np.asarray(sol.orders[0], dtype=np.float64).copy()
        min_raw = float(q.min()) if q.size else 0.0
        if (q < -NEG_TOL).any():
            raise PolicyError(
                f"{self.name}: order below -{NEG_TOL} at t={obs.t_idx} "
                f"(min {min_raw:.3e}) forecast_hash={fhash}")
        n_clip = int((q < 0.0).sum())
        np.clip(q, 0.0, None, out=q)

        wsum = math.fsum(
            (np.asarray(obs.w, dtype=np.float64) * q).tolist())
        cap = float(obs.k_g)
        violation = max(0.0, wsum - cap)
        if wsum > cap * (1.0 + CAP_REL_TOL):
            raise PolicyError(
                f"{self.name}: capacity exceeded at t={obs.t_idx} "
                f"(sum {wsum:.6f} > K_g {cap:.6f}, violation "
                f"{violation:.3e}); radial rescale is not allowed "
                f"forecast_hash={fhash}")

        self._diag = {
            "lp_objective": float(sol.objective),
            "lp_solve_s": float(solve_s),
            "n_neg_clipped": float(n_clip),
            "min_raw_order": min_raw,
            "weighted_order_sum": float(wsum),
            "capacity_slack": float(max(0.0, cap - wsum)),
            "capacity_violation": float(violation),
        }
        return q

    def last_diagnostics(self) -> dict[str, float]:
        return dict(self._diag)