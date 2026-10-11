"""ST7-03: strict replayer for the agent-selection A3 parquet.

Same lineage discipline as lp_policy's replayers (file + manifest entry +
sha256 + demand-path sha), but the required prediction_purpose is
"agent_candidate_selection_replay" -- the Stage 6 replayers hard-require
"rolling_decision_replay", so this artifact gets its own loader instead of
loosening the frozen Stage 6 code. Selection-only: candidate gating and
ranking read from here; H3/H5 cost measurement never does (ST7-01).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data.db import PROJECT_ROOT
from core_pipeline.forecast import prediction_store as quantile_store
from core_pipeline.forecast.common import file_sha256
from core_pipeline.forecast.tsmixer.config import QUANTILES_9
from core_pipeline.policies.lp_policy import ReplayError

AGENT_DIR = PROJECT_ROOT / "outputs" / "predictions_agent"
AGENT_PURPOSE = "agent_candidate_selection_replay"
MODEL = "tsmixer_a3"
SCENARIO = "agent"
FORECAST_DAYS = 28
N_SCEN = 9


class AgentSelectionReplayer:
    """Strict replay of one fold's weekly-selection quantile parquet."""

    def __init__(
        self,
        fold: int,
        expected_origins,
        expected_series,
        expected_path_sha: str | None = None,
        base_dir: Path | None = None,
    ):
        base = Path(base_dir) if base_dir is not None else AGENT_DIR
        fname = f"{MODEL}_{SCENARIO}_f{fold}.parquet"
        fpath = base / fname
        if not fpath.exists():
            raise ReplayError(
                f"missing agent selection predictions {fpath}; live model "
                "forward fallback is not allowed")
        man_path = base / f"{MODEL}_{SCENARIO}_manifest.json"
        if not man_path.exists():
            raise ReplayError(f"missing manifest {man_path}")
        manifest = json.loads(man_path.read_text())
        if manifest.get("prediction_purpose") != AGENT_PURPOSE:
            raise ReplayError(
                f"{man_path.name}: prediction_purpose "
                f"{manifest.get('prediction_purpose')!r} != {AGENT_PURPOSE!r}")
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
                "scenario path")

        df = pd.read_parquet(fpath)
        quantile_store.validate_frame(df, QUANTILES_9)
        exp_origins = sorted(int(o) for o in expected_origins)
        got_origins = sorted(int(o) for o in df["origin_idx"].unique())
        if got_origins != exp_origins:
            raise ReplayError(
                f"{fname}: origin set mismatch (got {got_origins}, "
                f"expected {exp_origins})")
        if sorted(entry.get("selection_origins", [])) != exp_origins:
            raise ReplayError(
                f"{fname}: manifest selection_origins "
                f"{entry.get('selection_origins')} != expected {exp_origins}")
        exp_series = sorted(str(s) for s in expected_series)
        got_series = sorted(str(s) for s in df["series_id"].unique())
        if got_series != exp_series:
            raise ReplayError(
                f"{fname}: series set mismatch "
                f"(got {len(got_series)}, expected {len(exp_series)})")

        self.fold = fold
        self.manifest_entry = entry
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
        """(28, n_sku, 9) quantile paths for days origin+1..origin+28."""
        pos = self._origin_pos.get(int(origin_idx))
        if pos is None:
            raise ReplayError(
                f"agent fold {self.fold}: origin {origin_idx} not in the "
                "selection parquet; live forward fallback is not allowed")
        try:
            cols = [self._series_pos[str(s)] for s in series_ids]
        except KeyError as exc:
            raise ReplayError(
                f"agent fold {self.fold}: unknown series "
                f"{exc.args[0]!r}") from None
        return np.transpose(self._arr[pos, cols, :, :], (1, 0, 2)).copy()