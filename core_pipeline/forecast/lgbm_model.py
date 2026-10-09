"""LightGBM global quantile model (Stage 3-B, decision ST3-03).

Contract highlights:
- 11 quantiles QUANTILES = {0.05, 0.1..0.9, 0.95}; one model per quantile,
  one shared hyperparameter set. Deciles align with A3's SAA scenarios
  (preregistration SS4); 0.05/0.95 form the central 90% interval base.
  Which model's intervals feed H2 is fixed in Stage 4 -- LightGBM
  intervals are not H2's input by default.
- Tuning protocol (4 steps, SS2 "retrain yes, re-tune no"):
  (1) pick the pre-generated trial whose validation WQL over the
      representative quantiles {0.1, 0.5, 0.9} is lowest;
  (2) with the winning params train all 11 quantiles once on the
      pre-fold-1 split to fix per-quantile best_iteration;
  (3) retrain folds 1/4/7/10 to each cutoff with the same params and the
      FIXED per-quantile iteration counts;
  (4) no early stopping or parameter reselection ever touches evaluation
      data.
- The number of executed trials is a prefix of the pre-generated order,
  decided by measured compute cost against a declared time budget --
  never by validation performance (the rule and measurements are logged).
- Reproducibility: seed 101, deterministic=True + force_col_wise=True,
  fixed num_threads, bagging_freq=1 so bagging_fraction is active.
  deterministic=True does NOT guarantee identical results across LightGBM
  versions or platforms; the manifest records versions for that reason.
- Models are saved (text) with sha256 hashes plus a manifest; a reload
  equality check guards the save/load round trip.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np

from core_pipeline.forecast.common import QUANTILES, file_sha256, write_json
from core_pipeline.forecast.dataset import TrainFrame
from core_pipeline.forecast.metrics import monotone_rearrange, wql

__all__ = ["QUANTILES", "file_sha256", "write_json"]  # re-exported
REP_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
TRAIN_SEED = 101          # seed A, simulator_spec v1 SS12
SEARCH_SEED = 20261009    # pre-generated trial order


def base_params(tau: float, num_threads: int = 4) -> dict:
    return {
        "objective": "quantile",
        "alpha": float(tau),
        "metric": "quantile",
        "seed": TRAIN_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(num_threads),
        "bagging_freq": 1,
        "verbosity": -1,
    }


def sample_trials(n: int, seed: int = SEARCH_SEED) -> list[dict]:
    """Deterministic pre-generated search order. Space:
    num_leaves 32..256 (log2-uniform), learning_rate 0.02..0.2
    (log-uniform), min_data_in_leaf in {50,100,200,500,1000},
    feature_fraction / bagging_fraction 0.6..1.0, lambda_l2 1e-3..10
    (log-uniform). bagging_freq=1 comes from base_params."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        trials.append({
            "num_leaves": int(round(2 ** rng.uniform(5.0, 8.0))),
            "learning_rate": float(10 ** rng.uniform(
                np.log10(0.02), np.log10(0.2))),
            "min_data_in_leaf": int(rng.choice([50, 100, 200, 500, 1000])),
            "feature_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "lambda_l2": float(10 ** rng.uniform(-3.0, 1.0)),
        })
    return trials


def select_best_trial(results: list[dict]) -> dict:
    """Min validation score; ties resolved by pre-generated order (first
    wins) -- deterministic, never performance-dependent re-ranking."""
    defined = [r for r in results if np.isfinite(r["score"])]
    if not defined:
        raise ValueError("no trial produced a finite score")
    return min(defined, key=lambda r: (r["score"], r["trial"]))


def make_dataset(frame: TrainFrame,
                 reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
    return lgb.Dataset(
        frame.X,
        label=frame.y,
        categorical_feature=list(frame.categorical),
        reference=reference,
        free_raw_data=True,
    )


def train_quantile(
    train_ds: lgb.Dataset,
    tau: float,
    params_extra: dict,
    num_boost_round: int,
    num_threads: int = 4,
    valid_ds: Optional[lgb.Dataset] = None,
    early_stopping_rounds: Optional[int] = None,
):
    """Returns (booster, used_iterations). With valid_ds + early stopping
    (tuning phase only), used_iterations = booster.best_iteration; fold
    retraining passes the fixed iteration count and NO validation set."""
    params = {**base_params(tau, num_threads), **params_extra}
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if valid_ds is not None:
        valid_sets = [valid_ds]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds,
                                                verbose=False))
    booster = lgb.train(
        params, train_ds, num_boost_round=num_boost_round,
        valid_sets=valid_sets, callbacks=callbacks,
    )
    used = booster.best_iteration or num_boost_round
    return booster, int(used)


def predict_matrix(
    boosters: dict, X, quantiles: Sequence[float] = QUANTILES
) -> tuple[np.ndarray, np.ndarray, float]:
    """(raw, rearranged, crossing_rate). Rearrangement (per-row sort) is
    the fixed post-processing for every evaluation and interval use; the
    pre-correction crossing rate is reported alongside (ST3-03)."""
    raw = np.column_stack([
        boosters[q].predict(
            X, num_iteration=getattr(boosters[q], "best_iteration", None))
        for q in quantiles
    ])
    rearranged, crossing_rate = monotone_rearrange(raw)
    return raw, rearranged, crossing_rate


def valid_wql(frame: TrainFrame, boosters: dict,
              quantiles: Sequence[float]) -> tuple[float, float]:
    """(post-rearrangement WQL, pre-correction crossing rate)."""
    _, rearranged, crossing = predict_matrix(boosters, frame.X, quantiles)
    return wql(frame.y, rearranged, quantiles), crossing


def save_booster(booster: lgb.Booster, path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return file_sha256(path)


def reload_max_abs_diff(booster: lgb.Booster, path: str | Path,
                        X_sample) -> float:
    """Save/load round-trip guard: max |pred(saved) - pred(live)| on a
    sample. Expected ~0; the fold runner fails if > 1e-9."""
    reloaded = lgb.Booster(model_file=str(path))
    a = booster.predict(
        X_sample, num_iteration=getattr(booster, "best_iteration", None))
    b = reloaded.predict(X_sample)
    return float(np.max(np.abs(a - b))) if len(a) else 0.0


def runtime_manifest(**extra) -> dict:
    return {
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        **extra,
    }


