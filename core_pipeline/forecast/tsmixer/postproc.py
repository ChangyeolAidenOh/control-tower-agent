"""Inference post-processing (ST4-01 D2).

A3 pipeline: inverse transform to raw units -> clamp at zero -> monotone
rearrangement (sort over the quantile axis). The SAA LP consumes only the
rearranged grid. The crossing rate is measured on the raw pre-clamp,
pre-sort outputs and reported per fold (clamping is monotone and can only
remove crossings, so measuring first is the conservative figure).

Point models (A1/A2/B) need no step here: softplus in the head already
guarantees non-negativity, shared across the three.
"""

import numpy as np


def finalize_quantiles(q_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """q_raw (N, H, 9) raw units, pre-rearrangement.

    Returns (q_final, crossing_rate) where crossing_rate is the fraction of
    (sample, horizon) rows with any adjacent inversion before processing."""
    if q_raw.ndim != 3:
        raise ValueError("expected (N, H, Q)")
    crossing = float(np.mean(np.any(np.diff(q_raw, axis=-1) < 0.0, axis=-1)))
    q = np.sort(np.maximum(q_raw, 0.0), axis=-1)
    return q, crossing
