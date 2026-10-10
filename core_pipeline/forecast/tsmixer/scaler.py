"""Per-series mean scaling (ST4-01 D1, accepted with conditions).

s_i = max(mean of d_{i,t} over the active span up to the retrain cutoff, eps),
zero-sales days included. Rules:
  - the scale is computed from training-period data only;
  - it may be refreshed only at retrain folds (1, 4, 7, 10);
  - non-retrain folds reuse the previous model together with its scale;
  - predictions are inverse-transformed to raw demand units before any loss
    or LP use (all losses are computed in raw units).

eps is a fixed constant guarding division by zero. Default 1.0 also floors
the scale for very-slow movers (sub-unit daily means are not inflated);
recorded in the manifest.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SeriesMeanScaler:
    eps: float = 1.0

    scale_: np.ndarray | None = None

    def fit(self, sales: np.ndarray, train_active_mask: np.ndarray) -> "SeriesMeanScaler":
        """sales (S, T) raw units; train_active_mask (S, T) True where the
        period is both active and at or before the retrain cutoff."""
        if sales.shape != train_active_mask.shape:
            raise ValueError("sales and mask shapes differ")
        mask = train_active_mask.astype(bool)
        counts = mask.sum(axis=1)
        sums = np.where(mask, sales, 0.0).sum(axis=1)
        means = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
        means = np.where(counts > 0, means, 0.0)
        self.scale_ = np.maximum(means, self.eps)
        return self

    def _require_fit(self) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return self.scale_

    def transform(self, sales: np.ndarray) -> np.ndarray:
        return sales / self._require_fit()[:, None]

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self._require_fit()[:, None]
