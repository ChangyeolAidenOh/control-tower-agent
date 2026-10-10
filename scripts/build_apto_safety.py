"""A-PTO safety-stock quantiles (ST6-11 (3)): per-SKU empirical quantiles
of the 3-day cumulative out-of-sample residual on the tuning validation
window, from the calibration model's predictions.

  e3(i, o) = sum_{h=1..3} y(i, o+h) - sum_{h=1..3} max(0, dhat(i, o+h))
  q_tau(i) = empirical quantile of {e3(i, o)} over validation origins
  tau in {0.75 (r=3), 5/6 (r=5), 10/11 (r=10)}  -- r=10 re-queries tau
  only; the residual distribution is NOT re-estimated (ST6-11 (3)).

Limits recorded in the manifest (ST6-11 (3)): same validation window as
HP selection (no independent-calibration claim), overlapping origins
(no independence claim).

Run: python -m scripts.build_apto_safety
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("outputs/apto/calibration_valid_predictions.parquet")
OUT_PATH = Path("outputs/apto/safety_quantiles.json")
TAUS = {"r3": 0.75, "r5": 5.0 / 6.0, "r10": 10.0 / 11.0}
K = 3  # protection window, D+3 contract (ST6-11 (3))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    y_cols = [f"y_raw_h{j}" for j in range(1, K + 1)]
    f_cols = [f"yhat_raw_h{j}" for j in range(1, K + 1)]
    y3 = df[y_cols].to_numpy(np.float64).sum(axis=1)
    d3 = np.maximum(0.0, df[f_cols].to_numpy(np.float64)).sum(axis=1)
    df = df.assign(e3=y3 - d3)

    out: dict[str, dict[str, float]] = {k: {} for k in TAUS}
    counts: dict[str, int] = {}
    for sid, grp in df.groupby("series_id", sort=True):
        e = np.sort(grp["e3"].to_numpy(np.float64))
        counts[str(sid)] = len(e)
        for key, tau in TAUS.items():
            out[key][str(sid)] = float(np.quantile(e, tau))

    n_set = sorted(set(counts.values()))
    payload = {
        "protocol": "ST6-11 (3)",
        "source_file": IN_PATH.name,
        "source_sha256": file_sha256(IN_PATH),
        "protection_days": K,
        "taus": TAUS,
        "residual_def": ("sum_{h=1..3} y - sum_{h=1..3} max(0, yhat), "
                         "raw demand space, calibration-model predictions "
                         "on tuning validation origins"),
        "n_series": len(counts),
        "samples_per_series": {"min": n_set[0], "max": n_set[-1]},
        "limits": [
            "calibration reuses the HP-selection validation window; "
            "no independent-calibration claim",
            "3-day cumulative residuals overlap across origins; "
            "no independence claim",
            "r=10 re-queries tau=10/11 on the same residual distribution; "
            "no re-estimation (ST6-01 (4) analogue)",
        ],
        "sample_counts": counts,
        "quantiles": out,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1))
    med = {k: float(np.median(list(v.values()))) for k, v in out.items()}
    print(f"series={len(counts)} samples/series="
          f"[{n_set[0]}, {n_set[-1]}] medians "
          f"r3={med['r3']:.3f} r5={med['r5']:.3f} r10={med['r10']:.3f}")
    print(f"written {OUT_PATH}")


if __name__ == "__main__":
    main()