"""Chronos-2 small reference predictions into the shared store (ST3-05).

Reference model only (NOT an H1 comparator): reported as the performance
of a publicly pretrained model. The Chronos public pretraining corpus
links a collection that contains M5, and absence of overlap for this
checkpoint cannot be guaranteed -- never describe these results as
verified leakage-free zero-shot.

Design (ST3-05): predict_df with known-future covariates built from ONE
calendar source split at the origin (wday, month, snap, is_event,
event_type string); target only in the historical frame; future frame =
exactly 28 days; cross_learning=False pinned; context_length=None
(history fits the 8192 cap); CPU. Output `predictions` is the median
under an internal name and is ignored; the 11 quantile columns are
rearranged with the fixed post-processing and stored (point = q0.50).

Measurement before the full run (approval condition 3):
  python -m scripts.run_chronos --origins-limit 1
  python -m scripts.run_chronos --origins-limit 3
then the full run with no limit. If the extrapolated budget is exceeded,
STOP and report -- a reduced evaluation protocol must be fixed in
advance, never chosen after seeing results.

Process layout: ALL database work happens first and the connection is
closed BEFORE chronos/torch is imported. A native crash was observed on
macOS when psycopg2 and torch shared a process with torch loaded first;
phase separation removes any window where the two native runtimes
interleave (lightgbm is likewise kept out of this import chain).
"""

import argparse
import hashlib
import resource
import sys
import time

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import chronos_support as cs
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)
from core_pipeline.forecast.metrics import monotone_rearrange

MODEL_ID = "autogluon/chronos-2-small"


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def state_dict_sha(model) -> str:
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        h.update(name.encode())
        h.update(sd[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def stage_from_db(fold_list: list[int], origins_limit: int) -> tuple[pd.DataFrame, dict]:
    """Phase A: everything that touches PostgreSQL. Returns one global
    staging panel (demand <= d_1941 as `target`, calendar covariates to
    d_1969) plus per-fold origin lists, then the connection is closed.
    Exposure control stays at split_frames(origin): the model only ever
    receives target rows <= origin (asserted there), so a single panel
    serves every fold."""
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    builder = FoldDatasetBuilder(DatasetConfig(), source)
    dp = source.load_demand_price(max_period_idx=source.max_demand_idx())
    dp = dp.rename(columns={"sales": "target"})[
        ["series_id", "period_idx", "target"]]
    cal = source.load_calendar(1, source.max_calendar_idx())
    static = source.load_static()[["series_id", "state_id"]]
    cal = cal.merge(static, on="state_id")[
        ["series_id", "period_idx", *cs.COVARIATE_COLUMNS]]
    panel = cal.merge(dp, on=["series_id", "period_idx"], how="left")
    origins_by_fold = {}
    for fold in fold_list:
        origins = list(builder.iter_test_origins(fold))
        origins_by_fold[fold] = (origins[:origins_limit] if origins_limit
                                 else origins)
    conn.close()
    return panel, origins_by_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1-12")
    ap.add_argument("--origins-limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))

    # Phase A: all DB work, connection closed before any torch import.
    t0 = time.time()
    panel, origins_by_fold = stage_from_db(fold_list, args.origins_limit)
    print(f"staged panel rows={len(panel)} folds={fold_list} "
          f"stage_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")

    # Phase B: torch/chronos only from here on.
    from chronos import Chronos2Pipeline  # lazy: heavy import
    import chronos
    import torch

    t0 = time.time()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    model_sha = state_dict_sha(pipeline.model)
    print(f"loaded {MODEL_ID} load_s={time.time() - t0:.0f} "
          f"sha={model_sha} rss_mb={rss_mb():.0f}")

    entries = []
    cov_hash = None
    for fold in fold_list:
        origins = origins_by_fold[fold]
        t_fold = time.time()
        sids, orgs, hors, raws, fixed = [], [], [], [], []
        for i, origin in enumerate(origins):
            t1 = time.time()
            hist_df, future_df = cs.split_frames(panel, origin, horizon=28)
            hist_df = hist_df.dropna(subset=["target"])
            if cov_hash is None:
                cov_hash = store.covariate_schema_hash(
                    future_df.columns, future_df.dtypes)
            out = pipeline.predict_df(
                hist_df, future_df=future_df,
                id_column="series_id", timestamp_column="timestamp",
                target="target", prediction_length=28,
                quantile_levels=list(QUANTILES),
                batch_size=args.batch_size, cross_learning=False, freq="D",
            )
            sid, hor, raw = cs.extract_quantile_matrix(
                out, origin, QUANTILES)
            fixed_m, _ = monotone_rearrange(raw)
            sids.append(sid)
            orgs.append(np.full(len(sid), origin, dtype=np.int64))
            hors.append(hor)
            raws.append(raw)
            fixed.append(fixed_m)
            if args.origins_limit:
                print(f"fold {fold} origin {origin} "
                      f"({i + 1}/{len(origins)}) t_s={time.time() - t1:.1f} "
                      f"rss_mb={rss_mb():.0f}")
        raw_all = np.concatenate(raws)
        df = store.build_frame(
            np.concatenate(sids), np.concatenate(orgs),
            np.concatenate(hors), np.concatenate(fixed),
        )
        if args.origins_limit:
            per_origin = (time.time() - t_fold) / len(origins)
            print(f"fold {fold} measured per-origin {per_origin:.1f}s -> "
                  f"full-run estimate {per_origin * 28 * 12 / 60:.0f} min; "
                  f"rows={len(df)} (measurement mode: not written)")
            continue
        entry = store.write_fold("chronos2s", fold, df, {
            "covariate_schema_hash": cov_hash,
            **store.crossing_stats(raw_all),
        })
        entries.append(entry)
        print(f"fold {fold} rows={entry['rows']} "
              f"crossing={entry['crossing_rate_before']:.4f} "
              f"t_s={time.time() - t_fold:.0f} rss_mb={rss_mb():.0f}")

    if entries:
        path = store.write_manifest("chronos2s", entries, {
            "model_id": MODEL_ID,
            "model_state_dict_sha256": model_sha,
            "chronos_version": getattr(chronos, "__version__", "?"),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": str(next(pipeline.model.parameters()).dtype),
            "context_length": None,
            "batch_size": args.batch_size,
            "cross_learning": False,
            "covariate_schema_hash": cov_hash,
            "overlap_caveat": (
                "public pretraining corpus links a collection containing "
                "M5; absence of overlap not guaranteed -- reference "
                "performance only, not verified zero-shot"
            ),
        })
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
