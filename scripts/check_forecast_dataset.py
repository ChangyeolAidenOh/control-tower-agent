"""Stage 3-A smoke check against the real mart (port 5435).

Verifies the dataset contract end to end on live data and prints the
hashes that every Stage 3 output must carry. Also records peak RSS for
the fold-1 full training-frame build (first datapoint of the 3-B staged
memory measurement). Exits non-zero on any violated assertion.

Run: python -m scripts.check_forecast_dataset
"""

import resource
import sys
import time

import numpy as np

from core_pipeline.data import db
from core_pipeline.forecast.dataset import (
    DatasetConfig,
    DemandHistory,
    FoldDatasetBuilder,
    MartPanelSource,
    TARGET_LT_CUM,
)

N_SERIES = 154
MAX_DEMAND = 1941
MAX_CALENDAR = 1969
FOLD1_TRAIN_END = 1605


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def check_inference(builder, source, origin: int) -> None:
    history = DemandHistory.from_source(source, max_period_idx=origin)
    frame = builder.inference_frame(history, origin)
    assert len(frame.X) == N_SERIES * 28, (origin, len(frame.X))
    horizons = sorted(frame.meta["horizon"].unique())
    assert horizons == list(range(1, 29)), (origin, horizons)
    assert int(frame.meta["target_idx"].max()) == origin + 28
    assert len(frame.y) == 0
    print(f"inference origin={origin} rows={len(frame.X)} "
          f"target_max={origin + 28}")


def main() -> None:
    db.load_dotenv()
    conn = db.connect(db.load_config())
    cfg = DatasetConfig()
    source = MartPanelSource(conn)

    assert source.max_demand_idx() == MAX_DEMAND, source.max_demand_idx()
    assert source.max_calendar_idx() == MAX_CALENDAR, source.max_calendar_idx()

    builder = FoldDatasetBuilder(cfg, source)
    print(f"spec_hash={cfg.spec_hash()} "
          f"encoder_hash={builder.encoder.encoder_hash()}")

    static = source.load_static()
    assert len(static) == N_SERIES, len(static)
    print(f"static series={len(static)} rss_mb={rss_mb():.0f}")

    t0 = time.time()
    tf = builder.training_frame(1)
    assert int(tf.meta["target_idx"].max()) == FOLD1_TRAIN_END
    assert int(tf.meta["origin_idx"].max()) == FOLD1_TRAIN_END - 1
    assert list(tf.X.columns) == list(cfg.feature_columns)
    assert len(tf.X) == len(tf.y) == len(tf.meta)
    assert not np.isnan(tf.y).any()
    print(f"fold1 per_horizon rows={len(tf.X)} "
          f"series={tf.meta['series_id'].nunique()} "
          f"elapsed_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")
    del tf

    t0 = time.time()
    tr, va = builder.tuning_split()
    assert int(tr.meta["target_idx"].max()) <= cfg.tune_train_target_end
    assert int(va.meta["target_idx"].min()) >= cfg.tune_valid_target_start
    assert int(va.meta["target_idx"].max()) <= cfg.tune_valid_target_end
    print(f"tuning_split train_rows={len(tr.X)} valid_rows={len(va.X)} "
          f"elapsed_s={time.time() - t0:.0f} rss_mb={rss_mb():.0f}")
    del tr, va

    lt = builder.training_frame(1, target_mode=TARGET_LT_CUM)
    assert (lt.meta["horizon"] == cfg.lead_time_cum_horizon).all()
    assert int(lt.meta["target_idx"].max()) <= FOLD1_TRAIN_END
    assert "tgt_event_type" not in lt.X.columns
    print(f"fold1 lt_cum rows={len(lt.X)}")
    del lt

    for origin in (FOLD1_TRAIN_END, 1914, MAX_DEMAND):
        check_inference(builder, source, origin)

    conn.close()
    print("smoke PASS")


if __name__ == "__main__":
    main()
