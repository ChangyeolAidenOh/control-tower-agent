"""Export TSMixer staging bundles for Colab training (ST4-04, DB phase).

Pure Phase A: PostgreSQL only, no torch anywhere in this import chain
(ST3-06). One bundle per retrain fold (1, 4, 7, 10) under
outputs/staging/tsmixer/fold_XX/; demand and price are physically
truncated at each fold's train_end through the single chokepoint
load_demand_price(max_period_idx), and the fold-1 bundle additionally
carries the tune-time scalers and tuning economic weights as-of d_1437
(ST4-04 P0-02). Economic parameters are loss/LP metadata only.

Run:
  python -m scripts.export_tsmixer_staging
  python -m scripts.export_tsmixer_staging --folds 1
"""

import argparse
from pathlib import Path

import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast.dataset import DatasetConfig, MartPanelSource
from core_pipeline.forecast.tsmixer import staging


def _read(conn, query: str) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(query)
    columns = [c[0] for c in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=columns)
    cur.close()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/staging/tsmixer")
    parser.add_argument("--folds", default=None,
                        help="comma-separated retrain folds; default all")
    args = parser.parse_args()
    retrain_folds = ([int(x) for x in args.folds.split(",")]
                     if args.folds else None)

    db.load_dotenv()
    conn = db.connect(db.load_config())
    try:
        source = MartPanelSource(conn)
        econ = _read(conn, "SELECT * FROM mart.sku_cost ORDER BY series_id")
        missing = [c for c in staging.ECON_REQUIRED if c not in econ.columns]
        if missing:
            raise ValueError(
                f"mart.sku_cost lacks {missing}; columns = {list(econ.columns)}")
        for col in staging.ECON_REQUIRED[1:]:
            econ[col] = pd.to_numeric(econ[col], errors="raise")
        bundles = _read(conn, "SELECT * FROM mart.bundles ORDER BY group_id")
        manifests = staging.export_staging(
            source, econ, bundles, Path(args.out),
            tune_train_end=DatasetConfig().tune_train_target_end,
            retrain_folds=retrain_folds)
    finally:
        conn.close()

    for m in manifests:
        print(f"fold {m['fold']:02d} train_end {m['train_end_idx']} "
              f"files {len(m['files'])}")
    print(f"encoder canonical / input schema hash: "
          f"{manifests[0]['encoder_hash']} / "
          f"{manifests[0]['input_schema_hash']}")


if __name__ == "__main__":
    main()
