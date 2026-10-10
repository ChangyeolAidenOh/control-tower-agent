"""TSMixer-Ext local inference into the shared stores (ST4-06).

Process layout (ST3-06, run_chronos lineage): ALL database work happens
first and the connection is closed BEFORE torch is imported. Phase A
stages demand (<= d_1941 via the single chokepoint), the full published
calendar (<= d_1969), statics, and per-fold origin lists from
FoldDatasetBuilder.iter_test_origins -- the same enumeration LightGBM
and Chronos used, so frames align across models. Phase B loads
checkpoints and predicts on CPU.

Mapping and integrity (ST4-04, all hard fail, never warn-and-continue):
fold -> checkpoint via checkpoint_fold_for (2,3->1; 5,6->4; 8,9->7;
11,12->10); encoder.json of the serving retrain bundle re-hashed and
compared to every checkpoint manifest; load_checkpoint verifies schema,
identity, input_schema_hash against local adapter code, state_dict sha
and the stored probe within tolerance; checkpoints sharing a retrain
fold must carry identical scale vectors (parity).

Outputs (ST4-01 P0): a1 / a2_r3 / a2_r5 -> point_store as
tsmixer_<key> with their prediction_kind; a3 -> 9-quantile frames
(inverse transform -> clamp -> monotone rearrangement; crossing rate
recorded per fold) into prediction_store with quantiles=QUANTILES_9 and
a decile9 manifest. Evaluation at r=10 reuses a2_r5 output; no r=10
checkpoint exists by contract.

Measurement before the full run (approval discipline):
  python -m scripts.run_tsmixer --folds 1 --origins-limit 1
  python -m scripts.run_tsmixer --folds 1 --origins-limit 3
then the full run. If the extrapolated budget is exceeded, STOP and
report; a reduced protocol must be fixed in advance.
"""

import argparse
import json
import time
from pathlib import Path

from core_pipeline.data import db
from core_pipeline.forecast import point_store
from core_pipeline.forecast import prediction_store as qstore
from core_pipeline.forecast.dataset import (
    DatasetConfig, FoldDatasetBuilder, MartPanelSource,
)
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND,
    QUANTILES_9,
    TSMixerConfig,
    checkpoint_fold_for,
    checkpoint_key,
)
from core_pipeline.forecast.tsmixer.staging import _canonical_hash

DEFAULT_MODELS = "a1,a2_r3,a2_r5,a3"


def parse_models(spec: str) -> list[tuple[str, int | None]]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if "_r" in token:
            mid, r = token.split("_r")
            out.append((mid, int(r)))
        else:
            out.append((token, None))
    for mid, r in out:
        checkpoint_key(mid, r)  # validates pairing, rejects r=10
    return out


def stage_from_db(fold_list: list[int]):
    """Phase A: everything that touches PostgreSQL; then close."""
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    builder = FoldDatasetBuilder(DatasetConfig(), source)
    dp = source.load_demand_price(max_period_idx=source.max_demand_idx())
    cal = source.load_calendar(1, source.max_calendar_idx())
    static = source.load_static()
    origins_by_fold = {f: list(builder.iter_test_origins(f))
                       for f in fold_list}
    max_cal = source.max_calendar_idx()
    conn.close()
    return dp, cal, static, origins_by_fold, max_cal


def load_encoder(staging_root: Path, retrain_fold: int) -> dict:
    path = staging_root / f"fold_{retrain_fold:02d}" / "encoder.json"
    enc = json.loads(path.read_text())
    recomputed = _canonical_hash(
        {k: enc[k] for k in ("event_type", "static", "cardinalities")})
    if recomputed != enc["hash"]:
        raise ValueError(f"encoder.json content/hash mismatch at {path}")
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1-12")
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS)
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--staging-root", default="outputs/staging/tsmixer")
    ap.add_argument("--origins-limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    lo, hi = (args.folds.split("-") + [args.folds.split("-")[0]])[:2]
    fold_list = list(range(int(lo), int(hi) + 1))
    models = parse_models(args.models)

    # Phase A: all DB work, connection closed before any torch import.
    t0 = time.time()
    dp, cal, static, origins_by_fold, max_cal = stage_from_db(fold_list)
    if args.origins_limit:
        origins_by_fold = {f: o[:args.origins_limit]
                           for f, o in origins_by_fold.items()}
    print(f"staged rows={len(dp)} folds={fold_list} "
          f"stage_s={time.time() - t0:.0f}")

    # Phase B: torch only from here on.
    import numpy as np
    from core_pipeline.forecast.metrics import monotone_rearrange  # noqa: F401
    from core_pipeline.forecast.tsmixer.checkpoint import load_checkpoint
    from core_pipeline.forecast.tsmixer.infer import (
        inference_pairs, long_arrays, predict,
    )
    from core_pipeline.forecast.tsmixer.panel import (
        build_panel, build_static_inputs,
    )
    from core_pipeline.forecast.tsmixer.postproc import finalize_quantiles

    cfg = TSMixerConfig()
    ckpt_dir = Path(args.ckpt_dir)
    staging_root = Path(args.staging_root)
    entries = {checkpoint_key(m, r): [] for m, r in models}
    runtime = {"device": "cpu", "batch_size": args.batch_size}

    by_retrain: dict[int, list[int]] = {}
    for fold in fold_list:
        by_retrain.setdefault(checkpoint_fold_for(fold), []).append(fold)

    for retrain_fold, folds in sorted(by_retrain.items()):
        encoder = load_encoder(staging_root, retrain_fold)
        loaded = {}
        ref_scale = None
        for mid, r in models:
            model, aux, manifest = load_checkpoint(
                ckpt_dir, mid, retrain_fold, r,
                expected_encoder_hash=encoder["hash"])
            if ref_scale is None:
                ref_scale = aux["scale"]
            elif not np.array_equal(ref_scale, aux["scale"]):
                raise ValueError(
                    f"scale mismatch across checkpoints at retrain fold "
                    f"{retrain_fold} ({checkpoint_key(mid, r)})")
            loaded[checkpoint_key(mid, r)] = (model, aux, manifest)
        first_aux = next(iter(loaded.values()))[1]
        panel = build_panel(dp, cal, static, encoder,
                            first_aux["scale"], first_aux["price_mean"],
                            idx_start=1, idx_end=max_cal)
        static_in = build_static_inputs(static, encoder)

        for fold in folds:
            origins = origins_by_fold[fold]
            rows, orgs = inference_pairs(panel, origins, cfg)
            t_fold = time.time()
            for key, (model, aux, manifest) in loaded.items():
                raw = predict(model, panel, static_in, rows, orgs, cfg,
                              batch_size=args.batch_size)
                sid, org_l, hor = long_arrays(panel, rows, orgs, cfg)
                meta = {
                    "checkpoint_key": key,
                    "retrain_fold": retrain_fold,
                    "state_dict_sha256": manifest["state_dict_sha256"],
                    "probe_max_abs_diff": aux["probe_max_abs_diff"],
                    "encoder_hash": encoder["hash"],
                    "n_origins": len(origins),
                }
                if key == "a3":
                    fixed, crossing = finalize_quantiles(raw)
                    frame = qstore.build_frame(
                        sid, org_l, hor,
                        fixed.reshape(-1, len(QUANTILES_9)), QUANTILES_9)
                    entry = qstore.write_fold(
                        "tsmixer_a3", fold, frame,
                        {**meta, "crossing_rate_before": crossing},
                        quantiles=QUANTILES_9)
                else:
                    frame = point_store.build_point_frame(
                        sid, org_l, hor, raw.reshape(-1))
                    entry = point_store.write_point_fold(
                        f"tsmixer_{key}", fold, frame,
                        MODEL_KIND[key.split("_")[0]], meta)
                entries[key].append(entry)
            print(f"fold {fold:02d} ckpt_f{retrain_fold} "
                  f"origins={len(origins)} "
                  f"fold_s={time.time() - t_fold:.0f}")

    for key, folds_entries in entries.items():
        if key == "a3":
            qpath = point_store.write_quantile_manifest(
                "tsmixer_a3", folds_entries, runtime,
                QUANTILES_9, "decile9")
        else:
            qpath = point_store.write_point_manifest(
                f"tsmixer_{key}", folds_entries, runtime,
                MODEL_KIND[key.split("_")[0]])
        print(f"manifest {qpath.name} folds={len(folds_entries)}")


if __name__ == "__main__":
    main()
