"""Stage 5 (ST5-05/ST5-06): generate the frozen B sample schedule.

Reads the measured budget outputs (outputs/b_lp_measure.json), takes
n_pb = min over r of the frozen-rule values, and writes
outputs/b_sample_schedule.json: per-bundle training origins for the tune
stage (fold 1) and the four final folds, plus the common validation
origins. Deterministic throughout; the file carries its own sha256 and
is kept in git (ST5-02).

Rules implemented exactly as logged:
- eligibility: o in [max_i(first_active_idx) + 56, train_end - 28]
  (tune uses tune_train_end - 28); boundary asserts check the full
  56-day input window and warm-up feasibility per bundle
- training origins: default_rng([20261010, fold, bundle_rank])
  permutation of the eligible range, sorted prefix of length n_pb;
  the origin SET is fixed across epochs (epoch order is drawn at
  training time with default_rng(20261010 + epoch))
- validation: 24 origins evenly spaced over
  [tune_train_end, train_end - 28], common to all bundles and both r

Run: python -m scripts.build_b_sample_schedule
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from core_pipeline.forecast.tsmixer.staging import load_bundle

SEED = 20261010
HISTORY_DAYS = 56
WARMUP_MIN_ORIGIN = 22
N_VALID_PER_BUNDLE = 24
FINAL_FOLDS = (1, 4, 7, 10)
STAGING_ROOT = Path("outputs/staging/tsmixer")
MEASURE_PATH = Path("outputs/b_lp_measure.json")
OUT_PATH = Path("outputs/b_sample_schedule.json")


def o_min_map(static) -> dict[str, int]:
    fa_max = static.groupby("group_id")["first_active_idx"].max()
    return {str(g): int(v) + HISTORY_DAYS for g, v in fa_max.items()}


def check_bounds(static, o_min: dict[str, int]):
    fa_max = static.groupby("group_id")["first_active_idx"].max()
    for gid, om in o_min.items():
        # 56-day input [o-55, o] must start strictly after first activation.
        assert om - (HISTORY_DAYS - 1) > int(fa_max[gid]), gid
        assert om >= WARMUP_MIN_ORIGIN, gid
    sizes = static.groupby("group_id")["series_id"].size()
    return {str(g): int(v) for g, v in sizes.items()}


def bundle_origins(o_min: dict[str, int], o_max: int, n_pb: int,
                   fold: int) -> dict:
    out = {}
    for rank, gid in enumerate(sorted(o_min)):
        eligible = np.arange(o_min[gid], o_max + 1)
        if eligible.size < n_pb:
            raise SystemExit(f"{gid}: eligible {eligible.size} < n_pb {n_pb}")
        rng = np.random.default_rng([SEED, fold, rank])
        origins = np.sort(rng.permutation(eligible)[:n_pb])
        out[gid] = {"o_min": o_min[gid], "o_max": o_max,
                    "eligible": int(eligible.size),
                    "origins": origins.tolist()}
    return out


def main():
    measure = json.loads(MEASURE_PATH.read_text())
    rule_values = {}
    for r in (3, 5):
        key = f"tune_f01_r{r}"
        if key not in measure:
            raise SystemExit(f"measurement missing: {key}")
        rule_values[r] = int(measure[key]["budget"]["n_per_bundle_rule_value"])
    n_pb = min(rule_values.values())
    if n_pb <= 0:
        raise SystemExit("ABORT: n_pb = 0 (ST5-05 stop rule)")

    fold1 = load_bundle(STAGING_ROOT / "fold_01")
    static = fold1["series_static"]
    o_min = o_min_map(static)
    n_sku = check_bounds(static, o_min)
    meta1 = fold1["fold_meta"]
    tune_end = int(meta1["tune_train_end_idx"])
    train_end1 = int(meta1["train_end_idx"])

    valid = sorted({int(round(v)) for v in
                    np.linspace(tune_end, train_end1 - 28,
                                N_VALID_PER_BUNDLE)})
    if len(valid) != N_VALID_PER_BUNDLE:
        raise SystemExit("validation origins collapsed after rounding")

    payload = {
        "seed": SEED,
        "n_per_bundle": n_pb,
        "rule_values_by_r": rule_values,
        "epoch_order_rule": "default_rng(20261010 + epoch) over the flat "
                            "(bundle, origin) list; origin set fixed",
        "validation": {"origins": valid,
                       "range": [tune_end, train_end1 - 28],
                       "scope": "common to all bundles and both r"},
        "n_sku_by_bundle": n_sku,
        "tune": {"fold": 1,
                 "bundles": bundle_origins(o_min, tune_end - 28, n_pb, 1)},
        "final": {},
        "exposures": {},
    }
    for f in FINAL_FOLDS:
        meta = load_bundle(STAGING_ROOT / f"fold_{f:02d}")["fold_meta"]
        o_max_f = int(meta["train_end_idx"]) - 28
        payload["final"][str(f)] = {
            "train_end_idx": int(meta["train_end_idx"]),
            "bundles": bundle_origins(o_min, o_max_f, n_pb, f)}

    def exposures(bundles):
        return {"samples": n_pb * len(bundles),
                "sku_origin_exposures": sum(n_pb * n_sku[g] for g in bundles)}

    payload["exposures"]["tune"] = exposures(payload["tune"]["bundles"])
    for f in FINAL_FOLDS:
        payload["exposures"][f"final_f{f:02d}"] = exposures(
            payload["final"][str(f)]["bundles"])

    canonical = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT_PATH.write_text(json.dumps(payload, indent=1))

    print(json.dumps({"n_per_bundle": n_pb,
                      "valid_origins": len(valid),
                      "tune_samples": payload["exposures"]["tune"]["samples"],
                      "final_folds": list(payload["final"]),
                      "sha256": payload["sha256"][:16]}))


if __name__ == "__main__":
    main()
