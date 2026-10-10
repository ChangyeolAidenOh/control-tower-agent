"""Staging bundles for Colab training (ST4-04). Torch-free (DB phase).

One bundle per retrain fold under <out_root>/fold_XX/. Demand and price are
physically truncated at the fold's train_end via the single leakage
chokepoint (PanelSource.load_demand_price(max_period_idx)), so observations
past the cutoff are absent from anything that leaves the machine. The
calendar slice also stops at train_end (training origins satisfy
t + 28 <= train_end).

As-of separation (ST4-04 P0-02): the fold-1 bundle carries TWO scaler/
price-normalization sets -- tune (computed through tune_train_end=d_1437,
reused frozen for HP-tuning validation) and final (through train_end) --
plus tuning economic weights h_tune = rho_tune * p_bar_tune computed from
prices listed through d_1437 (rho_tune = 1 / median(p_bar_tune)). Official
cost parameters (mart.sku_cost, selection-window based) are passed through
verbatim and remain the only parameters for final models.

economic_params and bundle capacities are loss/LP metadata only; they are
never model features (ST4-01 section 7).

Integrity: bundle_manifest.json records per-file sha256, byte size, rows
and period ranges. load_bundle() re-hashes every file and raises on any
mismatch -- never warn-and-continue (ST4-04 verdict section 6).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.forecast.tsmixer.config import (
    ADAPTER_VERSION,
    STATIC_CATEGORICALS,
    input_schema_hash,
)

STAGING_SCHEMA_VERSION = "tsmixer_staging_v1"
SCALE_FLOOR = 1.0  # ST4-02: scale_i = max(mean, 1.0), a floor not an addend
PARQUET_FILES = (
    "panel_train.parquet", "known_calendar.parquet",
    "series_static.parquet", "economic_params.parquet", "bundles.parquet",
)
BUNDLE_FILES = PARQUET_FILES + ("scalers.npz", "encoder.json", "fold_meta.json")

ECON_REQUIRED = ("series_id", "p_bar", "listed_days", "h",
                 "b_r3", "b_r5", "b_r10")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def build_encoder(calendar_df: pd.DataFrame, static_df: pd.DataFrame) -> dict:
    """Deterministic vocabularies: event_type over the FULL published
    calendar (known schedule, not demand -- no leakage), 'none' pinned to
    code 0 (adapter contract: 0 = no event); static categoricals over
    load_static(). Identical across bundles; hash compared on load."""
    events = sorted(set(map(str, calendar_df["event_type"])))
    events = ["none"] + [e for e in events if e != "none"]
    enc: dict = {"event_type": {v: i for i, v in enumerate(events)},
                 "static": {}}
    for col in STATIC_CATEGORICALS:
        values = sorted(set(map(str, static_df[col])))
        enc["static"][col] = {v: i for i, v in enumerate(values)}
    enc["cardinalities"] = {
        "event_type": len(enc["event_type"]),
        **{c: len(enc["static"][c]) for c in STATIC_CATEGORICALS},
    }
    enc["hash"] = _canonical_hash(
        {k: enc[k] for k in ("event_type", "static", "cardinalities")})
    return enc


def _active_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1)
    sums = np.where(mask, values, 0.0).sum(axis=1)
    means = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    return np.where(counts > 0, means, np.nan)


def compute_scalers(dp: pd.DataFrame, series_order: list[str],
                    cutoff: int) -> dict:
    """Per-series demand scale (ST4-02) and price mean, both as-of cutoff.
    scale: active-day mean sales (zero-sales days included), floored at
    SCALE_FLOOR. price_mean: listed-day (price_ffill notna) mean."""
    sub = dp[dp["period_idx"] <= cutoff]
    sales = sub.pivot(index="series_id", columns="period_idx", values="sales")
    sales = sales.reindex(series_order)
    active = sub.assign(_a=sub["is_active"].astype(float)).pivot(
        index="series_id", columns="period_idx", values="_a").reindex(series_order)
    price = sub.pivot(index="series_id", columns="period_idx",
                      values="price_ffill").reindex(series_order)
    s = sales.to_numpy(dtype=np.float64)
    a = np.nan_to_num(active.to_numpy(dtype=np.float64), nan=0.0) > 0
    p = price.to_numpy(dtype=np.float64)
    demand_mean = _active_mean(np.nan_to_num(s, nan=0.0), a)
    if not np.isfinite(demand_mean).all():
        bad = [series_order[i] for i in np.flatnonzero(~np.isfinite(demand_mean))]
        raise ValueError(f"no active days at or before {cutoff}: {bad}")
    scale = np.maximum(demand_mean, SCALE_FLOOR)
    floor_mask = demand_mean < SCALE_FLOOR
    price_mean = _active_mean(np.nan_to_num(p, nan=0.0), np.isfinite(p))
    if not np.isfinite(price_mean).all():
        bad = [series_order[i] for i in np.flatnonzero(~np.isfinite(price_mean))]
        raise ValueError(f"no listed days at or before {cutoff}: {bad}")
    return {
        "cutoff": int(cutoff),
        "scale": scale.astype(np.float64),
        "floor_mask": floor_mask,
        "price_mean": price_mean.astype(np.float64),
        "floor_count": int(floor_mask.sum()),
        "floor_ratio": float(floor_mask.mean()),
    }


def compute_tune_econ(dp: pd.DataFrame, series_order: list[str],
                      tune_end: int) -> pd.DataFrame:
    """Tuning-stage economic weights from prices listed through tune_end
    (ST4-04 P0-02): p_bar_tune (listed-day mean), rho_tune =
    1 / median(p_bar_tune), h_tune = rho_tune * p_bar_tune, b = r * h."""
    sub = dp[(dp["period_idx"] <= tune_end) & dp["price_ffill"].notna()]
    p_bar = sub.groupby("series_id")["price_ffill"].mean().reindex(series_order)
    if p_bar.isna().any():
        raise ValueError(
            f"no listed days at or before {tune_end}: "
            f"{list(p_bar.index[p_bar.isna()])}")
    rho = 1.0 / float(p_bar.median())
    h = rho * p_bar.to_numpy(dtype=np.float64)
    return pd.DataFrame({
        "series_id": series_order,
        "p_bar_tune": p_bar.to_numpy(dtype=np.float64),
        "rho_tune": rho,
        "h_tune": h,
        "b_tune_r3": 3.0 * h,
        "b_tune_r5": 5.0 * h,
    })


def _write_parquet(path: Path, df: pd.DataFrame) -> dict:
    df.to_parquet(path, index=False)
    entry = {"sha256": _sha256(path), "bytes": path.stat().st_size,
             "rows": int(len(df))}
    if "period_idx" in df.columns and len(df):
        entry["period_min"] = int(df["period_idx"].min())
        entry["period_max"] = int(df["period_idx"].max())
    return entry


def write_bundle(out_dir: Path, fold_row: dict, dp: pd.DataFrame,
                 calendar_df: pd.DataFrame, static_df: pd.DataFrame,
                 econ_df: pd.DataFrame, bundles_df: pd.DataFrame,
                 encoder: dict, tune: dict | None) -> dict:
    """tune (fold 1 only): {'tune_end': int, 'econ': DataFrame}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end = int(fold_row["train_end_idx"])
    if int(dp["period_idx"].max()) > train_end:
        raise ValueError("demand/price rows past train_end")
    if int(calendar_df["period_idx"].max()) > train_end:
        raise ValueError("calendar rows past train_end")
    missing = [c for c in ECON_REQUIRED if c not in econ_df.columns]
    if missing:
        raise ValueError(f"economic_params missing columns: {missing}")

    series_order = [str(s) for s in static_df["series_id"]]
    final = compute_scalers(dp, series_order, train_end)
    arrays = {
        "series_id": np.array(series_order),
        "scale_final": final["scale"],
        "floor_mask_final": final["floor_mask"],
        "price_mean_final": final["price_mean"],
    }
    meta_scalers = {"final": {k: final[k] for k in
                              ("cutoff", "floor_count", "floor_ratio")}}
    econ_out = econ_df.copy()
    if tune is not None:
        tune_set = compute_scalers(dp, series_order, int(tune["tune_end"]))
        arrays["scale_tune"] = tune_set["scale"]
        arrays["floor_mask_tune"] = tune_set["floor_mask"]
        arrays["price_mean_tune"] = tune_set["price_mean"]
        meta_scalers["tune"] = {k: tune_set[k] for k in
                                ("cutoff", "floor_count", "floor_ratio")}
        econ_out = econ_out.merge(tune["econ"], on="series_id", validate="1:1")

    manifest_files = {}
    for name, df in (("panel_train.parquet", dp),
                     ("known_calendar.parquet", calendar_df),
                     ("series_static.parquet", static_df),
                     ("economic_params.parquet", econ_out),
                     ("bundles.parquet", bundles_df)):
        manifest_files[name] = _write_parquet(out_dir / name, df)

    np.savez(out_dir / "scalers.npz", **arrays)
    (out_dir / "encoder.json").write_text(json.dumps(encoder, sort_keys=True))
    fold_meta = {
        "fold": int(fold_row["fold"]),
        "train_end_idx": train_end,
        "test_start_idx": int(fold_row["test_start_idx"]),
        "test_end_idx": int(fold_row["test_end_idx"]),
        "warmup_start_idx": int(fold_row["warmup_start_idx"]),
        "scalers": meta_scalers,
        "scale_floor": SCALE_FLOOR,
        "staging_schema_version": STAGING_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "input_schema_hash": input_schema_hash(),
        "encoder_hash": encoder["hash"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if tune is not None:
        fold_meta["tune_train_end_idx"] = int(tune["tune_end"])
    (out_dir / "fold_meta.json").write_text(json.dumps(fold_meta, indent=2))
    for name in ("scalers.npz", "encoder.json", "fold_meta.json"):
        path = out_dir / name
        manifest_files[name] = {"sha256": _sha256(path),
                                "bytes": path.stat().st_size}

    manifest = {
        "fold": int(fold_row["fold"]),
        "train_end_idx": train_end,
        "encoder_hash": encoder["hash"],
        "staging_schema_version": STAGING_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "input_schema_hash": input_schema_hash(),
        "files": manifest_files,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_staging(source, econ_df: pd.DataFrame, bundles_df: pd.DataFrame,
                   out_root: Path, tune_train_end: int,
                   retrain_folds: list[int] | None = None) -> list[dict]:
    """source: PanelSource-shaped object (load_static / load_folds /
    load_demand_price / load_calendar / max_calendar_idx). Demand and price
    pass only through load_demand_price(max_period_idx=train_end)."""
    out_root = Path(out_root)
    static_df = source.load_static()
    folds = source.load_folds()
    retrain = folds[folds["is_retrain"].astype(bool)]
    if retrain_folds is not None:
        retrain = retrain[retrain["fold"].isin(retrain_folds)]
    full_calendar = source.load_calendar(1, source.max_calendar_idx())
    encoder = build_encoder(full_calendar, static_df)

    manifests = []
    for _, row in retrain.sort_values("fold").iterrows():
        train_end = int(row["train_end_idx"])
        dp = source.load_demand_price(max_period_idx=train_end)
        cal = full_calendar[full_calendar["period_idx"] <= train_end]
        tune = None
        if int(row["fold"]) == int(retrain["fold"].min()):
            tune = {"tune_end": int(tune_train_end),
                    "econ": compute_tune_econ(
                        dp, [str(s) for s in static_df["series_id"]],
                        int(tune_train_end))}
        out_dir = out_root / f"fold_{int(row['fold']):02d}"
        manifests.append(write_bundle(
            out_dir, row.to_dict(), dp, cal, static_df,
            econ_df, bundles_df, encoder, tune))
    return manifests


def load_bundle(bundle_dir: Path, expected_encoder_hash: str | None = None,
                expected_input_schema_hash: str | None = None) -> dict:
    """Verify every file hash against bundle_manifest.json, then load.
    Any mismatch raises -- never warn-and-continue."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = bundle_dir / name
        if not path.exists():
            raise ValueError(f"missing bundle file: {name}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"sha256 mismatch for {name}: "
                             f"manifest {entry['sha256'][:12]} != file {actual[:12]}")
    if manifest["staging_schema_version"] != STAGING_SCHEMA_VERSION:
        raise ValueError("staging schema version mismatch")
    encoder = json.loads((bundle_dir / "encoder.json").read_text())
    if expected_encoder_hash and encoder["hash"] != expected_encoder_hash:
        raise ValueError("encoder hash mismatch")
    if (expected_input_schema_hash
            and manifest["input_schema_hash"] != expected_input_schema_hash):
        raise ValueError("input schema hash mismatch")
    out = {
        "manifest": manifest,
        "fold_meta": json.loads((bundle_dir / "fold_meta.json").read_text()),
        "encoder": encoder,
        "scalers": dict(np.load(bundle_dir / "scalers.npz", allow_pickle=False)),
    }
    for name in PARQUET_FILES:
        out[name.removesuffix(".parquet")] = pd.read_parquet(bundle_dir / name)
    return out
