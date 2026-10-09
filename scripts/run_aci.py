"""Run the delayed adaptive CQR margin replay over stored predictions
(Stage 3-D, ST3-07) for both models: lgbm and chronos2s.

Both are Stage 3 reference analyses; the H2 primary input model is fixed
in Stage 4. Replay state is one online process over origins d_1605..
d_1940 (fold continuity); feedback for a bundle issued at origin o is
applied only when the clock reaches o + 28. Truth flows through
DemandHistory (observed_through guard), never read ahead from parquet.

Outputs per model:
  outputs/intervals/{model}_aci.parquet
    series_id, origin_idx, horizon, target_idx, lo_base, hi_base,
    lo_adj, hi_adj, m_at_origin, updates_before
  outputs/intervals/{model}_aci_summary.json
    PRIMARY coverage = full period over E = {target_idx <= 1941}
    (cold-start rows included); cold-start-excluded figures are
    auxiliary diagnostics only, per the H2 evaluation freeze.

Run: python -m scripts.run_aci
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import aci
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import write_json
from core_pipeline.forecast.dataset import DemandHistory, MartPanelSource

OUT_DIR = Path("outputs/intervals")
MODELS = ("lgbm", "chronos2s")
LAST_TRUTH = 1941


def stage_from_db() -> tuple[DemandHistory, pd.DataFrame]:
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    full = DemandHistory.from_source(source, max_period_idx=LAST_TRUTH)
    folds = source.load_folds()
    conn.close()
    return full, folds


def load_model_frame(model: str, folds: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold in folds["fold"]:
        df = store.load_fold(model, int(fold))
        df["fold"] = int(fold)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["origin_idx", "series_id", "horizon"],
                          ignore_index=True)


def to_cube(df: pd.DataFrame, col: str, origins, series) -> np.ndarray:
    return df[col].to_numpy(dtype=np.float64).reshape(
        len(origins), len(series), aci.HORIZON)


def summarize(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict:
    d = df[mask]
    return {
        "scope": name,
        "rows": int(len(d)),
        "coverage_before": float(d["cov_before"].mean()),
        "coverage_after": float(d["cov_after"].mean()),
        "mean_width_before": float(d["width_before"].mean()),
        "mean_width_after": float(d["width_after"].mean()),
    }


def main() -> None:
    full, folds = stage_from_db()
    series = list(full.series_ids)
    truth = full.observed_through(LAST_TRUTH)

    for model in MODELS:
        t0 = time.time()
        df = load_model_frame(model, folds)
        origins = np.sort(df["origin_idx"].unique())
        exp_series = sorted(df["series_id"].unique())
        if exp_series != sorted(series):
            raise SystemExit(f"{model}: series mismatch vs mart")
        # replay history: same demand matrix, clock opened at origins[0]
        hist = DemandHistory(series, full.idx_start,
                             full.observed_through(LAST_TRUTH),
                             current_idx=int(origins[0]))
        # cube axes must match history.series_ids order
        df["series_rank"] = df["series_id"].map(
            {s: i for i, s in enumerate(series)})
        df = df.sort_values(["origin_idx", "series_rank", "horizon"],
                            ignore_index=True)
        lo = to_cube(df, "q0.05", origins, series)
        hi = to_cube(df, "q0.95", origins, series)
        res = aci.replay(origins, lo, hi, hist)

        out = df[["series_id", "origin_idx", "horizon", "target_idx",
                  "fold"]].copy()
        out["lo_base"] = df["q0.05"].astype(np.float64)
        out["hi_base"] = df["q0.95"].astype(np.float64)
        for key in ("lo_adj", "hi_adj", "m_at_origin"):
            out[key] = res[key].reshape(-1)
        out["updates_before"] = res["updates_before"].reshape(-1)

        evalable = out["target_idx"].to_numpy() <= LAST_TRUTH
        tcol = out["target_idx"].to_numpy() - full.idx_start
        srow = df["series_rank"].to_numpy()
        y = np.where(evalable, truth[srow, np.clip(tcol, 0, truth.shape[1] - 1)],
                     np.nan)
        lo_b = np.maximum(out["lo_base"].to_numpy(), 0.0)
        hi_b = np.maximum(out["hi_base"].to_numpy(), lo_b)
        out["cov_before"] = (y >= lo_b) & (y <= hi_b)
        out["cov_after"] = ((y >= out["lo_adj"].to_numpy())
                            & (y <= out["hi_adj"].to_numpy()))
        out["width_before"] = hi_b - lo_b
        out["width_after"] = out["hi_adj"] - out["lo_adj"]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{model}_aci.parquet"
        out.drop(columns=["cov_before", "cov_after", "width_before",
                          "width_after"]).to_parquet(path, index=False)

        warm = out["updates_before"].to_numpy() > 0
        summary = {
            "model": model,
            "method": ("delayed adaptive CQR margin tracker "
                       "(online quantile tracking / conformal-PID "
                       "proportional term; NOT Gibbs-Candes ACI, no "
                       "inherited coverage guarantee)"),
            "params": {
                "alpha": aci.ALPHA, "gamma": aci.GAMMA,
                "m_bounds": [aci.M_MIN, aci.M_MAX],
                "release_lag_days": aci.RELEASE_LAG,
                "eps_rule": "max(1.0, 0.1 * selection-window mean demand)",
                "gamma_note": "pre-fixed project choice, not a "
                              "literature-standard default",
            },
            "primary_full_period": summarize(out, evalable, "target<=1941"),
            "aux_cold_start_excluded": summarize(
                out, evalable & warm, "target<=1941 & updates_before>0"),
            "by_fold": [summarize(out, evalable
                                  & (out["fold"].to_numpy() == f),
                                  f"fold {f}")
                        for f in sorted(out["fold"].unique())],
            "by_horizon": [summarize(out, evalable
                                     & (out["horizon"].to_numpy() == h),
                                     f"h {h}")
                           for h in range(1, 29)],
            "tracker": {
                "origins_released": res["origins_released"],
                "origins_unreleased": res["origins_unreleased"],
                "m_final_mean": float(res["m_final"].mean()),
                "m_final_max": float(res["m_final"].max()),
                "m_final_min": float(res["m_final"].min()),
                "m_final_mean_by_h": [float(v) for v in
                                      res["m_final"].mean(axis=0)],
            },
            "prediction_store_files_sha": {
                str(f): store.file_sha256(
                    store.PRED_DIR / f"{model}_f{f}.parquet")
                for f in sorted(out["fold"].unique())
            },
        }
        write_json(OUT_DIR / f"{model}_aci_summary.json", summary)
        print(f"{model}: rows={len(out)} "
              f"cov_before={summary['primary_full_period']['coverage_before']:.4f} "
              f"cov_after={summary['primary_full_period']['coverage_after']:.4f} "
              f"m_mean={summary['tracker']['m_final_mean']:.4f} "
              f"t_s={time.time() - t0:.0f}")
    print("wrote outputs/intervals/*")


if __name__ == "__main__":
    main()
