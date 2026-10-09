"""Stage 3-F integrated metrics report (ST3-09). No new methodology:
this runner re-aggregates FROZEN artifacts under explicitly labeled
evaluation masks and writes one machine report + one human report.

Masks (every table carries evaluation_mask_id; never mix in one table):
  stage3_in_window  -- ST3-05 scoring freeze: per fold, target_idx in
                       [test_start, test_end]. PRIMARY for daily
                       accuracy. 750,288 rows total; per-h sample
                       collapses with h (h=28 -> 1 origin per fold),
                       so per-h rows carry n_forecasts/n_origins/
                       n_folds and h-curves are NOT pure forecast-
                       distance effects (origin sets differ by h).
  h2_full_period    -- ST3-07 H2 freeze: target_idx <= 1941, all
                       origins, cold start included. PRIMARY for the
                       delayed-adaptation coverage diagnostics.
  observable_aux    -- exploratory: all target_idx <= 1941 for daily
                       accuracy; auxiliary only (does not change the
                       preregistered endpoints).
  lt3_eval          -- Stage 3-E set (o + 3 <= 1941), cited from
                       lt3_eval.json; a DIFFERENT target and sample --
                       never ranked against daily tables.

Scope of claims (fixed): no H1/H2 verdicts here (Stage 6 preregistered
analyses); Chronos-2 rows carry the pretraining-overlap caveat; the
3-E direct model has the lowest cumulative WQL of the three methods --
this does not establish service-level/ordering-cost improvement nor
identify the temporal dependence structure.

Run: python -m scripts.report_stage3
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.data import db
from core_pipeline.forecast import prediction_store as store
from core_pipeline.forecast.common import QUANTILES, write_json
from core_pipeline.forecast.dataset import DemandHistory, MartPanelSource
from core_pipeline.forecast.metrics import (
    macro_mean, mase_table, nonzero_mape, pinball, seasonal_naive_mae,
    wape, wql,
)

DAILY_MODELS = ("lgbm", "chronos2s")
LAST_TRUTH = 1941
JSON_OUT = Path("outputs/reports/stage3_report.json")
MD_OUT = Path("docs/stage3_report.md")
CHRONOS_CAVEAT = ("Chronos-2 Small: public pretraining corpus links a "
                  "collection containing M5; overlap cannot be ruled "
                  "out -- reference performance, not verified zero-shot.")


def load_daily(model: str, folds: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, fr in folds.iterrows():
        df = store.load_fold(model, int(fr["fold"]))
        store.assert_daily_frame(df)
        df["fold"] = int(fr["fold"])
        df["test_start"] = int(fr["test_start_idx"])
        df["test_end"] = int(fr["test_end_idx"])
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def attach_truth(df: pd.DataFrame, truth: np.ndarray, sid_rank: dict,
                 idx_start: int) -> pd.DataFrame:
    t = df["target_idx"].to_numpy()
    ok = t <= LAST_TRUTH
    y = np.full(len(df), np.nan)
    rows = df["series_id"].map(sid_rank).to_numpy()
    y[ok] = truth[rows[ok], t[ok] - idx_start]
    df = df.copy()
    df["y"] = y
    return df


def accuracy_block(d: pd.DataFrame, qcols, mask_id: str,
                   denoms: dict) -> dict:
    y = d["y"].to_numpy()
    qm = d[qcols].to_numpy(dtype=np.float64)
    point = d["q0.50"].to_numpy(dtype=np.float64)
    mape, n_nonzero = nonzero_mape(y, point)
    cell_mases = []
    n_undef_cells = 0
    for f in sorted(d["fold"].unique()):
        dm = d[d["fold"] == f]
        tab = mase_table(dm["y"].to_numpy(), dm["q0.50"].to_numpy(),
                         dm["series_id"], denoms[int(f)])
        cell_mases.append(tab["mase"].to_numpy())
        n_undef_cells += int((~tab["defined"]).sum())
    mase_macro, n_undef = macro_mean(np.concatenate(cell_mases))
    return {
        "evaluation_mask_id": mask_id,
        "rows": int(len(d)),
        "wql": wql(y, qm, QUANTILES),
        "wape": wape(y, point),
        "nonzero_mape": mape,
        "nonzero_mape_rows": n_nonzero,
        "nonzero_mape_excluded_rows": int(len(d) - n_nonzero),
        "mase_macro_fold_series_cells": mase_macro,
        "mase_undefined_cells": n_undef,
    }


def main() -> None:
    db.load_dotenv()
    conn = db.connect(db.load_config())
    source = MartPanelSource(conn)
    full = DemandHistory.from_source(source, max_period_idx=LAST_TRUTH)
    folds = source.load_folds()
    conn.close()

    truth = full.observed_through(LAST_TRUTH)
    sid_rank = {s: i for i, s in enumerate(full.series_ids)}
    qcols = store.quantile_columns()
    # MASE denominators: per fold x series over that fold's TRAIN window
    denoms = {}
    for _, fr in folds.iterrows():
        te = int(fr["train_end_idx"])
        denoms[int(fr["fold"])] = {
            s: seasonal_naive_mae(truth[i, : te - full.idx_start + 1])
            for s, i in sid_rank.items()
        }

    report = {"decision": "ST3-09", "masks": {
        "stage3_in_window": "per fold target_idx in [test_start, "
                            "test_end] (ST3-05 freeze; PRIMARY daily)",
        "h2_full_period": "target_idx <= 1941, cold start included "
                          "(ST3-07 freeze; PRIMARY adaptation)",
        "observable_aux": "target_idx <= 1941 (exploratory auxiliary)",
        "lt3_eval": "o + 3 <= 1941 (Stage 3-E set; separate target)",
    }, "notes": [
        CHRONOS_CAVEAT,
        "No H1/H2 verdicts in Stage 3; those are Stage 6 preregistered "
        "analyses.",
        "Per-h curves evaluate different origin sets per h; not pure "
        "forecast-distance effects.",
        "Stage 3-E direct cumulative model has the lowest cumulative "
        "WQL of the three methods; this does not establish service-"
        "level or ordering-cost improvement, nor identify the temporal "
        "dependence structure.",
    ], "daily": {}, "horizon": {}, "adaptation": {}, "lead_time": {}}

    daily_frames = {}
    for model in DAILY_MODELS:
        df = attach_truth(load_daily(model, folds), truth, sid_rank,
                          full.idx_start)
        daily_frames[model] = df
        in_win = df[(df["target_idx"] >= df["test_start"])
                    & (df["target_idx"] <= df["test_end"])]
        if len(in_win) != 750_288:
            raise SystemExit(f"{model}: stage3 mask rows {len(in_win)} "
                             "!= 750288")
        manifest = json.loads(
            (store.PRED_DIR / f"{model}_manifest.json").read_text())
        cross = [f["crossing_rate_before"] for f in manifest["folds"]]
        entry = {
            "primary": accuracy_block(in_win, qcols, "stage3_in_window",
                                      denoms),
            "aux_observable": accuracy_block(
                df[df["target_idx"] <= LAST_TRUTH], qcols,
                "observable_aux", denoms),
            "by_fold_wql": {
                str(f): wql(g["y"].to_numpy(),
                            g[qcols].to_numpy(dtype=np.float64), QUANTILES)
                for f, g in in_win.groupby("fold")},
            "crossing_rate_before_min_max": [min(cross), max(cross)],
            "runtime": manifest.get("runtime", {}),
        }
        if model == "chronos2s":
            entry["caveat"] = CHRONOS_CAVEAT
        report["daily"][model] = entry

        hor = {}
        for h, g in in_win.groupby("horizon"):
            expected = (29 - int(h)) * 154 * 12
            if len(g) != expected:
                raise SystemExit(f"{model} h={h}: rows {len(g)} != "
                                 f"{expected}")
            hor[str(int(h))] = {
                "wql": wql(g["y"].to_numpy(),
                           g[qcols].to_numpy(dtype=np.float64), QUANTILES),
                "n_forecasts": int(len(g)),
                "n_origins": int(g["origin_idx"].nunique()),
                "n_folds": int(g["fold"].nunique()),
            }
        report["horizon"][model] = {
            "evaluation_mask_id": "stage3_in_window", "by_h": hor}

    for model in DAILY_MODELS:
        iv = pd.read_parquet(f"outputs/intervals/{model}_aci.parquet")
        iv = attach_truth(
            iv.merge(folds[["fold", "test_start_idx", "test_end_idx"]]
                     .rename(columns={"test_start_idx": "test_start",
                                      "test_end_idx": "test_end"}),
                     on="fold"),
            truth, sid_rank, full.idx_start)
        lo_b = np.maximum(iv["lo_base"].to_numpy(), 0.0)
        hi_b = np.maximum(iv["hi_base"].to_numpy(), lo_b)
        y = iv["y"].to_numpy()
        cov_b = (y >= lo_b) & (y <= hi_b)
        cov_a = ((y >= iv["lo_adj"].to_numpy())
                 & (y <= iv["hi_adj"].to_numpy()))
        masks = {
            "h2_full_period": iv["target_idx"].to_numpy() <= LAST_TRUTH,
            "stage3_in_window": (
                (iv["target_idx"] >= iv["test_start"])
                & (iv["target_idx"] <= iv["test_end"])).to_numpy(),
        }
        summ = json.loads(Path(
            f"outputs/intervals/{model}_aci_summary.json").read_text())
        report["adaptation"][model] = {
            "method": summ["method"], "params": summ["params"],
            "feedback_delay_days": 28,
            "intervals_as_issued": True,
            "tracker": summ["tracker"],
            "coverage": {
                mid: {
                    "evaluation_mask_id": mid,
                    "n_scored": int(m.sum()),
                    "coverage_before": float(cov_b[m].mean()),
                    "coverage_after": float(cov_a[m].mean()),
                    "mean_width_before": float((hi_b - lo_b)[m].mean()),
                    "mean_width_after": float(
                        (iv["hi_adj"] - iv["lo_adj"])[m].mean()),
                } for mid, m in masks.items()
            },
        }

    report["lead_time"] = {
        "evaluation_mask_id": "lt3_eval",
        "source": "outputs/reports/lt3_eval.json",
        **json.loads(Path("outputs/reports/lt3_eval.json").read_text()),
    }

    write_json(JSON_OUT, report)
    write_markdown(report)
    for model in DAILY_MODELS:
        p = report["daily"][model]["primary"]
        print(f"{model}: wql={p['wql']:.5f} wape={p['wape']:.5f} "
              f"mase_macro={p['mase_macro_fold_series_cells']:.4f} "
              f"(undef={p['mase_undefined_cells']}) rows={p['rows']}")
    print(f"wrote {JSON_OUT} and {MD_OUT}")


def write_markdown(r: dict) -> None:
    L = []
    add = L.append
    add("# Stage 3 Report -- Probabilistic Forecasting, Delayed "
        "Interval Adaptation, Lead-Time Demand\n")
    add("> Masks are never mixed within a table; each table names its "
        "evaluation_mask_id. No H1/H2 verdicts here (Stage 6).\n")

    add("## 1. Daily probabilistic forecasting "
        "(mask: stage3_in_window, 750,288 rows)\n")
    add("| model | WQL | WAPE | MASE (macro, fold x series) | undefined "
        "cells | nonzero-MAPE (rows) | crossing before (min-max) |")
    add("|---|---|---|---|---|---|---|")
    for m, e in r["daily"].items():
        p = e["primary"]
        cmin, cmax = e["crossing_rate_before_min_max"]
        add(f"| {m} | {p['wql']:.5f} | {p['wape']:.5f} | "
            f"{p['mase_macro_fold_series_cells']:.4f} | "
            f"{p['mase_undefined_cells']} | {p['nonzero_mape']:.4f} "
            f"({p['nonzero_mape_rows']:,}) | {cmin:.3f}-{cmax:.3f} |")
    add(f"\n*{CHRONOS_CAVEAT}*\n")
    add("Auxiliary (observable_aux, exploratory): " + "; ".join(
        f"{m} WQL {e['aux_observable']['wql']:.5f} "
        f"({e['aux_observable']['rows']:,} rows)"
        for m, e in r["daily"].items()) + "\n")

    add("## 2. Horizon-dependent diagnostics (mask: stage3_in_window)\n")
    add("| h | " + " | ".join(
        f"{m} WQL" for m in r["horizon"]) + " | n_forecasts | n_origins "
        "| n_folds |")
    add("|---|" + "---|" * (len(r["horizon"]) + 3))
    models = list(r["horizon"])
    for h in map(str, range(1, 29)):
        cells = [f"{r['horizon'][m]['by_h'][h]['wql']:.5f}"
                 for m in models]
        ref = r["horizon"][models[0]]["by_h"][h]
        add(f"| {h} | " + " | ".join(cells)
            + f" | {ref['n_forecasts']:,} | {ref['n_origins']} "
              f"| {ref['n_folds']} |")
    add("\nOrigin sets differ by h (h=28: one origin per fold), so "
        "h-curves are not pure forecast-distance effects.\n")

    add("## 3. Delayed interval adaptation (feedback delay 28 days, "
        "intervals as issued)\n")
    for m, e in r["adaptation"].items():
        add(f"### {m}\n")
        add("| mask | n_scored | coverage before | coverage after | "
            "width before | width after |")
        add("|---|---|---|---|---|---|")
        for mid, c in e["coverage"].items():
            add(f"| {mid} | {c['n_scored']:,} | "
                f"{c['coverage_before']:.4f} | {c['coverage_after']:.4f}"
                f" | {c['mean_width_before']:.3f} | "
                f"{c['mean_width_after']:.3f} |")
        t = e["tracker"]
        add(f"\nm_final mean {t['m_final_mean']:.4f} "
            f"[{t['m_final_min']:.3f}, {t['m_final_max']:.3f}]; "
            f"origins released {t['origins_released']}, "
            f"unreleased {t['origins_unreleased']}. Method: "
            f"{e['method']}\n")

    add("## 4. Lead-time cumulative demand (mask: lt3_eval -- separate "
        "target; never ranked against the daily tables)\n")
    add("| method | coverage90 | mean width | WQL |")
    add("|---|---|---|---|")
    for m, e in r["lead_time"]["methods"].items():
        p = e["pooled"]
        add(f"| {m} | {p['coverage90']:.4f} | {p['mean_width90']:.3f} "
            f"| {p['wql']:.5f} |")
    add("\nThe direct cumulative model has the lowest cumulative WQL "
        "of the three methods. This does not establish service-level "
        "or ordering-cost improvement, nor identify the temporal "
        "dependence structure (estimation error and coupling "
        "assumptions are confounded).\n")

    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(L))


if __name__ == "__main__":
    main()
