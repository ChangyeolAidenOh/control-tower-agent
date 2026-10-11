"""Build the forecast display extract for the dashboard (9-D).

Contract: stage9_dashboard_contract.md section 5.1 (D3, ST9-02/ST9-07).
Mechanical, text/performance-blind selection, fixed before viewing any
curves: folds {1, 7}; per category (sorted cat_id) the lexicographically
first bundle_id; within the bundle the series at index (n-1)//2 of
mean_daily_active ascending (tie-break: id); origin = min origin_idx in
that fold's lgbm parquet. Actuals come from staging.panel at extract
time only (core_pipeline.data.db connect(load_config())); the dashboard
runtime reads the JSON alone.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dashboard" / "assets"

FOLDS = [1, 7]
MODELS = ["lgbm", "chronos2s"]
HISTORY_DAYS = 28
LAST_OBSERVED_IDX = 1941

SELECTION_RULE = (
    "Folds {1, 7}. Per category (cat_id ascending) the lexicographically "
    "first bundle_id; within the bundle the series at index (n-1)//2 of "
    "mean_daily_active ascending (tie-break: id); origin = min origin_idx "
    "in that fold's lgbm prediction parquet. Fixed before viewing any "
    "curves (ST9-07); realized values only for target_idx <= 1941."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_series() -> list[dict]:
    subset = pd.read_csv(ROOT / "data" / "processed" / "sku_subset_v2.csv")
    subset["series_id"] = subset["item_id"] + "__" + subset["store_id"]
    picks = []
    for cat in sorted(subset["cat_id"].unique()):
        cat_rows = subset[subset["cat_id"] == cat]
        bundle = sorted(cat_rows["bundle_id"].unique())[0]
        b_rows = cat_rows[cat_rows["bundle_id"] == bundle].sort_values(
            ["mean_daily_active", "id"], ascending=[True, True]
        ).reset_index(drop=True)
        row = b_rows.iloc[(len(b_rows) - 1) // 2]
        picks.append(
            {
                "cat_id": cat,
                "bundle_id": bundle,
                "series_id": row["series_id"],
                "mean_daily_active": float(row["mean_daily_active"]),
            }
        )
    return picks


def load_panel(conn, series_ids: list[str]) -> pd.DataFrame:
    query = (
        "SELECT series_id, period_idx, sales FROM staging.panel "
        "WHERE dataset = 'm5' AND series_id = ANY(%s)"
    )
    with conn.cursor() as cur:
        cur.execute(query, (series_ids,))
        rows = cur.fetchall()
        if not rows:
            cur.execute(
                "SELECT DISTINCT series_id FROM staging.panel "
                "WHERE dataset = 'm5' LIMIT 5"
            )
            print("no panel rows for selected series; requested:", series_ids)
            print("panel series_id sample:", [r[0] for r in cur.fetchall()])
            sys.exit(1)
    return pd.DataFrame(rows, columns=["series_id", "period_idx", "sales"])


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from core_pipeline.data.db import connect, load_config
    from core_pipeline.forecast import prediction_store

    picks = select_series()
    series_ids = [p["series_id"] for p in picks]
    with connect(load_config()) as conn:
        panel = load_panel(conn, series_ids)

    sources = {}
    entries = []
    for fold in FOLDS:
        preds = {}
        for model in MODELS:
            df = prediction_store.load_fold(model, fold)
            path = ROOT / "outputs" / "predictions" / f"{model}_f{fold}.parquet"
            if path.exists():
                sources[str(path.relative_to(ROOT))] = sha256_file(path)
            preds[model] = df
        origin = int(preds["lgbm"]["origin_idx"].min())
        for pick in picks:
            sid = pick["series_id"]
            lgbm = preds["lgbm"]
            block = lgbm[(lgbm["series_id"] == sid) & (lgbm["origin_idx"] == origin)]
            if block.empty:
                print(f"no lgbm rows: fold {fold}, origin {origin}, {sid}")
                print("fold series sample:", lgbm["series_id"].unique()[:5].tolist())
                return 1
            block = block.sort_values("horizon")
            target_idx = block["target_idx"].astype(int).tolist()
            ch = preds["chronos2s"]
            ch_block = ch[(ch["series_id"] == sid) & (ch["origin_idx"] == origin)]
            ch_block = ch_block.sort_values("horizon")
            sp = panel[panel["series_id"] == sid].set_index("period_idx")["sales"]
            hist_idx = list(range(origin - HISTORY_DAYS + 1, origin + 1))
            realized_idx = [t for t in target_idx if t <= LAST_OBSERVED_IDX]
            entries.append(
                {
                    "fold": fold,
                    "origin_idx": origin,
                    "series_id": sid,
                    "bundle_id": pick["bundle_id"],
                    "cat_id": pick["cat_id"],
                    "history": {
                        "t": hist_idx,
                        "y": [float(sp.get(t, float("nan"))) for t in hist_idx],
                    },
                    "realized": {
                        "t": realized_idx,
                        "y": [float(sp.get(t, float("nan"))) for t in realized_idx],
                    },
                    "lgbm": {
                        "t": target_idx,
                        "q05": block["q0.05"].round(4).tolist(),
                        "q50": block["q0.50"].round(4).tolist(),
                        "q95": block["q0.95"].round(4).tolist(),
                    },
                    "chronos2s": {
                        "t": ch_block["target_idx"].astype(int).tolist(),
                        "q50": ch_block["q0.50"].round(4).tolist(),
                    },
                }
            )

    extract = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selection_rule": SELECTION_RULE,
            "db_handle": "core_pipeline.data.db.connect(load_config())",
            "last_observed_idx": LAST_OBSERVED_IDX,
            "band_note": (
                "band = lgbm q0.05-q0.95 (90%); chronos2s shown as median "
                "line only (different quantile sets are never drawn as one band)"
            ),
            "sources": sources,
        },
        "series": entries,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "forecast_extract.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(extract, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"extract {out.relative_to(ROOT)} sha256 {sha256_file(out)}")
    print(f"series entries {len(entries)} (expected {len(FOLDS) * 3})")
    for e in entries:
        print(f"  fold {e['fold']} origin {e['origin_idx']} {e['cat_id']} {e['series_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
