"""Read-only access to the Stage 2 mart interface (mart_readme.md section 7).

Stage 2 code reads only the objects listed there; this module centralizes the
queries so scripts stay thin. All fetches return numpy-ready structures with a
deterministic SKU order (sorted series_id within each bundle).
"""

import numpy as np


def fetch_window(conn, name: str) -> tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT start_idx, end_idx FROM mart.windows WHERE window_name = %s", (name,)
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"window not found: {name}")
    return int(row[0]), int(row[1])


def fetch_bundle_capacity(conn, kappa: float) -> dict[str, float]:
    column = "k_g_kappa_" + f"{kappa:.1f}".replace(".", "_")
    cur = conn.cursor()
    cur.execute(f"SELECT group_id, {column} FROM mart.bundles ORDER BY group_id")
    return {gid: float(kg) for gid, kg in cur.fetchall()}


def fetch_demand_by_bundle(conn, max_idx: int) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Latent-demand panels per bundle: {group_id: (series_ids, matrix)} where
    matrix has shape (max_idx, n_sku) and row j holds day period_idx = j + 1."""
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM mart.bundles ORDER BY group_id")
    group_ids = [r[0] for r in cur.fetchall()]
    out = {}
    for gid in group_ids:
        cur.execute(
            """
            SELECT series_id, period_idx, sales FROM mart.sku_daily
            WHERE group_id = %s AND period_idx <= %s
            ORDER BY series_id, period_idx
            """,
            (gid, max_idx),
        )
        rows = cur.fetchall()
        series = tuple(sorted({r[0] for r in rows}))
        idx = {s: j for j, s in enumerate(series)}
        mat = np.zeros((max_idx, len(series)))
        for sid, p, sales in rows:
            mat[p - 1, idx[sid]] = sales
        out[gid] = (series, mat)
    return out


def fetch_sku_costs(conn) -> dict[str, dict[str, float]]:
    """{series_id: {"h": ..., "b_r3": ..., "b_r5": ..., "b_r10": ...}}"""
    cur = conn.cursor()
    cur.execute("SELECT series_id, h, b_r3, b_r5, b_r10 FROM mart.sku_cost")
    return {
        sid: {"h": float(h), "b_r3": float(b3), "b_r5": float(b5), "b_r10": float(b10)}
        for sid, h, b3, b5, b10 in cur.fetchall()
    }


def cost_arrays(series: tuple[str, ...], costs: dict[str, dict[str, float]],
                r: int) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (h, b) arrays for a bundle's series order at cost ratio r."""
    h = np.array([costs[s]["h"] for s in series])
    b = np.array([costs[s][f"b_r{r}"] for s in series])
    return h, b


def fetch_folds(conn) -> list[dict]:
    """All rolling-origin folds, sorted by fold number. Row indices are
    1-based period_idx values (mart.folds, build_mart.py)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fold, test_start_idx, test_end_idx, train_end_idx,
               warmup_start_idx, warmup_days, is_retrain
        FROM mart.folds ORDER BY fold
        """
    )
    cols = ("fold", "test_start_idx", "test_end_idx", "train_end_idx",
            "warmup_start_idx", "warmup_days", "is_retrain")
    return [dict(zip(cols, row)) for row in cur.fetchall()]
