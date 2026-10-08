import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from core_pipeline.data import m5_adapter, vn2_adapter
from core_pipeline.data.db import PROJECT_ROOT, config_sha256, connect, load_config, run_sql_file

ADAPTERS = {"m5": m5_adapter, "vn2": vn2_adapter}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(duck, path):
    return duck.execute(
        "SELECT count(*) FROM read_csv_auto(?, header=true, all_varchar=true)", [str(path)]
    ).fetchone()[0]


def copy_file(cur, table, columns, path, header):
    cols = ", ".join(columns)
    opts = "FORMAT csv, HEADER true" if header else "FORMAT csv"
    with open(path, "rb") as f, cur.copy(f"COPY {table} ({cols}) FROM STDIN ({opts})") as copy:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            copy.write(chunk)


def copy_query(cur, table, columns, duck, query):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    duck.execute(f"COPY ({query}) TO '{tmp_path}' (HEADER false)")
    copy_file(cur, table, columns, tmp_path, header=False)
    Path(tmp_path).unlink()


def record_manifest(cur, dataset, path, source_rows, loaded_rows, row_filter, loader_version):
    cur.execute(
        """
        INSERT INTO raw.file_manifest
            (dataset, file_name, file_path, sha256, byte_size, source_rows, loaded_rows, row_filter, loader_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (dataset, path.name, str(path), sha256_file(path), path.stat().st_size,
         source_rows, loaded_rows, row_filter, loader_version),
    )


def load_dataset(dataset, cfg=None):
    cfg = cfg or load_config()
    adapter = ADAPTERS[dataset]
    ds_cfg = cfg["datasets"][dataset]
    loader_version = cfg["loader_version"]
    started = datetime.now(timezone.utc)
    duck = duckdb.connect()
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM raw.file_manifest WHERE dataset = %s", (dataset,))
        adapter.load_raw(cur, duck, ds_cfg, loader_version, PROJECT_ROOT, record_manifest, copy_file, copy_query)
        run_sql_file(cur, PROJECT_ROOT / "sql" / "staging" / f"{dataset}.sql")
        cur.execute(
            """
            SELECT count(DISTINCT series_id), min(period_idx), max(period_idx), count(*)
            FROM staging.panel WHERE dataset = %s
            """,
            (dataset,),
        )
        series_count, period_min, period_max, panel_rows = cur.fetchone()
        cur.execute(
            """
            INSERT INTO staging.load_runs
                (dataset, loader_version, config_sha256, series_count, period_min, period_max, panel_rows,
                 started_at, finished_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (dataset, loader_version, config_sha256(), series_count, period_min, period_max, panel_rows,
             started, datetime.now(timezone.utc)),
        )
        conn.commit()
    duck.close()
    print(f"{dataset}: series={series_count} periods={period_min}..{period_max} panel_rows={panel_rows}")
    return series_count, period_min, period_max, panel_rows
