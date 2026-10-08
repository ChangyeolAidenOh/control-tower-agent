import hashlib
import os
from pathlib import Path

import psycopg
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path=PROJECT_ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path=PROJECT_ROOT / "configs" / "data.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def config_sha256(path=PROJECT_ROOT / "configs" / "data.yaml"):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect(cfg):
    load_dotenv()
    db = cfg["db"]
    return psycopg.connect(
        host=os.environ[db["host_env"]],
        port=int(os.environ[db["port_env"]]),
        dbname=os.environ[db["dbname_env"]],
        user=os.environ[db["user_env"]],
        password=os.environ[db["password_env"]],
    )


def run_sql_file(cur, path):
    cur.execute(Path(path).read_text())
