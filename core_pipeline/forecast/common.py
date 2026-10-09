"""Lightweight shared constants and IO for both model stacks (ST3-05).

This module must stay free of lightgbm and torch imports: loading both
native runtimes in one process duplicates the OpenMP runtime (libomp)
on macOS and segfaults. Each runner therefore imports only its own
model library, and everything shared lives here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

QUANTILES: tuple[float, ...] = (
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
