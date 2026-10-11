"""Snapshot loader with integrity gate for the Stage 9 dashboard.

Runtime contract (stage9_dashboard_contract.md section 1): read
dashboard/assets only, verify the deployed snapshot against the sha256
recorded in its manifest, and expose nothing on mismatch.
"""

import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


class SnapshotIntegrityError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    with open(ASSETS / "display_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshot() -> tuple[dict, dict]:
    """Return (snapshot, manifest) after the integrity gate."""
    manifest = load_manifest()
    snap_path = ASSETS / "display_snapshot.json"
    actual = _sha256(snap_path)
    expected = manifest.get("snapshot_sha256", "")
    if actual != expected:
        raise SnapshotIntegrityError(
            f"snapshot sha256 mismatch: manifest {expected[:16]}..., "
            f"file {actual[:16]}..."
        )
    with open(snap_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    return snapshot, manifest
