"""ST7-02 D5: generate and freeze the numeric-search sample schedule.

Writes outputs/agent_search_schedule.json (schedule + canonical sha256
embedded alongside it in a small wrapper). --verify regenerates from the
DB bundle listing and asserts bitwise equality with the stored schedule,
so the runner can trust the file after a sha check alone.

Run:
  python -m scripts.build_agent_search_schedule
  python -m scripts.build_agent_search_schedule --verify
"""

import argparse
import json
from pathlib import Path

from core_pipeline.agent.search import build_schedule, canonical_sha
from core_pipeline.data import db
from core_pipeline.data.mart_reader import fetch_demand_by_bundle, fetch_folds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "outputs" / "agent_search_schedule.json"


def load_bundle_series() -> tuple[dict, list[int]]:
    db.load_dotenv()
    conn = db.connect(db.load_config())
    try:
        folds = [f["fold"] for f in fetch_folds(conn)]
        max_idx = max(f["test_end_idx"] for f in fetch_folds(conn))
        demand = fetch_demand_by_bundle(conn, max_idx)
    finally:
        conn.close()
    bundle_series = {gid: tuple(demand[gid][0]) for gid in demand}
    return bundle_series, sorted(folds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    bundle_series, folds = load_bundle_series()
    schedule = build_schedule(bundle_series, folds)
    sha = canonical_sha(schedule)
    n_eps = len(schedule["episodes"])

    if args.verify:
        stored = json.loads(OUT_PATH.read_text())
        if stored["schedule_sha256"] != sha:
            raise SystemExit(
                f"schedule sha mismatch: stored {stored['schedule_sha256'][:12]} "
                f"!= regenerated {sha[:12]}")
        if stored["schedule"] != schedule:
            raise SystemExit("schedule content mismatch despite sha match path")
        print(f"verify ok episodes={n_eps} sha={sha[:12]}")
        return

    OUT_PATH.write_text(json.dumps(
        {"schedule_sha256": sha, "schedule": schedule},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    print(f"wrote {OUT_PATH.name} episodes={n_eps} sha={sha[:12]}")


if __name__ == "__main__":
    main()