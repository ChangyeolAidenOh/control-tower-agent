"""Probe the frozen Stage 6 episode artifact for Stage 8 lagging-metric fields.

Read-only. No writes, no manifest. Run before registering ST8 aggregation rules.

Usage:
    python -m scripts.probe_stage8_fields
    python -m scripts.probe_stage8_fields --input outputs/stage6_eval_core_step6.json
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELD_GROUPS = {
    "cost": r"cost|holding|penalty|terminal",
    "service_csl": r"csl|service|fill",
    "stockout": r"stockout|short|unmet|lost|miss",
    "inventory": r"inv|on_hand|onhand|stock(?!out)|doi|days|pipeline",
    "demand_sales": r"demand|sales|units",
    "keys_flags": r"policy|fold|bundle|group|week|^r$|r_env|seed|scenario|guardrail|block|binding|flag",
}

KEY_CANDIDATES = ["policy", "r", "r_env", "fold", "bundle", "group_id", "week", "episode"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_episode_rows(obj):
    """Return (location, rows) for the largest list of dicts in the JSON."""
    best = ("", [])
    stack = [("$", obj)]
    while stack:
        loc, node = stack.pop()
        if isinstance(node, list):
            if node and all(isinstance(x, dict) for x in node[:20]):
                if len(node) > len(best[1]):
                    best = (loc, node)
            elif node and isinstance(node[0], (list, dict)):
                for i, x in enumerate(node[:5]):
                    stack.append((f"{loc}[{i}]", x))
        elif isinstance(node, dict):
            for k, v in node.items():
                stack.append((f"{loc}.{k}", v))
    return best


def summarize_value(v, limit=60):
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def profile_rows(rows):
    n = len(rows)
    presence = Counter()
    types = defaultdict(set)
    num_min = {}
    num_max = {}
    samples = defaultdict(list)
    subkeys = defaultdict(Counter)
    for row in rows:
        for k, v in row.items():
            presence[k] += 1
            types[k].add(type(v).__name__)
            if isinstance(v, bool):
                pass
            elif isinstance(v, (int, float)):
                num_min[k] = v if k not in num_min else min(num_min[k], v)
                num_max[k] = v if k not in num_max else max(num_max[k], v)
            if isinstance(v, dict):
                for sk in v:
                    subkeys[k][sk] += 1
            if len(samples[k]) < 3:
                samples[k].append(summarize_value(v))
    print(f"rows: {n}")
    print(f"distinct keys: {len(presence)}")
    print()
    print("key | present | types | numeric_range | samples")
    for k in sorted(presence):
        rng = ""
        if k in num_min:
            rng = f"[{num_min[k]:.6g}, {num_max[k]:.6g}]"
        print(f"{k} | {presence[k]}/{n} | {','.join(sorted(types[k]))} | {rng} | {' ; '.join(samples[k])}")
    if subkeys:
        print()
        print("nested dict subkeys (level 1):")
        for k in sorted(subkeys):
            items = ", ".join(f"{sk}({c})" for sk, c in subkeys[k].most_common())
            print(f"  {k}: {items}")
    return presence, subkeys


def match_groups(presence, subkeys):
    print()
    print("candidate field groups (case-insensitive regex match):")
    all_names = set(presence)
    for parent, sub in subkeys.items():
        all_names.update(f"{parent}.{sk}" for sk in sub)
    for group, pattern in FIELD_GROUPS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        hits = sorted(name for name in all_names if rx.search(name))
        print(f"  {group}: {hits if hits else 'NO MATCH'}")


def population_check(rows):
    print()
    print("population structure:")
    present_keys = [k for k in KEY_CANDIDATES if any(k in r for r in rows[:50])]
    print(f"  key columns found: {present_keys}")
    for k in present_keys:
        vals = sorted({str(r.get(k)) for r in rows})
        if len(vals) <= 30:
            print(f"  distinct {k} ({len(vals)}): {vals}")
        else:
            print(f"  distinct {k}: {len(vals)} values, first 5 {vals[:5]}")
    pol_key = "policy" if "policy" in present_keys else None
    r_key = "r" if "r" in present_keys else ("r_env" if "r_env" in present_keys else None)
    if pol_key and r_key:
        counts = Counter((r[pol_key], r[r_key]) for r in rows)
        bad = {k: v for k, v in counts.items() if v != 480}
        print(f"  (policy, r) cells: {len(counts)}, rows/cell expected 480")
        if bad:
            print(f"  MISMATCH cells: {bad}")
        else:
            print("  all cells == 480")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/stage6_eval_core_step6.json")
    parser.add_argument("--confirmatory", default="outputs/stage6_confirmatory.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"input not found: {in_path}")
        candidates = sorted(Path("outputs").glob("stage6*"))
        print("outputs/stage6* candidates:")
        for c in candidates:
            print(f"  {c} ({c.stat().st_size} bytes)")
        return

    digest = sha256_of(in_path)
    print(f"input: {in_path}")
    print(f"sha256: {digest}")
    conf_path = Path(args.confirmatory)
    if conf_path.exists():
        conf = json.load(open(conf_path))
        ref_name = conf.get("input_file")
        ref_sha = conf.get("input_sha256")
        if ref_name == in_path.name:
            status = "MATCH" if ref_sha == digest else "MISMATCH"
            print(f"confirmatory input_sha256: {ref_sha} -> {status}")
        else:
            print(f"confirmatory references different file: {ref_name}")
    else:
        print(f"confirmatory not found at {conf_path}, sha cross-check skipped")
    print()

    data = json.load(open(in_path))
    if isinstance(data, dict):
        print("top-level keys:")
        for k, v in data.items():
            desc = f"list len {len(v)}" if isinstance(v, list) else type(v).__name__
            print(f"  {k}: {desc}")
        print()

    loc, rows = find_episode_rows(data)
    if not rows:
        print("no list-of-dict episode rows found; top-level dump above is all we have")
        return
    print(f"episode rows located at: {loc}")
    presence, subkeys = profile_rows(rows)
    match_groups(presence, subkeys)
    population_check(rows)

    print()
    print("companion files outputs/stage6*:")
    for c in sorted(Path("outputs").glob("stage6*")):
        print(f"  {c.name} ({c.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
