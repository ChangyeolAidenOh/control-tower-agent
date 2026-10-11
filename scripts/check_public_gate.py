"""Public-release gate checks for the Stage 9 dashboard repo (9-G).

Contract: stage9_dashboard_contract.md section 3. Four checks:
1. Tracked files: no env/credential files, no untracked-only raw dirs
   leaking in, no secret patterns in the tracked agent JSON or assets.
   request_id fields are reported as informational (trace identifiers,
   not credentials; removing them would break the frozen anchor sha).
2. Asset size: dashboard/assets stays within a few MB.
3. Dashboard requirements: no training/optimization libraries.
4. Snapshot whitelist review flags from the manifest.
Exit code 0 only if every hard check passes.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"), "anthropic api key"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "api key"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{15,}"), "bearer token"),
    (re.compile(r"(?i)(api[_-]?key|secret|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9]"), "credential assignment"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "email address"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws key"),
]

FORBIDDEN_TRACKED = [".env", "id_rsa", ".pem", ".key"]
FORBIDDEN_DIRS = ["outputs/agent_llm_raw/"]
HEAVY_LIBS = ["torch", "catboost", "cvxpy", "lightgbm", "psycopg", "chronos", "tensorflow", "xgboost"]

ASSET_LIMIT_MB = 10.0


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def scan_text(label: str, text: str, findings: list) -> None:
    for pattern, name in SECRET_PATTERNS:
        m = pattern.search(text)
        if m:
            findings.append(f"{label}: {name} pattern matched ({m.group()[:20]}...)")


def main() -> int:
    findings = []
    infos = []
    files = tracked_files()

    for f in files:
        low = f.lower()
        if any(low.endswith(t) or f"/{t}" in low for t in FORBIDDEN_TRACKED):
            findings.append(f"forbidden tracked file: {f}")
        if any(f.startswith(d) for d in FORBIDDEN_DIRS):
            findings.append(f"raw response file tracked: {f}")

    agent_path = "outputs/agent_llm_official.json"
    if agent_path in files:
        agent = json.loads((ROOT / agent_path).read_text(encoding="utf-8"))
        n_req = 0
        for w in agent["weeks"]:
            for blk in (w.get("explanation", {}), w.get("generation", {}).get("call", {})):
                if blk.get("request_id"):
                    n_req += 1
            scan_text(f"{agent_path} week f{w['fold']}w{w['week']} {w['group_id']}",
                      json.dumps(w, ensure_ascii=False), findings)
        infos.append(
            f"{agent_path}: {n_req} request_id fields present (trace identifiers, "
            "not credentials; kept to preserve the frozen anchor sha)"
        )

    assets_dir = ROOT / "dashboard" / "assets"
    total = 0
    for p in sorted(assets_dir.glob("*")):
        size = p.stat().st_size
        total += size
        scan_text(f"assets/{p.name}", p.read_text(encoding="utf-8"), findings)
    total_mb = total / (1 << 20)
    if total_mb > ASSET_LIMIT_MB:
        findings.append(f"assets too large: {total_mb:.1f} MB > {ASSET_LIMIT_MB} MB")
    infos.append(f"assets total {total_mb:.2f} MB")

    req = (ROOT / "dashboard" / "requirements.txt").read_text(encoding="utf-8").lower()
    for lib in HEAVY_LIBS:
        if lib in req:
            findings.append(f"heavy library in dashboard requirements: {lib}")

    manifest = json.loads((assets_dir / "display_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pending"):
        findings.append(f"manifest pending not empty: {manifest['pending']}")
    infos.append(f"whitelist {manifest['whitelist_version']}, case review: {manifest['case_texts_review']}")
    infos.append(f"snapshot sha {manifest['snapshot_sha256'][:16]}...")

    print("== informational ==")
    for i in infos:
        print(" ", i)
    if findings:
        print("== FINDINGS (gate blocked) ==")
        for f in findings:
            print(" ", f)
        return 1
    print("== gate PASSED ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
