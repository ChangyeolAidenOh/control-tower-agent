"""ST7-06: freeze the LLM adapter configuration before any paid call.

Writes outputs/agent_llm_freeze.json with (a) the canonical sha of
freeze_payload() -- model id, sampling policy, schema, prompt templates,
budget rule --, (b) the sha256 of the adapter source file itself, (c) the
installed anthropic SDK version and a presence check of every exception
class the error taxonomy depends on, and (d) the frozen smoke episode set.
The freeze is complete when this manifest is committed; ST7-06 records the
commit hash. Any later change to a frozen element requires an explicit
re-freeze entry in the decision log (ST7-04).

Run:
  python -m scripts.freeze_llm_adapter
"""

import datetime
import hashlib
import json
from pathlib import Path

from core_pipeline.agent import llm_adapter
from core_pipeline.agent.search import canonical_sha

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "core_pipeline" / "agent" / "llm_adapter.py"
OUT_PATH = PROJECT_ROOT / "outputs" / "agent_llm_freeze.json"

EXCEPTION_CLASSES = (
    "APITimeoutError", "RateLimitError", "AuthenticationError",
    "PermissionDeniedError", "APIStatusError", "APIConnectionError",
    "InternalServerError",
)

SMOKE_EPISODES = {
    "fold": 1,
    "bundle_ranks": [0, 1],
    "weeks": [0, 1, 2, 3],
    "n_episodes": 8,
    "n_calls_max": 16,
}


def main() -> None:
    import anthropic

    missing = [n for n in EXCEPTION_CLASSES if not hasattr(anthropic, n)]
    if missing:
        raise SystemExit(
            f"anthropic {anthropic.__version__}: missing exception classes "
            f"{missing}; adjust classify_exception before freezing")

    payload = llm_adapter.freeze_payload()
    manifest = {
        "task": "ST7-06 LLM adapter freeze",
        "date": datetime.date.today().isoformat(),
        "freeze_payload_sha256": canonical_sha(payload),
        "freeze_payload": payload,
        "adapter_file": str(ADAPTER_PATH.relative_to(PROJECT_ROOT)),
        "adapter_file_sha256": hashlib.sha256(
            ADAPTER_PATH.read_bytes()).hexdigest(),
        "schema_sha256": llm_adapter.schema_sha(),
        "anthropic_sdk_version": anthropic.__version__,
        "exception_classes_checked": list(EXCEPTION_CLASSES),
        "smoke_episodes": SMOKE_EPISODES,
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"frozen payload_sha={manifest['freeze_payload_sha256'][:12]} "
          f"adapter_sha={manifest['adapter_file_sha256'][:12]} "
          f"sdk={anthropic.__version__}")
    print(f"wrote {OUT_PATH.name}")


if __name__ == "__main__":
    main()