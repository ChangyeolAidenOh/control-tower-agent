"""ST7-06 (freeze pending): LLM candidate-generation adapter.

Frozen elements (hash-recorded by scripts/freeze_llm_adapter.py before any
paid call): model id, prompt templates, structured-output schema, sampling
policy (temperature/top_p/top_k OMITTED -- API defaults; raw responses are
stored with sha256 for post-hoc reproduction), max_tokens, timeout,
max_retries=0 (the official SDK retries transient errors twice by default,
which would violate the no-retry contract of ST7-01/preregistration s8, so
it is disabled explicitly), and the conservative budget rule.

Structured outputs: output_config.format json_schema (GA). The API requires
additionalProperties: false on every object, so sku_multipliers cannot be a
dynamic-key object; the wire format is an ARRAY of {series_id, multiplier}
pairs, converted here to the D1 dict envelope and validated through the
SAME parse_candidates path as any other source. Duplicate series_id entries
inside one candidate's array are detected BEFORE the dict collapse and
invalidate that candidate only (duplicate_sku_entry).

Error taxonomy (ST7-06, reviewer-fixed): technical API failures (timeout,
429, 5xx, connection) are recorded as tech errors with ZERO additional
calls; auth/permission failures and budget exhaustion abort the run;
stop_reason max_tokens or refusal fails that call (no continuation call);
none of these are counted as invalid candidates. Information set (L2):
the prompt carries only close-of-t_w facts -- never the selection
scenarios, other bundles/folds, or baseline performance numbers.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from core_pipeline.agent.candidates import (
    MAX_CANDIDATES,
    MAX_LISTED_SKUS,
    MULT_GRID,
    TAU_GRID,
    ParsedCandidate,
    ParseResult,
    parse_candidates,
)
from core_pipeline.agent.search import canonical_sha

MODEL = "claude-haiku-5-5"
GEN_MAX_TOKENS = 8192
EXPLAIN_MAX_TOKENS = 1024
TIMEOUT_S = 60.0
SDK_MAX_RETRIES = 0
TOTAL_BUDGET_USD = 10.0
PRICE_IN_PER_MTOK = 0.10
PRICE_OUT_PER_MTOK = 0.50
RECENT_DAYS = 28

OUTCOME_OK = "ok"
OUTCOME_PARSE_FAILED = "parse_failed"
OUTCOME_MAX_TOKENS = "max_tokens_truncated"
OUTCOME_REFUSAL = "model_refusal"
OUTCOME_TIMEOUT = "tech_timeout"
OUTCOME_CONNECTION = "tech_connection"
OUTCOME_RATE_LIMITED = "tech_rate_limited"
OUTCOME_SERVER_ERROR = "tech_server_error"
OUTCOME_FATAL_AUTH = "fatal_auth"
OUTCOME_FATAL_BUDGET = "fatal_budget"
TECH_OUTCOMES = (OUTCOME_TIMEOUT, OUTCOME_CONNECTION,
                 OUTCOME_RATE_LIMITED, OUTCOME_SERVER_ERROR)
FATAL_OUTCOMES = (OUTCOME_FATAL_AUTH, OUTCOME_FATAL_BUDGET)

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "tau_prime": {"type": ["number", "null"]},
                    "sku_multipliers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "series_id": {"type": "string"},
                                "multiplier": {"type": "number"},
                            },
                            "required": ["series_id", "multiplier"],
                            "additionalProperties": False,
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_id", "tau_prime",
                             "sku_multipliers", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def schema_sha() -> str:
    return canonical_sha(CANDIDATE_SCHEMA)


GEN_SYSTEM = (
    "You are the candidate-generation module of an inventory control tower. "
    "You propose weekly order-policy adjustments for ONE product bundle under "
    "a shared ordering-capacity constraint. You only see information observed "
    "up to the decision point; you never see future demand. Your candidates "
    "are screened by a deterministic simulator gate (capacity and service "
    "level) and ranked by simulated cost; infeasible or off-grid candidates "
    "are discarded, so strict format compliance matters."
)

GEN_RULES = (
    "Action contract:\n"
    "- tau_prime: base-stock quantile for this week, one of {taus}, or null "
    "to keep the base value {base_tau}.\n"
    "- sku_multipliers: per-SKU order multipliers for this week, each value "
    "one of {mults}; at most {max_skus} SKUs listed; unlisted SKUs stay at "
    "1.0; 0.0 means skip ordering that SKU this week.\n"
    "- Propose up to {max_cands} candidates; only the first {max_cands} are "
    "kept. Diversity across candidates is useful because a gate rejects "
    "candidates whose simulated bundle-week service level (median over "
    "evaluation scenarios) falls below {floor} or whose orders exceed the "
    "shared capacity.\n"
    "- Costs: holding h_i per unit-day, shortage b_i per unmet unit. "
    "Ordering less saves holding but risks shortage; the base policy has "
    "tended to under-order for high-b SKUs when capacity binds."
)


def _fmt_series_table(series_ids, h, b, recent, inv, pipeline_sum) -> str:
    lines = ["series_id | h | b | inv | in_transit | last_28d_demand"]
    for i, sid in enumerate(series_ids):
        hist = ",".join(str(int(x)) for x in recent[:, i])
        lines.append(
            f"{sid} | {h[i]:.4f} | {b[i]:.4f} | {inv[i]:.1f} | "
            f"{pipeline_sum[i]:.1f} | {hist}")
    return "\n".join(lines)


def build_generation_prompt(
    group_id: str,
    series_ids,
    h: np.ndarray,
    b: np.ndarray,
    recent_demand: np.ndarray,       # (RECENT_DAYS, n_sku), up to and incl. t_w
    inv: np.ndarray,
    pipeline: np.ndarray,            # (n_sku, L)
    k_g: float,
    base_tau: float,
    week: int,                       # 1-based
    prev_week: dict | None,          # None for week 1
) -> str:
    """Deterministic user prompt. Information set = close of t_w only (L2):
    no selection scenarios, no other bundles/folds, no baseline numbers."""
    n_sku = len(series_ids)
    if recent_demand.shape != (RECENT_DAYS, n_sku):
        raise ValueError(
            f"recent_demand shape {recent_demand.shape} != "
            f"({RECENT_DAYS}, {n_sku})")
    if (week == 1) != (prev_week is None):
        raise ValueError("prev_week must be None exactly for week 1")
    rules = GEN_RULES.format(
        taus=list(TAU_GRID), base_tau=f"{base_tau:.2f}",
        mults=list(MULT_GRID), max_skus=MAX_LISTED_SKUS,
        max_cands=MAX_CANDIDATES, floor="0.90")
    parts = [
        rules,
        f"Bundle {group_id}: {n_sku} SKUs, shared ordering capacity "
        f"K_g = {k_g:.4f} units/day (sum of all SKU orders per day).",
        f"Planning week {week} of 4. State at the decision point "
        "(close of the day before the first action day):",
        _fmt_series_table(series_ids, h, b, recent_demand, inv,
                          pipeline.sum(axis=1)),
    ]
    if prev_week is not None:
        parts.append(
            "Previous week realized summary: "
            f"{int(prev_week['stockout_sku_days'])} SKU-days with unmet "
            f"demand, {int(prev_week['binding_days'])} capacity-binding "
            f"days, action taken: tau={prev_week['selected_tau']:.2f}, "
            f"{int(prev_week['n_modified_skus'])} SKU multipliers != 1.0.")
    parts.append(
        "Return the candidates JSON now. Use short candidate_id values "
        "(c01..c20) and one-sentence rationales.")
    return "\n\n".join(parts)


def build_explanation_prompt(group_id: str, week: int, selected: dict,
                             gate: dict) -> str:
    """Post-selection narration input (L6): selected action + gate stats
    only -- no future demand, no realized evaluation cost."""
    return (
        "Write a 3-5 sentence operator-facing explanation of this week's "
        f"approved inventory action for bundle {group_id}, week {week}. "
        f"Action: tau={selected['tau_eff']}, multipliers="
        f"{selected['multipliers']}. Candidate gate statistics: "
        f"{gate['evaluated_candidates']} evaluated, "
        f"{gate['capacity_rejected']} capacity-rejected, "
        f"{gate['csl_rejected']} service-level-rejected. State what the "
        "action changes versus the base policy and why it passed the gate. "
        "Do not speculate about future demand or final costs.")


def convert_structured_envelope(obj) -> tuple[dict, dict[int, str]]:
    """Wire format (sku_multipliers array of pairs) -> D1 dict envelope.
    Returns (envelope, slot_flags); slot_flags[i] names a transport
    violation for array slot i that the dict collapse would hide."""
    flags: dict[int, str] = {}
    out_cands = []
    cands = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(cands, list):
        raise ValueError("missing_candidates_array")
    for i, cand in enumerate(cands):
        if not isinstance(cand, dict):
            out_cands.append(cand)
            continue
        new_cand = dict(cand)
        pairs = cand.get("sku_multipliers")
        if isinstance(pairs, list):
            mult: dict[str, float] = {}
            for p in pairs:
                if (not isinstance(p, dict) or "series_id" not in p
                        or "multiplier" not in p):
                    flags[i] = "bad_multiplier_entry"
                    break
                sid = p["series_id"]
                if sid in mult:
                    flags[i] = "duplicate_sku_entry"
                    break
                mult[str(sid)] = p["multiplier"]
            new_cand["sku_multipliers"] = mult
        out_cands.append(new_cand)
    return {"candidates": out_cands}, flags


def process_generation_response(text: str, series_ids,
                                base_tau: float) -> ParseResult:
    """Full validation path for one generation response: json -> wire
    conversion -> D1 parse_candidates -> transport-flag invalidation."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return ParseResult(parse_failed=True,
                           failure_reason=f"json_error:{exc}")
    try:
        envelope, flags = convert_structured_envelope(obj)
    except ValueError as exc:
        return ParseResult(parse_failed=True, failure_reason=str(exc))
    extras = tuple(sorted(set(obj) - {"candidates"}))
    result = parse_candidates(json.dumps(envelope), series_ids, base_tau)
    if result.parse_failed:
        return result
    fixed = []
    for c in result.candidates:
        reason = flags.get(c.index)
        if reason is not None and c.is_valid:
            fixed.append(ParsedCandidate(
                index=c.index, candidate_id=c.candidate_id, action=None,
                invalid_reason=reason))
        else:
            fixed.append(c)
    return ParseResult(parse_failed=False, failure_reason=None,
                       candidates=fixed,
                       truncated_count=result.truncated_count,
                       envelope_extra_keys=extras)


@dataclass
class CostLedger:
    """Conservative budget guard (ST7-06): a call is allowed only if the
    already-spent cost PLUS the worst case of the next call (estimated
    input + full max_tokens output) stays within the budget."""
    budget_usd: float = TOTAL_BUDGET_USD
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @staticmethod
    def call_cost(input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * PRICE_IN_PER_MTOK
                + output_tokens * PRICE_OUT_PER_MTOK) / 1e6

    @staticmethod
    def estimate_input_tokens(prompt_chars: int) -> int:
        return prompt_chars // 3 + 256

    def can_afford(self, prompt_chars: int, max_tokens: int) -> bool:
        worst = self.call_cost(self.estimate_input_tokens(prompt_chars),
                               max_tokens)
        return self.spent_usd + worst <= self.budget_usd

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.spent_usd += self.call_cost(int(input_tokens),
                                         int(output_tokens))


@dataclass
class CallRecord:
    outcome: str
    text: str = ""
    stop_reason: str | None = None
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    response_sha256: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


def make_client():
    from anthropic import Anthropic
    return Anthropic(max_retries=SDK_MAX_RETRIES, timeout=TIMEOUT_S)


def classify_exception(exc) -> str:
    import anthropic
    if isinstance(exc, anthropic.APITimeoutError):
        return OUTCOME_TIMEOUT
    if isinstance(exc, anthropic.RateLimitError):
        return OUTCOME_RATE_LIMITED
    if isinstance(exc, (anthropic.AuthenticationError,
                        anthropic.PermissionDeniedError)):
        return OUTCOME_FATAL_AUTH
    if isinstance(exc, anthropic.APIStatusError):
        return (OUTCOME_SERVER_ERROR if exc.status_code >= 500
                else OUTCOME_FATAL_AUTH)
    if isinstance(exc, anthropic.APIConnectionError):
        return OUTCOME_CONNECTION
    raise exc


def _finish(response, t0: float) -> CallRecord:
    text = "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text")
    raw = json.loads(response.model_dump_json())
    rec = CallRecord(
        outcome=OUTCOME_OK,
        text=text,
        stop_reason=response.stop_reason,
        request_id=getattr(response, "_request_id", None) or response.id,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=time.perf_counter() - t0,
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        raw=raw,
    )
    if response.stop_reason == "max_tokens":
        rec.outcome = OUTCOME_MAX_TOKENS
    elif response.stop_reason == "refusal":
        rec.outcome = OUTCOME_REFUSAL
    return rec


def call_generation(client, ledger: CostLedger, system: str,
                    prompt: str) -> CallRecord:
    """One candidate-generation call. No retries of any kind (ST7-01); a
    technical failure is recorded and the episode falls back to no-action."""
    if not ledger.can_afford(len(system) + len(prompt), GEN_MAX_TOKENS):
        return CallRecord(outcome=OUTCOME_FATAL_BUDGET,
                          error="budget guard blocked the call")
    t0 = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=GEN_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": CANDIDATE_SCHEMA}},
        )
    except Exception as exc:
        outcome = classify_exception(exc)
        return CallRecord(outcome=outcome, error=repr(exc),
                          latency_s=time.perf_counter() - t0)
    rec = _finish(response, t0)
    ledger.record(rec.input_tokens, rec.output_tokens)
    return rec


def call_explanation(client, ledger: CostLedger, prompt: str) -> CallRecord:
    """One post-selection explanation call (L6): pure record, failure ->
    empty string, never retried, measured OUTSIDE the H5 selection time."""
    if not ledger.can_afford(len(prompt), EXPLAIN_MAX_TOKENS):
        return CallRecord(outcome=OUTCOME_FATAL_BUDGET,
                          error="budget guard blocked the call")
    t0 = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=EXPLAIN_MAX_TOKENS,
            system="You write concise operational summaries.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        outcome = classify_exception(exc)
        return CallRecord(outcome=outcome, error=repr(exc),
                          latency_s=time.perf_counter() - t0)
    rec = _finish(response, t0)
    ledger.record(rec.input_tokens, rec.output_tokens)
    return rec


def freeze_payload() -> dict:
    """Everything ST7-06 freezes, in hashable form."""
    return {
        "model": MODEL,
        "sampling": "temperature/top_p/top_k omitted (API defaults)",
        "gen_max_tokens": GEN_MAX_TOKENS,
        "explain_max_tokens": EXPLAIN_MAX_TOKENS,
        "timeout_s": TIMEOUT_S,
        "sdk_max_retries": SDK_MAX_RETRIES,
        "budget_usd": TOTAL_BUDGET_USD,
        "price_in_per_mtok": PRICE_IN_PER_MTOK,
        "price_out_per_mtok": PRICE_OUT_PER_MTOK,
        "schema": CANDIDATE_SCHEMA,
        "gen_system": GEN_SYSTEM,
        "gen_rules_template": GEN_RULES,
        "recent_days": RECENT_DAYS,
        "wire_format": "sku_multipliers as array of {series_id, multiplier}",
    }