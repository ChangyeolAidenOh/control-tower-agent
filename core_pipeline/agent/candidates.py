"""ST7-02 D1: candidate JSON schema, validation, and action normalization.

Envelope contract: a single JSON object {"candidates": [...]}. Envelope-level
malformation (invalid JSON, missing/ill-typed "candidates", duplicate envelope
keys) is an EPISODE parse failure -> zero candidates (ST7-01). Extra envelope
keys are recorded but not fatal. Candidate-level violations invalidate ONLY
that candidate; the remaining candidates survive (ST7-02 D1).

Slot rule: the first MAX_CANDIDATES array elements occupy the budget slots in
array order (no favorable selection beyond 20); the rest are truncated and
counted. Validation happens per slot.

Action normalization (identity excludes candidate_id/rationale):
  tau_eff      = snapped tau_prime, or base_tau when tau_prime is null/absent
  multipliers  = listed (sku, m) pairs with m != 1.0, sorted by sku
  no-action    = tau_eff == base_tau and no multipliers
  n_modified   = number of multipliers with m != 1.0 (tie-break metric)
  tau_delta    = |tau_eff - base_tau| (null normalizes to 0; tie-break metric)

The listed-SKU limit (<= 10) applies to the keys as written in the JSON
(ST7-01 "named SKUs"), while identity/tie-break use the effective m != 1.0
count (ST7-02 D1); both are enforced here.
"""

import json
from dataclasses import dataclass, field

TAU_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
MULT_GRID = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5)
GRID_TOL = 1e-9
MAX_LISTED_SKUS = 10
MAX_CANDIDATES = 20
ALLOWED_KEYS = frozenset({"candidate_id", "tau_prime", "sku_multipliers", "rationale"})
_DUP_MARKER = "__dup_keys__"


def _pairs_hook(pairs):
    """Detect duplicate keys inside any JSON object (json.loads keeps the last
    one silently otherwise); a marker key is injected for the validator."""
    out = {}
    dups = []
    for k, v in pairs:
        if k in out:
            dups.append(k)
        out[k] = v
    if dups:
        out[_DUP_MARKER] = tuple(dups)
    return out


def snap_to_grid(x, grid, tol: float = GRID_TOL) -> float | None:
    """Return the exact grid value within tol of x, or None (off-grid).
    Booleans are rejected (JSON true/false are not grid numbers)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    for g in grid:
        if abs(float(x) - g) <= tol:
            return g
    return None


@dataclass(frozen=True)
class AgentAction:
    """Normalized action; equality over (base_tau, tau_eff, multipliers) is the
    duplicate-candidate criterion (candidate_id/rationale excluded)."""
    base_tau: float
    tau_eff: float
    multipliers: tuple[tuple[str, float], ...]   # sorted, m != 1.0 only

    @property
    def n_modified_skus(self) -> int:
        return len(self.multipliers)

    @property
    def tau_delta(self) -> float:
        return abs(self.tau_eff - self.base_tau)

    @property
    def is_no_action(self) -> bool:
        return self.tau_eff == self.base_tau and not self.multipliers

    def multiplier_map(self) -> dict[str, float]:
        return dict(self.multipliers)


def make_action(tau_prime, multipliers, base_tau: float,
                series_ids) -> AgentAction:
    """Construct a normalized action from already-trusted inputs (numeric
    search arm, tests). Raises ValueError on any contract breach so the
    sampler cannot silently emit an illegal action."""
    base = snap_to_grid(base_tau, TAU_GRID)
    if base is None:
        raise ValueError(f"base_tau {base_tau!r} is not on TAU_GRID")
    if tau_prime is None:
        tau_eff = base
    else:
        tau_eff = snap_to_grid(tau_prime, TAU_GRID)
        if tau_eff is None:
            raise ValueError(f"tau_prime {tau_prime!r} is not on TAU_GRID")
    series = set(map(str, series_ids))
    mult = dict(multipliers) if multipliers else {}
    if len(mult) > MAX_LISTED_SKUS:
        raise ValueError(f"{len(mult)} listed SKUs > {MAX_LISTED_SKUS}")
    norm = []
    for sid, m in mult.items():
        if str(sid) not in series:
            raise ValueError(f"unknown series_id {sid!r}")
        mv = snap_to_grid(m, MULT_GRID)
        if mv is None:
            raise ValueError(f"multiplier {m!r} is not on MULT_GRID")
        if mv != 1.0:
            norm.append((str(sid), mv))
    return AgentAction(base_tau=base, tau_eff=tau_eff,
                       multipliers=tuple(sorted(norm)))


@dataclass(frozen=True)
class ParsedCandidate:
    index: int                       # slot position in the (truncated) array
    candidate_id: str | None         # label only; duplicates allowed
    action: "AgentAction | None"     # None when invalid
    invalid_reason: str | None

    @property
    def is_valid(self) -> bool:
        return self.action is not None


@dataclass
class ParseResult:
    parse_failed: bool
    failure_reason: str | None
    candidates: list[ParsedCandidate] = field(default_factory=list)
    truncated_count: int = 0
    envelope_extra_keys: tuple[str, ...] = ()

    @property
    def valid(self) -> list[ParsedCandidate]:
        return [c for c in self.candidates if c.is_valid]

    def stats(self) -> dict:
        """ST7-02 D5 accounting units (per episode, per arm)."""
        seen: list[AgentAction] = []
        duplicate_slots = 0
        for c in self.valid:
            if c.action in seen:
                duplicate_slots += 1
            else:
                seen.append(c.action)
        return {
            "candidate_slots_used": len(self.candidates),
            "valid_candidates": len(self.valid),
            "invalid_candidates": len(self.candidates) - len(self.valid),
            "duplicate_valid_slots": duplicate_slots,
            "truncated_count": self.truncated_count,
        }


def _validate_candidate(index: int, cand, base_tau: float,
                        series: frozenset[str]) -> ParsedCandidate:
    def invalid(reason: str, cid=None) -> ParsedCandidate:
        return ParsedCandidate(index=index, candidate_id=cid, action=None,
                               invalid_reason=reason)

    if not isinstance(cand, dict):
        return invalid("not_an_object")
    if _DUP_MARKER in cand:
        return invalid("duplicate_json_keys")
    unknown = set(cand) - ALLOWED_KEYS
    if unknown:
        return invalid(f"unknown_field:{sorted(unknown)[0]}")
    cid = cand.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        return invalid("missing_candidate_id")
    if "rationale" in cand and not isinstance(cand["rationale"], str):
        return invalid("bad_rationale", cid)

    tau_prime = cand.get("tau_prime")
    if tau_prime is not None:
        tau_eff = snap_to_grid(tau_prime, TAU_GRID)
        if tau_eff is None:
            return invalid("tau_off_grid", cid)
    else:
        tau_eff = base_tau

    mult_obj = cand.get("sku_multipliers", {})
    if mult_obj is None:
        mult_obj = {}
    if not isinstance(mult_obj, dict):
        return invalid("bad_sku_multipliers", cid)
    if _DUP_MARKER in mult_obj:
        return invalid("duplicate_sku_key", cid)
    if len(mult_obj) > MAX_LISTED_SKUS:
        return invalid("too_many_skus", cid)
    norm = []
    for sid, m in mult_obj.items():
        if sid not in series:
            return invalid(f"unknown_sku:{sid}", cid)
        mv = snap_to_grid(m, MULT_GRID)
        if mv is None:
            return invalid("multiplier_off_grid", cid)
        if mv != 1.0:
            norm.append((sid, mv))

    action = AgentAction(base_tau=base_tau, tau_eff=tau_eff,
                         multipliers=tuple(sorted(norm)))
    return ParsedCandidate(index=index, candidate_id=cid, action=action,
                           invalid_reason=None)


def parse_candidates(text: str, series_ids, base_tau: float) -> ParseResult:
    """Parse one LLM response into validated, normalized candidates.

    Episode-level failures return ParseResult(parse_failed=True) and the
    caller records zero candidates for the episode (ST7-01). Candidate-level
    failures only mark that slot invalid."""
    base = snap_to_grid(base_tau, TAU_GRID)
    if base is None:
        raise ValueError(f"base_tau {base_tau!r} is not on TAU_GRID")
    series = frozenset(map(str, series_ids))

    try:
        obj = json.loads(text, object_pairs_hook=_pairs_hook)
    except (json.JSONDecodeError, TypeError) as exc:
        return ParseResult(parse_failed=True, failure_reason=f"json_error:{exc}")
    if not isinstance(obj, dict):
        return ParseResult(parse_failed=True, failure_reason="envelope_not_object")
    if _DUP_MARKER in obj:
        return ParseResult(parse_failed=True,
                           failure_reason="duplicate_envelope_keys")
    if "candidates" not in obj or not isinstance(obj["candidates"], list):
        return ParseResult(parse_failed=True,
                           failure_reason="missing_candidates_array")

    extras = tuple(sorted(set(obj) - {"candidates"}))
    raw = obj["candidates"]
    slots = raw[:MAX_CANDIDATES]
    truncated = max(0, len(raw) - MAX_CANDIDATES)
    parsed = [_validate_candidate(i, c, base, series)
              for i, c in enumerate(slots)]
    return ParseResult(parse_failed=False, failure_reason=None,
                       candidates=parsed, truncated_count=truncated,
                       envelope_extra_keys=extras)