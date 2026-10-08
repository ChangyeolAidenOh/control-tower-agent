"""2-F: evaluation demand-path construction -- censoring stress and demand
shock (spec v0.1 section 5).

Paths are built ONCE per (fold, scenario) and shared by every compared policy
(paired comparison; decision log 2-A). All randomness enters here, through
hierarchical streams:
  censor designation: spawn_stream(seeds.censor, fold, bundle_idx)
  shock draw:         spawn_stream(seeds.shock, fold)   -- fold level, so that
                      exactly ONE bundle per fold is shocked (drawing per
                      bundle would break that contract).

Censoring stress (scenario axis p in {0.025, 0.06, 0.17}; core p = 0):
M5 zero-sales days may hide stockouts, so the stress builds an alternative
latent path: within the designation window, n_i = round_half_up(p x active
days) zero-sales days per SKU are designated (uniformly, without replacement)
and their demand is replaced by the mean of that SKU's NON-designated days in
the preceding 28 days of the activity period. Active days = window days on or
after the SKU's first positive-sale day in the full history. When a SKU has
fewer eligible zero-sales days than n_i, all of them are designated and the
achieved share p_hat_i = designated / active is recorded (preregistration:
p_hat reported). round_half_up is used (documented choice; numpy's banker's
rounding would make round(0.5) = 0).

Demand shock (part of the standard evaluation paths, not a scenario axis):
per fold, one bundle and one start day are drawn at fold level; that bundle's
demand is multiplied by 1.5 on 7 consecutive test-window days (all SKUs).

Composition order is CENSOR THEN SHOCK: the shock multiplies imputed values
inside its span too ("demand ran 50% higher that week"), and since the shock
never creates or removes zero-sales days outside its multiplication, the
designation set is computed on the original path.
"""

from dataclasses import dataclass

import numpy as np

from core_pipeline.simulator.engine import SimConfig, spawn_stream


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


@dataclass(frozen=True)
class CensoringResult:
    demand: np.ndarray        # full-history copy with designated days replaced
    designated: np.ndarray    # bool mask, same shape as demand (True = imputed)
    n_active: np.ndarray      # (n_sku,) active days inside the window
    n_target: np.ndarray      # (n_sku,) round_half_up(p x active)
    n_designated: np.ndarray  # (n_sku,) actually designated
    p_hat: np.ndarray         # (n_sku,) designated / active (0 where no activity)


def apply_censoring(
    mat: np.ndarray,          # (n_hist_days, n_sku) full latent history, day 1 = row 0
    p: float,
    rng: np.random.Generator,
    *,
    window_rows: tuple[int, int],   # inclusive row range to designate within
    lookback_days: int,
) -> CensoringResult:
    n_days, n_sku = mat.shape
    lo, hi = window_rows
    out = mat.copy()
    designated = np.zeros_like(mat, dtype=bool)
    n_active = np.zeros(n_sku, dtype=int)
    n_target = np.zeros(n_sku, dtype=int)
    n_desig = np.zeros(n_sku, dtype=int)

    for i in range(n_sku):
        positive = np.nonzero(mat[:, i] > 0)[0]
        if positive.size == 0:
            continue
        act_start = positive[0]
        window_days = np.arange(max(lo, act_start), hi + 1)
        n_active[i] = window_days.size
        if p <= 0.0 or window_days.size == 0:
            continue
        eligible = window_days[mat[window_days, i] == 0.0]
        target = _round_half_up(p * window_days.size)
        n_target[i] = target
        take = min(target, eligible.size)
        if take > 0:
            chosen = rng.choice(eligible, size=take, replace=False)
            chosen = np.sort(chosen)
            designated[chosen, i] = True
            n_desig[i] = take
            for day in chosen:
                lb = max(act_start, day - lookback_days)
                span = np.arange(lb, day)
                span = span[~designated[span, i]]
                if span.size > 0:
                    out[day, i] = float(mat[span, i].mean())
                # no prior non-designated activity days: value left unchanged

    p_hat = np.divide(n_desig, n_active, out=np.zeros(n_sku), where=n_active > 0)
    return CensoringResult(
        demand=out, designated=designated, n_active=n_active,
        n_target=n_target, n_designated=n_desig, p_hat=p_hat,
    )


@dataclass(frozen=True)
class ShockDraw:
    group_id: str
    start_offset: int         # 0-based day offset inside the test window
    duration_days: int
    multiplier: float


def draw_shock(
    group_ids: list[str], test_window_days: int, cfg: SimConfig, fold: int,
) -> ShockDraw:
    """Fold-level draw (seed D): one bundle and one start day per fold."""
    rng = spawn_stream(cfg.seeds.shock, fold)
    gid = group_ids[int(rng.integers(0, len(group_ids)))]
    max_start = test_window_days - cfg.shock_duration_days
    start = int(rng.integers(0, max_start + 1))
    return ShockDraw(
        group_id=gid, start_offset=start,
        duration_days=cfg.shock_duration_days, multiplier=cfg.shock_multiplier,
    )


def apply_shock(
    window: np.ndarray,       # (test_window_days, n_sku) of the SHOCKED bundle
    draw: ShockDraw,
) -> np.ndarray:
    out = window.copy()
    sl = slice(draw.start_offset, draw.start_offset + draw.duration_days)
    out[sl] = out[sl] * draw.multiplier
    return out


@dataclass(frozen=True)
class BundlePaths:
    """Evaluation inputs for one fold x bundle under one scenario."""
    history: np.ndarray       # rows < warmup_start (policy observation history)
    warmup: np.ndarray        # warm-up window demand
    test: np.ndarray          # test window demand (shock applied if drawn here)
    censoring: CensoringResult | None
    shocked: bool


def build_fold_paths(
    demand_by_bundle: dict[str, tuple[tuple[str, ...], np.ndarray]],
    *,
    fold: int,
    warmup_start_row: int,    # 0-based row of warm-up day 1
    test_start_row: int,      # 0-based row of test day 1
    test_end_row: int,        # 0-based row of test last day (inclusive)
    p: float,
    cfg: SimConfig,
) -> dict[str, BundlePaths]:
    """Assemble per-bundle paths for one fold and censoring level p.
    Censoring designates within the test window only; the shock lands on the
    fold-level drawn bundle. Deterministic given (seeds, fold, p)."""
    group_ids = sorted(demand_by_bundle)
    test_days = test_end_row - test_start_row + 1
    shock = draw_shock(group_ids, test_days, cfg, fold)
    out = {}
    for b_idx, gid in enumerate(group_ids):
        _, mat = demand_by_bundle[gid]
        cens = None
        path = mat
        if p > 0.0:
            rng = spawn_stream(cfg.seeds.censor, fold, b_idx)
            cens = apply_censoring(
                mat, p, rng,
                window_rows=(test_start_row, test_end_row),
                lookback_days=cfg.censor_replacement_lookback_days,
            )
            path = cens.demand
        test = path[test_start_row: test_end_row + 1]
        shocked = gid == shock.group_id
        if shocked:
            test = apply_shock(test, shock)
        out[gid] = BundlePaths(
            history=path[:warmup_start_row],
            warmup=path[warmup_start_row: test_start_row],
            test=test,
            censoring=cens,
            shocked=shocked,
        )
    return out
