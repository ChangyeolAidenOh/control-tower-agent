"""Delayed Adaptive CQR Margin Tracker (Stage 3-D, ST3-07).

Methodology naming (fixed): this is ONLINE QUANTILE TRACKING of a
width-normalized CQR margin -- the proportional term of conformal PID --
under a hard 28-day feedback delay. It is a member of the adaptive
conformal family referenced by preregistered H2, but it is NOT the
Gibbs-Candes ACI miscoverage-level update, and no claim is made that
their coverage guarantees carry over.

Feedback contract (preregistration section 2 / simulator spec section 6):
    release(origin, h) = origin + 28   for EVERY h in 1..28.
The scores of a 28-day forecast bundle issued at `origin` enter the
trackers only when the replay clock reaches origin + 28, i.e. once the
whole bundle's targets are observed. Using targets at origin + h as they
arrive would be a different protocol and is deliberately not implemented.

Interval validity (reviewer-mandated safeguards, not post-hoc tuning):
- effective width w_eff = max(hi - lo, eps_i) is used on BOTH sides
  (score normalization and margin application), so a zero-width base
  interval can still be widened once the margin grows;
- eps_i = max(1.0, 0.1 * mean selection-window demand of series i),
  fixed before any result is seen;
- over-shrunk intervals (hi_t < lo_t under negative margin) collapse to
  the base-interval midpoint, then both ends are clipped at 0 (demand
  is non-negative), so 0 <= lo_adj <= hi_adj always holds;
- the tracker update uses the miss indicator of the ACTUALLY ISSUED
  interval (after collapse/clipping), not the raw normalized score:
  m <- clip(m + GAMMA * (miss - ALPHA), M_MIN, M_MAX).

All constants below are fixed project choices recorded before results
were available; GAMMA in particular is a project-chosen learning rate,
not a literature-standard default.
"""

from __future__ import annotations

import numpy as np

from core_pipeline.forecast.dataset import DemandHistory

ALPHA = 0.10          # target miscoverage of the 90% base interval
GAMMA = 0.01          # pre-fixed learning rate (project choice)
M_MIN, M_MAX = -0.5, 5.0
EPS_ABS = 1.0         # absolute width floor (one demand unit)
EPS_REL = 0.1         # relative floor vs selection-window mean demand
RELEASE_LAG = 28      # feedback delay in days, == horizon by contract
HORIZON = 28


def eps_floor(selection_mean: np.ndarray) -> np.ndarray:
    """eps_i = max(EPS_ABS, EPS_REL * mean daily demand <= d_1605)."""
    return np.maximum(EPS_ABS, EPS_REL * np.asarray(selection_mean, float))


def adjust_intervals(lo: np.ndarray, hi: np.ndarray, m: np.ndarray,
                     w_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Margin-adjusted, always-valid output intervals (vectorized)."""
    lo_t = lo - m * w_eff
    hi_t = hi + m * w_eff
    inverted = hi_t < lo_t
    if inverted.any():
        mid = 0.5 * (lo + hi)
        lo_t = np.where(inverted, mid, lo_t)
        hi_t = np.where(inverted, mid, hi_t)
    lo_adj = np.maximum(lo_t, 0.0)
    hi_adj = np.maximum(hi_t, 0.0)
    return lo_adj, hi_adj


def update_m(m: np.ndarray, miss: np.ndarray) -> np.ndarray:
    return np.clip(m + GAMMA * (miss.astype(float) - ALPHA), M_MIN, M_MAX)


def replay(origins: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           history: DemandHistory) -> dict:
    """Run the delayed online replay over a contiguous daily origin grid.

    origins: sorted int array (n_origins,), consecutive days.
    lo, hi:  (n_origins, n_series, HORIZON) base q0.05 / q0.95, series
             axis ordered as history.series_ids, horizon axis = 1..28.
    history: current_idx == origins[0]; preloaded through >= origins[-1].

    Returns issued intervals and tracker bookkeeping. State is ONE
    process across the whole grid (fold-boundary continuity, ST3-04/3D-4);
    origins with origin + RELEASE_LAG > origins[-1] are issued but never
    release feedback inside the replay (the trailing-28 contract).
    """
    origins = np.asarray(origins, dtype=np.int64)
    n_o, n_s, n_h = lo.shape
    if n_h != HORIZON or hi.shape != lo.shape or len(origins) != n_o:
        raise ValueError("shape mismatch")
    if not (np.diff(origins) == 1).all():
        raise ValueError("origins must be consecutive days")
    if history.current_idx != origins[0]:
        raise ValueError("history must start observed exactly at origins[0]")
    if n_s != len(history.series_ids):
        raise ValueError("series axis mismatch")

    sel = history.observed_through(origins[0])
    eps = eps_floor(sel.mean(axis=1))[:, None]          # (n_s, 1)
    w_eff = np.maximum(hi - lo, eps[None, :, :])        # (n_o, n_s, n_h)

    m = np.zeros((n_s, n_h))
    upd = np.zeros((n_s, n_h), dtype=np.int64)
    lo_adj = np.empty_like(lo, dtype=np.float64)
    hi_adj = np.empty_like(hi, dtype=np.float64)
    m_at = np.empty_like(lo, dtype=np.float64)
    upd_before = np.empty((n_o, n_s, n_h), dtype=np.int64)
    pos = {int(o): k for k, o in enumerate(origins)}
    idx0 = history.idx_start

    for k, o in enumerate(origins):
        t = int(o)
        if k > 0:
            history = history.advance()                 # reveal day t
        assert history.current_idx == t
        rel = t - RELEASE_LAG
        if rel in pos:                                  # release bundle
            j = pos[rel]
            obs = history.observed_through(t)
            y = obs[:, rel + 1 - idx0: rel + HORIZON + 1 - idx0]
            assert y.shape == (n_s, HORIZON)
            miss = (y < lo_adj[j]) | (y > hi_adj[j])    # issued interval
            m = update_m(m, miss)
            upd += 1
        m_at[k] = m
        upd_before[k] = upd
        lo_adj[k], hi_adj[k] = adjust_intervals(lo[k], hi[k], m, w_eff[k])

    released = int((origins + RELEASE_LAG <= origins[-1]).sum())
    return {
        "lo_adj": lo_adj, "hi_adj": hi_adj, "m_at_origin": m_at,
        "updates_before": upd_before, "m_final": m, "eps": eps[:, 0],
        "origins_released": released,
        "origins_unreleased": int(n_o - released),
    }
