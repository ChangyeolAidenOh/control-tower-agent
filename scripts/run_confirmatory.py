"""Stage 6 confirmatory analysis (ST6-13; preregistration section 3/6 frozen).

Input: outputs/stage6_eval_core_step6.json ONLY (12,960 episode rows).
Per environment r in {3, 5, 10} (no primary r is designated by the
preregistration; conclusions are scoped per r, cross-r stability is the
sensitivity axis of decision_definition section 6):

  H0   implementation-identity checks (a) deterministic single-period LP
       q*(yhat) = yhat and downstream cost/gradient == cost-weighted
       pinball (rel tol 1e-4, off kinks); (b) discrete-F newsvendor
       argmin == F^-1(tau), tau = b/(h+b). Hard gate before any CI.
  H1   Delta_j = C(A_j) - C(B), j in {a2, a3, a4}: moving-block bootstrap
       lower bound (5th percentile, one-sided 95%) > 0 for ALL three
       (intersection-union, no correction).
  H1SL Delta_SL = CSL(B) - CSL(A_j): lower bound > -0.01 (delta = 1%p).
  H1b  Delta_PTO = C(apto) - C(B): lower bound > 0.
  ref  a5 perfect-information gap, judgment-free (not a lower bound,
       ST6-09); fold 12 carries truncated_oracle_horizon, folds 1-11
       also reported separately.

Mechanics (frozen): episode = week x bundle (480 per policy-r); paired
differences on identical demand paths; week index t = 4*(fold rank) +
(rank of week within fold), t = 1..48; moving blocks = L consecutive
weeks carrying ALL 10 bundles, L = 2 primary, {1, 4} sensitivity,
non-circular starts 1..48-L+1, 48/L blocks per resample, B = 2,000,
rng seed 20261011; decision bound = 5th percentile of the bootstrap
distribution of the mean; central 2.5/97.5 reported alongside; ACF of
the bundle-summed weekly difference series (lags 1-4) reported first.
The sign convention is fixed through improvement_over_baseline().
The whole bootstrap is computed twice and must match bitwise.

Run: python -m scripts.run_confirmatory
"""

import datetime
import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from core_pipeline.data.db import PROJECT_ROOT

IN_PATH = PROJECT_ROOT / "outputs" / "stage6_eval_core_step6.json"
OUT_PATH = PROJECT_ROOT / "outputs" / "stage6_confirmatory.json"
R_EVAL = (3, 5, 10)
H1_BASELINES = ("a2", "a3", "a4")
CAND = "b"
PTO = "apto"
B_BOOT = 2000
BLOCKS = (2, 1, 4)          # first entry = primary
SEED = 20261011
SL_DELTA = 0.01             # 1 percentage point, CSL in [0, 1]
N_WEEKS, N_BUNDLES = 48, 10


def improvement_over_baseline(baseline_cost: np.ndarray,
                              candidate_cost: np.ndarray) -> np.ndarray:
    """Frozen sign rule: positive = candidate better (preregistration
    section 3). NEVER reimplement this difference elsewhere."""
    return np.asarray(baseline_cost, dtype=np.float64) - np.asarray(
        candidate_cost, dtype=np.float64)


# ----------------------------------------------------------------- H0 ---- #

def h0_deterministic() -> dict:
    """(a) single-period LP q*(yhat) = yhat; downstream cost and gradient
    at q* equal the cost-weighted pinball value and gradient."""
    rng = np.random.default_rng(7)
    worst_q = worst_v = worst_g = 0.0
    for _ in range(50):
        h = float(rng.uniform(0.5, 3.0))
        b = float(rng.uniform(1.0, 20.0))
        yhat = float(rng.uniform(0.0, 10.0))
        d = float(rng.uniform(0.0, 10.0))
        if abs(yhat - d) < 0.3:        # stay off the kink (prereg H0)
            continue
        q = cp.Variable(nonneg=True)
        cost = h * cp.pos(q - yhat) + b * cp.pos(yhat - q)
        cp.Problem(cp.Minimize(cost)).solve(solver=cp.CLARABEL)
        worst_q = max(worst_q, abs(float(q.value) - yhat) / max(yhat, 1e-9))

        def downstream(y):
            return h * max(y - d, 0.0) + b * max(d - y, 0.0)

        tau = b / (h + b)
        pin_v = (h + b) * (tau * max(d - yhat, 0.0)
                           + (1 - tau) * max(yhat - d, 0.0))
        worst_v = max(worst_v, abs(downstream(yhat) - pin_v)
                      / max(abs(pin_v), 1e-9))
        eps = 1e-5
        g_num = (downstream(yhat + eps) - downstream(yhat - eps)) / (2 * eps)
        g_pin = h if yhat > d else -b
        worst_g = max(worst_g, abs(g_num - g_pin) / max(abs(g_pin), 1e-9))
    ok = worst_q < 1e-4 and worst_v < 1e-4 and worst_g < 1e-4
    return {"pass": bool(ok), "worst_rel_q": worst_q,
            "worst_rel_value": worst_v, "worst_rel_grad": worst_g}


def h0_distribution() -> dict:
    """(b) discrete F: exact-expectation newsvendor argmin == F^-1(tau)."""
    rng = np.random.default_rng(11)
    n_fail = 0
    for _ in range(50):
        k = int(rng.integers(3, 8))
        vals = np.sort(rng.choice(np.arange(0, 30), size=k, replace=False))
        p = rng.dirichlet(np.ones(k))
        h = float(rng.uniform(0.5, 3.0))
        b = float(rng.uniform(1.0, 20.0))
        tau = b / (h + b)
        cdf = np.cumsum(p)
        if np.any(np.abs(cdf - tau) < 1e-9):   # tie at tau: skip (kink)
            continue
        target = vals[int(np.searchsorted(cdf, tau))]
        exp_cost = [float(np.sum(p * (h * np.maximum(q - vals, 0)
                                      + b * np.maximum(vals - q, 0))))
                    for q in vals]
        if vals[int(np.argmin(exp_cost))] != target:
            n_fail += 1
    return {"pass": n_fail == 0, "n_fail": n_fail}


# ------------------------------------------------------------ loading ---- #

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_tensor(episodes: list[dict], policies: tuple[str, ...]) -> dict:
    """{(policy, r): (cost[48, 10], csl[48, 10])} with identical
    (fold, week, bundle) alignment across policies; week index by rank
    within fold (works for both 1..4-per-fold and global numbering)."""
    folds = sorted({e["fold"] for e in episodes})
    if len(folds) != 12:
        raise RuntimeError(f"expected 12 folds, got {folds}")
    week_rank: dict[tuple, int] = {}
    for fo in folds:
        weeks = sorted({e["week"] for e in episodes if e["fold"] == fo})
        if len(weeks) != 4:
            raise RuntimeError(f"fold {fo}: weeks {weeks} != 4 distinct")
        for i, w in enumerate(weeks):
            week_rank[(fo, w)] = folds.index(fo) * 4 + i
    bundles = sorted({e["group_id"] for e in episodes})
    if len(bundles) != N_BUNDLES:
        raise RuntimeError(f"{len(bundles)} bundles != {N_BUNDLES}")
    b_pos = {g: i for i, g in enumerate(bundles)}

    out = {}
    for pol in policies:
        for r in R_EVAL:
            cost = np.full((N_WEEKS, N_BUNDLES), np.nan)
            csl = np.full((N_WEEKS, N_BUNDLES), np.nan)
            for e in episodes:
                if e["policy"] != pol or e["r"] != r:
                    continue
                t = week_rank[(e["fold"], e["week"])]
                g = b_pos[e["group_id"]]
                if not np.isnan(cost[t, g]):
                    raise RuntimeError(f"duplicate episode {pol} r={r} "
                                       f"fold={e['fold']} week={e['week']}")
                cost[t, g] = e["cost"]
                csl[t, g] = e["csl"]
            if np.isnan(cost).any():
                raise RuntimeError(f"{pol} r={r}: missing episodes")
            out[(pol, r)] = (cost, csl)
    return out, bundles, {str(k): v for k, v in week_rank.items()}


# ---------------------------------------------------------- bootstrap ---- #

def acf(series: np.ndarray, lags=(1, 2, 3, 4)) -> dict:
    x = series - series.mean()
    den = float(np.sum(x * x))
    return {f"lag{k}": float(np.sum(x[k:] * x[:-k]) / den) for k in lags}


def block_bootstrap(weekly_mean: np.ndarray, length: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean of a 48-long weekly series:
    moving blocks of `length` weeks (all bundles travel with the week,
    so the weekly bundle-mean is the exchangeable unit)."""
    t = weekly_mean.shape[0]
    n_blocks = t // length
    starts = rng.integers(0, t - length + 1, size=(B_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(
        B_BOOT, n_blocks * length)
    return weekly_mean[idx].mean(axis=1)


def ci_block(diff_mat: np.ndarray, rng_seed_offset: int) -> dict:
    """diff_mat (48, 10) paired per-episode differences -> point estimate,
    ACF of the bundle-summed weekly series, and per-block-length bounds."""
    weekly_mean = diff_mat.mean(axis=1)
    out = {
        "point_estimate": float(diff_mat.mean()),
        "acf_weekly_sum": acf(diff_mat.sum(axis=1)),
        "blocks": {},
    }
    for li, length in enumerate(BLOCKS):
        rng = np.random.default_rng(SEED + 1000 * rng_seed_offset + li)
        dist = block_bootstrap(weekly_mean, length, rng)
        out["blocks"][f"L{length}"] = {
            "lower_5pct": float(np.percentile(dist, 5.0)),
            "two_sided_2p5": float(np.percentile(dist, 2.5)),
            "two_sided_97p5": float(np.percentile(dist, 97.5)),
        }
    out["primary"] = out["blocks"][f"L{BLOCKS[0]}"]
    return out


# --------------------------------------------------------------- main ---- #

def analyse(tensor) -> dict:
    res: dict = {}
    for r in R_EVAL:
        b_cost, b_csl = tensor[(CAND, r)]
        rr: dict = {"h1": {}, "h1_sl": {}, "h1_b": None, "a5_reference": None}
        off = 0
        for j in H1_BASELINES:
            a_cost, a_csl = tensor[(j, r)]
            d_cost = improvement_over_baseline(a_cost, b_cost)
            c = ci_block(d_cost, off); off += 1
            c["pass"] = c["primary"]["lower_5pct"] > 0.0
            rr["h1"][j] = c
            d_sl = b_csl - a_csl        # Delta_SL = CSL(B) - CSL(A_j)
            s = ci_block(d_sl, off); off += 1
            s["pass"] = s["primary"]["lower_5pct"] > -SL_DELTA
            rr["h1_sl"][j] = s
        rr["h1_pass_intersection"] = all(
            rr["h1"][j]["pass"] for j in H1_BASELINES)
        rr["h1_sl_pass_all"] = all(
            rr["h1_sl"][j]["pass"] for j in H1_BASELINES)

        p_cost, _ = tensor[(PTO, r)]
        c = ci_block(improvement_over_baseline(p_cost, b_cost), off); off += 1
        c["pass"] = c["primary"]["lower_5pct"] > 0.0
        rr["h1_b"] = c

        a5_cost, _ = tensor[("a5", r)]
        gap = improvement_over_baseline(b_cost, a5_cost)   # >0: a5 cheaper
        rr["a5_reference"] = {
            "judgment": "none (not a lower bound, ST6-09; fold 12 "
                        "truncated_oracle_horizon)",
            "mean_gap_b_minus_a5_all": float(gap.mean()),
            "mean_gap_b_minus_a5_folds_1_11": float(gap[:44, :].mean()),
        }
        res[f"r{r}"] = rr
    return res


def main() -> None:
    h0a, h0b = h0_deterministic(), h0_distribution()
    print(f"H0 (a) deterministic: pass={h0a['pass']} "
          f"(q {h0a['worst_rel_q']:.2e}, v {h0a['worst_rel_value']:.2e}, "
          f"g {h0a['worst_rel_grad']:.2e})")
    print(f"H0 (b) distribution:  pass={h0b['pass']} n_fail={h0b['n_fail']}")
    if not (h0a["pass"] and h0b["pass"]):
        raise SystemExit("H0 failed -- fix before any confirmatory claim "
                         "(preregistration section 3)")

    payload = json.loads(IN_PATH.read_text())
    episodes = payload["episodes"]
    pols = (CAND,) + H1_BASELINES + (PTO, "a5")
    tensor, bundles, _ = episode_tensor(episodes, pols)
    print(f"loaded {len(episodes)} episode rows; aligned "
          f"{len(pols)} policies x {len(R_EVAL)} r x 48w x {N_BUNDLES}g")

    res1 = analyse(tensor)
    res2 = analyse(tensor)
    if json.dumps(res1, sort_keys=True) != json.dumps(res2, sort_keys=True):
        raise SystemExit("double-pass mismatch -- nondeterminism in the "
                         "bootstrap (ST6-13 gate)")

    for r in R_EVAL:
        rr = res1[f"r{r}"]
        parts = []
        for j in H1_BASELINES:
            c = rr["h1"][j]
            parts.append(f"{j}:Δ̂={c['point_estimate']:.1f} "
                         f"lb={c['primary']['lower_5pct']:.1f} "
                         f"{'PASS' if c['pass'] else 'FAIL'}")
        print(f"r={r} H1 [{' | '.join(parts)}] -> "
              f"{'PASS' if rr['h1_pass_intersection'] else 'FAIL'} "
              f"(intersection-union)")
        sl = [f"{j}:lb={rr['h1_sl'][j]['primary']['lower_5pct']:+.4f}"
              f"{'✓' if rr['h1_sl'][j]['pass'] else '✗'}"
              for j in H1_BASELINES]
        c = rr["h1_b"]
        print(f"      H1-SL [{' | '.join(sl)}] (delta={SL_DELTA})  "
              f"H1-b Δ̂={c['point_estimate']:.1f} "
              f"lb={c['primary']['lower_5pct']:.1f} "
              f"{'PASS' if c['pass'] else 'FAIL'}")
        a5 = rr["a5_reference"]
        print(f"      a5 gap(B−A5): all={a5['mean_gap_b_minus_a5_all']:.1f} "
              f"folds1-11={a5['mean_gap_b_minus_a5_folds_1_11']:.1f} "
              "(no judgment)")

    OUT_PATH.write_text(json.dumps({
        "task": "Stage 6 confirmatory analysis (ST6-13)",
        "date": datetime.date.today().isoformat(),
        "input_file": IN_PATH.name,
        "input_sha256": file_sha256(IN_PATH),
        "seed": SEED, "B": B_BOOT,
        "blocks": {"primary": BLOCKS[0], "sensitivity": list(BLOCKS[1:])},
        "sl_delta": SL_DELTA,
        "r_interpretation": "per-r judgment; no primary r designated "
                            "(ST6-13); cross-r stability = sensitivity axis",
        "h0": {"deterministic": h0a, "distribution": h0b},
        "results": res1,
        "limits": [
            "H1: training-exposure asymmetry across policies",
            "H1: A4 state reuses validation-window information",
            "H1: training surrogate != rolling evaluation cost",
            "A-PTO: 13/27 tuning trials under the frozen time budget",
            "A-PTO: tuning/retraining moved to Colab GPU (ST6-12)",
            "A-PTO: safety calibration shares the HP validation window",
        ],
        "bundles": bundles,
    }, indent=1) + "\n")
    print(f"written {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()