import argparse
import json
import time

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.float64)

EXTREME_LOGIT_LEVELS = [-5.0, -10.0, -20.0, -40.0]


def solver_kwargs(solver):
    if solver == "ECOS":
        return {"solve_method": "ECOS", "feastol": 1e-9, "abstol": 1e-9, "reltol": 1e-9, "max_iters": 500}
    if solver == "SCS":
        return {"solve_method": "SCS", "eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 100000}
    if solver == "CLARABEL":
        return {"solve_method": "Clarabel"}
    raise ValueError(solver)


def build_layer(n_sku, horizon, lead_time, hold_cost_vec, short_cost_vec, use_capacity):
    # Same economics as spike 0-A v2: per-SKU h/b, shared capacity, eps_reg = 0 (frozen).
    d_hat = cp.Parameter((n_sku, horizon), nonneg=True)
    init_inv = cp.Parameter(n_sku, nonneg=True)
    pipeline = cp.Parameter((n_sku, lead_time), nonneg=True)
    q = cp.Variable((n_sku, horizon), nonneg=True)
    inv = cp.Variable((n_sku, horizon), nonneg=True)
    short = cp.Variable((n_sku, horizon), nonneg=True)

    constraints = []
    for t in range(horizon):
        prev = init_inv if t == 0 else inv[:, t - 1]
        arrival = pipeline[:, t] if t < lead_time else q[:, t - lead_time]
        constraints.append(inv[:, t] - short[:, t] == prev + arrival - d_hat[:, t])

    params = [d_hat, init_inv, pipeline]
    if use_capacity:
        cap = cp.Parameter(horizon, nonneg=True)
        constraints.append(cp.sum(q, axis=0) <= cap)
        params.append(cap)

    cost = cp.sum(cp.multiply(hold_cost_vec[:, None], inv)) + cp.sum(cp.multiply(short_cost_vec[:, None], short))
    problem = cp.Problem(cp.Minimize(cost), constraints)
    assert problem.is_dpp(), "problem is not DPP"
    return CvxpyLayer(problem, parameters=params, variables=[q])


def true_cost(q, y, init_inv, pipeline, lead_time, hold_t, short_t):
    # Piecewise-linear lost-sales cost under realized demand y, always fp64.
    n_sku, horizon = q.shape
    inv = init_inv
    total = torch.zeros((), dtype=torch.float64)
    for t in range(horizon):
        arrival = pipeline[:, t] if t < lead_time else q[:, t - lead_time]
        avail = inv + arrival
        short = torch.relu(y[:, t] - avail)
        inv = torch.relu(avail - y[:, t])
        total = total + (hold_t * inv).sum() + (short_t * short).sum()
    return total


def softplus_inv(x):
    return torch.log(torch.expm1(x.clamp_min(1e-6)))


def make_instance(n_sku, horizon, lead_time, seed, cap_ratio, n_zero_sku):
    rng = np.random.default_rng(seed)
    mean_demand = rng.uniform(10.0, 30.0, size=n_sku)
    hold_cost = rng.uniform(0.5, 2.0, size=n_sku)
    short_cost = rng.uniform(3.0, 8.0, size=n_sku)
    y = rng.gamma(shape=4.0, scale=mean_demand[:, None] / 4.0, size=(n_sku, horizon))
    zero_idx = np.arange(n_zero_sku)
    if n_zero_sku > 0:
        y[zero_idx, :] = 0.0
        mean_demand[zero_idx] = 0.0
    d_hat0 = np.clip(y + rng.normal(0.0, 0.15 * np.maximum(mean_demand[:, None], 1.0), size=y.shape), 1e-3, None)
    if n_zero_sku > 0:
        # Deliberately wrong positive forecast on zero-demand SKUs: descent must drive it toward 0.
        d_hat0[zero_idx, :] = 5.0
    # Per-SKU scale = train mean with floor 1.0 (ST4-02 scale contract analog).
    scale = np.maximum(mean_demand, 1.0)
    init_inv = np.maximum(mean_demand, 1.0) * 1.5
    pipeline = np.tile(np.maximum(mean_demand, 1.0)[:, None] * 0.8, (1, lead_time))
    cap = np.full(horizon, cap_ratio * np.maximum(mean_demand.sum(), 1.0))
    return y, d_hat0, scale, init_inv, pipeline, cap, hold_cost, short_cost, zero_idx


class HeadRunner:
    # Full path: z (head_dtype) -> softplus -> x scale -> cast fp64 -> LP -> true cost minus
    # perfect-information cost (regret; the subtracted term is constant in z).

    def __init__(self, layer, inst, lead_time, solver, head_dtype):
        (self.d_hat0, self.scale, self.init_inv, self.pipeline,
         self.cap, self.y, self.hold, self.short) = inst
        self.layer = layer
        self.lead_time = lead_time
        self.kwargs = solver_kwargs(solver)
        self.head_dtype = head_dtype
        self.scale_h = self.scale.to(head_dtype)[:, None]
        with torch.no_grad():
            self.oracle_cost = self._downstream(self.y.clone()).item()

    def init_logits(self, d_hat=None):
        d = self.d_hat0 if d_hat is None else d_hat
        z = softplus_inv(d / self.scale[:, None].clamp_min(1e-9)).to(self.head_dtype)
        return z.clone().requires_grad_(True)

    def _downstream(self, d_fp64):
        params = [d_fp64, self.init_inv, self.pipeline]
        if self.cap is not None:
            params.append(self.cap)
        (q,) = self.layer(*params, solver_args=self.kwargs)
        return true_cost(q, self.y, self.init_inv, self.pipeline, self.lead_time, self.hold, self.short)

    def loss(self, z):
        d = torch.nn.functional.softplus(z) * self.scale_h
        return self._downstream(d.to(torch.float64)) - self.oracle_cost

    def fd_full_path(self, n_entries, fd_eps, seed):
        z = self.init_logits()
        loss = self.loss(z)
        loss.backward()
        grad_bp = z.grad.detach().clone().to(torch.float64)
        f0 = loss.item()

        rng = np.random.default_rng(seed)
        idx = rng.choice(z.numel(), size=min(n_entries, z.numel()), replace=False)
        n_cols = z.shape[1]
        rel_err_smooth = []
        n_kink = 0
        with torch.no_grad():
            base = z.detach().clone()
        for flat in idx:
            i, t = divmod(int(flat), n_cols)
            plus = base.clone()
            minus = base.clone()
            plus[i, t] += fd_eps
            minus[i, t] -= fd_eps
            with torch.no_grad():
                fp = self.loss(plus).item()
                fm = self.loss(minus).item()
            right = (fp - f0) / fd_eps
            left = (f0 - fm) / fd_eps
            scale = max(abs(left), abs(right), 1e-6)
            if abs(left - right) > 0.05 * scale:
                n_kink += 1
                continue
            central = (fp - fm) / (2.0 * fd_eps)
            rel_err_smooth.append(abs(grad_bp[i, t].item() - central) / max(abs(central), 1e-6))

        return {
            "n_checked": int(len(idx)),
            "kink_fraction": n_kink / len(idx),
            "n_smooth": len(rel_err_smooth),
            "smooth_frac_rel_err_below_0.05": float(np.mean(np.array(rel_err_smooth) < 0.05)) if rel_err_smooth else None,
            "smooth_median_rel_err": float(np.median(rel_err_smooth)) if rel_err_smooth else None,
            "nan_or_inf_in_backprop": int((~torch.isfinite(grad_bp)).sum().item()),
            "grad_norm": float(grad_bp.norm().item()),
        }

    def extreme_logits(self, n_entries, seed):
        rng = np.random.default_rng(seed)
        n_cols = self.d_hat0.shape[1]
        out = []
        for level in EXTREME_LOGIT_LEVELS:
            z = self.init_logits()
            idx = rng.choice(z.numel(), size=min(n_entries, z.numel()), replace=False)
            with torch.no_grad():
                for flat in idx:
                    i, t = divmod(int(flat), n_cols)
                    z[i, t] = level
            z = z.detach().clone().requires_grad_(True)
            try:
                loss = self.loss(z)
                loss.backward()
            except Exception as exc:
                out.append({"logit_level": level, "error": repr(exc)})
                continue
            g = z.grad.detach().to(torch.float64)
            g_sel = torch.stack([g[divmod(int(f), n_cols)] for f in idx])
            out.append({
                "logit_level": level,
                "sigmoid_level": float(torch.sigmoid(torch.tensor(level, dtype=torch.float64)).item()),
                "injected_grad_abs_max": float(g_sel.abs().max().item()),
                "injected_grad_abs_median": float(g_sel.abs().median().item()),
                "injected_grad_all_finite": bool(torch.isfinite(g_sel).all().item()),
                "full_grad_nonfinite": int((~torch.isfinite(g)).sum().item()),
                "loss": float(loss.item()),
            })
        return {"extreme_logits": out}

    def descent(self, n_steps, lr, zero_idx=None, tag="descent"):
        z = self.init_logits()
        opt = torch.optim.Adam([z], lr=lr)
        losses, grad_norms = [], []
        zero_traj = []
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad()
            loss = self.loss(z)
            loss.backward()
            g = z.grad.detach().to(torch.float64)
            grad_norms.append(float(g.norm().item()))
            losses.append(float(loss.item()))
            if zero_idx is not None and len(zero_idx) > 0:
                with torch.no_grad():
                    d = (torch.nn.functional.softplus(z) * self.scale_h).to(torch.float64)
                zero_traj.append({
                    "zero_grad_norm": float(g[zero_idx].norm().item()),
                    "zero_d_mean": float(d[zero_idx].mean().item()),
                    "zero_d_max": float(d[zero_idx].max().item()),
                    "zero_logit_min": float(z.detach()[zero_idx].min().item()),
                })
            opt.step()
        sec_per_step = (time.perf_counter() - t0) / n_steps
        out = {
            f"{tag}_regret_start": losses[0],
            f"{tag}_regret_min": float(min(losses)),
            f"{tag}_regret_end": losses[-1],
            f"{tag}_rel_improvement": float((losses[0] - min(losses)) / max(abs(losses[0]), 1e-9)),
            f"{tag}_grad_norm_first5_mean": float(np.mean(grad_norms[:5])),
            f"{tag}_grad_norm_last5_mean": float(np.mean(grad_norms[-5:])),
            f"{tag}_grad_all_finite": bool(np.isfinite(grad_norms).all()),
            f"{tag}_sec_per_step": sec_per_step,
        }
        if zero_traj:
            out.update({
                "zero_d_mean_start": zero_traj[0]["zero_d_mean"],
                "zero_d_mean_end": zero_traj[-1]["zero_d_mean"],
                "zero_d_max_end": zero_traj[-1]["zero_d_max"],
                "zero_logit_min_end": zero_traj[-1]["zero_logit_min"],
                "zero_grad_norm_start": zero_traj[0]["zero_grad_norm"],
                "zero_grad_norm_end": zero_traj[-1]["zero_grad_norm"],
                "zero_grad_vanish_step": next(
                    (k for k, r in enumerate(zero_traj) if r["zero_grad_norm"] < 1e-6), None),
                "zero_traj_every5": zero_traj[::5],
            })
        return out


def run_config(cfg, args):
    n_zero = cfg.get("n_zero_sku", 0)
    y, d0, scale, init_inv, pipeline, cap, hold, short, zero_idx = make_instance(
        cfg["n_sku"], cfg["horizon"], args.lead_time, args.seed, args.cap_ratio, n_zero)
    layer = build_layer(cfg["n_sku"], cfg["horizon"], args.lead_time, hold, short, True)
    head_dtype = torch.float32 if cfg["head_dtype"] == "fp32" else torch.float64
    inst = (
        torch.tensor(d0), torch.tensor(scale), torch.tensor(init_inv), torch.tensor(pipeline),
        torch.tensor(cap), torch.tensor(y), torch.tensor(hold), torch.tensor(short),
    )
    runner = HeadRunner(layer, inst, args.lead_time, args.solver, head_dtype)
    out = {**cfg, "solver": args.solver, "oracle_cost": runner.oracle_cost}
    fd_eps = 1e-2 if cfg["head_dtype"] == "fp32" else 1e-3
    if cfg["experiment"] == "fd":
        out.update(runner.fd_full_path(n_entries=30, fd_eps=fd_eps, seed=args.seed))
    elif cfg["experiment"] == "extreme":
        out.update(runner.extreme_logits(n_entries=6, seed=args.seed))
    elif cfg["experiment"] == "zero_descent":
        zi = torch.tensor(zero_idx, dtype=torch.long)
        out.update(runner.descent(n_steps=args.zero_steps, lr=args.lr, zero_idx=zi, tag="zero"))
    elif cfg["experiment"] == "descent":
        out.update(runner.descent(n_steps=args.descent_steps, lr=args.lr, tag="head"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="CLARABEL", choices=["CLARABEL", "ECOS", "SCS"])
    parser.add_argument("--lead_time", type=int, default=2)
    parser.add_argument("--cap_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--descent_steps", type=int, default=30)
    parser.add_argument("--zero_steps", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="outputs/spike_4b_results.json")
    args = parser.parse_args()

    shapes = [(8, 14)] if args.quick else [(8, 14), (8, 28)]
    configs = []
    for n_sku, horizon in shapes:
        for hd in ["fp64", "fp32"]:
            configs.append(dict(experiment="fd", n_sku=n_sku, horizon=horizon, head_dtype=hd))
        configs.append(dict(experiment="extreme", n_sku=n_sku, horizon=horizon, head_dtype="fp32"))
        configs.append(dict(experiment="zero_descent", n_sku=n_sku, horizon=horizon, head_dtype="fp32", n_zero_sku=n_sku // 2))
        configs.append(dict(experiment="descent", n_sku=n_sku, horizon=horizon, head_dtype="fp32"))

    results = []
    for cfg in configs:
        try:
            res = run_config(cfg, args)
        except Exception as exc:
            res = {**cfg, "solver": args.solver, "error": repr(exc)}
        results.append(res)
        print(json.dumps(res))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
