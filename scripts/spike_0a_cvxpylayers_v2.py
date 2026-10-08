import argparse
import json
import time

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.float64)


def solver_kwargs(solver):
    if solver == "ECOS":
        return {"solve_method": "ECOS", "feastol": 1e-9, "abstol": 1e-9, "reltol": 1e-9, "max_iters": 500}
    if solver == "SCS":
        return {"solve_method": "SCS", "eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 100000}
    if solver == "CLARABEL":
        return {"solve_method": "Clarabel"}
    raise ValueError(solver)


def build_layer(n_sku, horizon, lead_time, hold_cost_vec, short_cost_vec, eps_reg, use_capacity):
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
    if eps_reg > 0:
        cost = cost + eps_reg * cp.sum_squares(q)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    assert problem.is_dpp(), "problem is not DPP"
    return CvxpyLayer(problem, parameters=params, variables=[q])


def true_cost(q, y, init_inv, pipeline, lead_time, hold_t, short_t):
    n_sku, horizon = q.shape
    inv = init_inv
    total = torch.zeros((), dtype=q.dtype)
    for t in range(horizon):
        arrival = pipeline[:, t] if t < lead_time else q[:, t - lead_time]
        avail = inv + arrival
        short = torch.relu(y[:, t] - avail)
        inv = torch.relu(avail - y[:, t])
        total = total + (hold_t * inv).sum() + (short_t * short).sum()
    return total


def make_instance(n_sku, horizon, lead_time, seed, cap_ratio):
    rng = np.random.default_rng(seed)
    mean_demand = rng.uniform(10.0, 30.0, size=n_sku)
    hold_cost = rng.uniform(0.5, 2.0, size=n_sku)
    short_cost = rng.uniform(3.0, 8.0, size=n_sku)
    y = rng.gamma(shape=4.0, scale=mean_demand[:, None] / 4.0, size=(n_sku, horizon))
    d_hat = np.clip(y + rng.normal(0.0, 0.15 * mean_demand[:, None], size=y.shape), 0.1, None)
    init_inv = mean_demand * 1.5
    pipeline = np.tile(mean_demand[:, None] * 0.8, (1, lead_time))
    cap = np.full(horizon, cap_ratio * mean_demand.sum())
    return y, d_hat, init_inv, pipeline, cap, hold_cost, short_cost


class Runner:
    def __init__(self, layer, inst, lead_time, solver):
        self.layer = layer
        self.d_hat, self.init_inv, self.pipeline, self.cap, self.y, self.hold, self.short = inst
        self.lead_time = lead_time
        self.kwargs = solver_kwargs(solver)

    def loss(self, d_var):
        params = [d_var, self.init_inv, self.pipeline]
        if self.cap is not None:
            params.append(self.cap)
        (q,) = self.layer(*params, solver_args=self.kwargs)
        return true_cost(q, self.y, self.init_inv, self.pipeline, self.lead_time, self.hold, self.short)

    def fd_with_kink_detection(self, n_entries, fd_eps, seed):
        d_var = self.d_hat.clone().requires_grad_(True)
        loss = self.loss(d_var)
        loss.backward()
        grad_bp = d_var.grad.detach().clone()
        f0 = loss.item()

        rng = np.random.default_rng(seed)
        idx = rng.choice(self.d_hat.numel(), size=min(n_entries, self.d_hat.numel()), replace=False)
        n_cols = self.d_hat.shape[1]
        rel_err_smooth = []
        n_kink = 0
        for flat in idx:
            i, t = divmod(int(flat), n_cols)
            plus = self.d_hat.clone()
            minus = self.d_hat.clone()
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
            "nan_in_backprop": int(torch.isnan(grad_bp).sum().item()),
            "grad_norm": float(grad_bp.norm().item()),
        }

    def jitter(self, n_rep, jitter_sd, seed):
        torch.manual_seed(seed)
        grads = []
        for _ in range(n_rep):
            d_var = (self.d_hat + jitter_sd * torch.randn_like(self.d_hat)).clamp_min(0.1).requires_grad_(True)
            self.loss(d_var).backward()
            grads.append(d_var.grad.detach().flatten())
        g = torch.stack(grads)
        return {"grad_rel_spread_under_jitter": float(g.std(dim=0).norm().item() / max(g.mean(dim=0).norm().item(), 1e-9))}

    def descent(self, n_steps, lr):
        d_var = self.d_hat.clone().requires_grad_(True)
        opt = torch.optim.Adam([d_var], lr=lr)
        costs = []
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad()
            loss = self.loss(d_var)
            loss.backward()
            opt.step()
            with torch.no_grad():
                d_var.clamp_(min=0.1)
            costs.append(loss.item())
        with torch.no_grad():
            oracle_cost = self.loss(self.y.clone()).item()
        return {
            "descent_cost_start": costs[0],
            "descent_cost_min": float(min(costs)),
            "descent_cost_end": costs[-1],
            "descent_rel_improvement": float((costs[0] - min(costs)) / max(costs[0], 1e-9)),
            "cost_with_perfect_forecast": oracle_cost,
            "sec_per_step": (time.perf_counter() - t0) / n_steps,
        }


def run_config(n_sku, horizon, lead_time, eps_reg, use_capacity, cap_ratio, solver, seed):
    y, d_hat, init_inv, pipeline, cap, hold, short = make_instance(n_sku, horizon, lead_time, seed, cap_ratio)
    layer = build_layer(n_sku, horizon, lead_time, hold, short, eps_reg, use_capacity)
    inst = (
        torch.tensor(d_hat), torch.tensor(init_inv), torch.tensor(pipeline),
        torch.tensor(cap) if use_capacity else None, torch.tensor(y),
        torch.tensor(hold), torch.tensor(short),
    )
    runner = Runner(layer, inst, lead_time, solver)
    out = {"n_sku": n_sku, "horizon": horizon, "eps_reg": eps_reg, "shared_capacity": use_capacity, "solver": solver}
    out.update(runner.fd_with_kink_detection(n_entries=30, fd_eps=1e-3, seed=seed))
    out.update(runner.jitter(n_rep=8, jitter_sd=1e-3, seed=seed))
    out.update(runner.descent(n_steps=30, lr=0.5))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="ECOS", choices=["ECOS", "SCS", "CLARABEL"])
    parser.add_argument("--lead_time", type=int, default=2)
    parser.add_argument("--cap_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/spike_0a_v2_results.json")
    args = parser.parse_args()

    configs = []
    for n_sku, horizon in [(8, 14), (8, 28), (32, 28)]:
        for eps_reg in [0.0, 1e-2, 1e-1]:
            configs.append(dict(n_sku=n_sku, horizon=horizon, eps_reg=eps_reg, use_capacity=True))

    results = []
    for cfg in configs:
        try:
            res = run_config(cfg["n_sku"], cfg["horizon"], args.lead_time, cfg["eps_reg"], cfg["use_capacity"], args.cap_ratio, args.solver, args.seed)
        except Exception as exc:
            res = {**cfg, "solver": args.solver, "error": repr(exc)}
        results.append(res)
        print(json.dumps(res))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()