import argparse
import json
import time

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.float64)


def build_layer(n_sku, horizon, lead_time, hold_cost, short_cost, eps_reg, use_capacity):
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

    cost = hold_cost * cp.sum(inv) + short_cost * cp.sum(short)
    if eps_reg > 0:
        cost = cost + eps_reg * cp.sum_squares(q)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    assert problem.is_dpp(), "problem is not DPP"
    layer = CvxpyLayer(problem, parameters=params, variables=[q])
    return layer


def true_cost(q, y, init_inv, pipeline, lead_time, hold_cost, short_cost):
    # Piecewise-linear lost-sales inventory cost under realized demand y.
    n_sku, horizon = q.shape
    inv = init_inv
    total = torch.zeros((), dtype=q.dtype)
    for t in range(horizon):
        arrival = pipeline[:, t] if t < lead_time else q[:, t - lead_time]
        avail = inv + arrival
        short = torch.relu(y[:, t] - avail)
        inv = torch.relu(avail - y[:, t])
        total = total + hold_cost * inv.sum() + short_cost * short.sum()
    return total


def make_instance(n_sku, horizon, lead_time, seed, cap_ratio):
    rng = np.random.default_rng(seed)
    mean_demand = rng.uniform(10.0, 30.0, size=n_sku)
    y = rng.gamma(shape=4.0, scale=mean_demand[:, None] / 4.0, size=(n_sku, horizon))
    d_hat = np.clip(y + rng.normal(0.0, 0.15 * mean_demand[:, None], size=y.shape), 0.1, None)
    init_inv = mean_demand * 1.5
    pipeline = np.tile(mean_demand[:, None] * 0.8, (1, lead_time))
    cap = np.full(horizon, cap_ratio * mean_demand.sum())
    return y, d_hat, init_inv, pipeline, cap


def layer_loss(layer, d_hat_t, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver):
    params = [d_hat_t, init_inv_t, pipeline_t]
    if cap_t is not None:
        params.append(cap_t)
    (q,) = layer(*params, solver_args={"solve_method": solver})
    return true_cost(q, y_t, init_inv_t, pipeline_t, lead_time, hold_cost, short_cost), q


def finite_diff_check(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_entries, fd_eps, seed):
    d_hat_t, init_inv_t, pipeline_t, cap_t, y_t = inst_t
    d_var = d_hat_t.clone().requires_grad_(True)
    loss, _ = layer_loss(layer, d_var, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver)
    loss.backward()
    grad_bp = d_var.grad.detach().clone()

    rng = np.random.default_rng(seed)
    n_total = d_hat_t.numel()
    idx = rng.choice(n_total, size=min(n_entries, n_total), replace=False)
    grad_fd = np.zeros(len(idx))
    for k, flat in enumerate(idx):
        i, t = divmod(int(flat), d_hat_t.shape[1])
        plus = d_hat_t.clone()
        minus = d_hat_t.clone()
        plus[i, t] += fd_eps
        minus[i, t] -= fd_eps
        with torch.no_grad():
            lp, _ = layer_loss(layer, plus, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver)
            lm, _ = layer_loss(layer, minus, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver)
        grad_fd[k] = (lp - lm).item() / (2.0 * fd_eps)

    grad_bp_sel = grad_bp.flatten()[idx].numpy()
    abs_err = np.abs(grad_bp_sel - grad_fd)
    scale = np.maximum(np.abs(grad_fd), 1e-6)
    return {
        "n_checked": int(len(idx)),
        "nan_in_backprop": int(torch.isnan(grad_bp).sum().item()),
        "zero_grad_fraction": float((grad_bp.abs() < 1e-9).double().mean().item()),
        "fd_max_abs_err": float(abs_err.max()),
        "fd_median_rel_err": float(np.median(abs_err / scale)),
        "fd_frac_rel_err_below_0.05": float(np.mean(abs_err / scale < 0.05)),
        "grad_norm": float(grad_bp.norm().item()),
    }


def jitter_stability(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_rep, jitter_sd, seed):
    d_hat_t, init_inv_t, pipeline_t, cap_t, y_t = inst_t
    torch.manual_seed(seed)
    grads = []
    for _ in range(n_rep):
        d_var = (d_hat_t + jitter_sd * torch.randn_like(d_hat_t)).clamp_min(0.1).requires_grad_(True)
        loss, _ = layer_loss(layer, d_var, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver)
        loss.backward()
        grads.append(d_var.grad.detach().flatten())
    g = torch.stack(grads)
    mean_norm = g.mean(dim=0).norm().item()
    spread = g.std(dim=0).norm().item()
    return {"grad_rel_spread_under_jitter": float(spread / max(mean_norm, 1e-9))}


def timing(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_rep):
    d_hat_t, init_inv_t, pipeline_t, cap_t, y_t = inst_t
    times = []
    for _ in range(n_rep):
        d_var = d_hat_t.clone().requires_grad_(True)
        t0 = time.perf_counter()
        loss, _ = layer_loss(layer, d_var, init_inv_t, pipeline_t, cap_t, y_t, lead_time, hold_cost, short_cost, solver)
        loss.backward()
        times.append(time.perf_counter() - t0)
    return {"sec_per_forward_backward_median": float(np.median(times)), "sec_max": float(np.max(times))}


def run_config(n_sku, horizon, lead_time, hold_cost, short_cost, eps_reg, use_capacity, cap_ratio, solver, seed):
    layer = build_layer(n_sku, horizon, lead_time, hold_cost, short_cost, eps_reg, use_capacity)
    y, d_hat, init_inv, pipeline, cap = make_instance(n_sku, horizon, lead_time, seed, cap_ratio)
    inst_t = (
        torch.tensor(d_hat),
        torch.tensor(init_inv),
        torch.tensor(pipeline),
        torch.tensor(cap) if use_capacity else None,
        torch.tensor(y),
    )
    out = {
        "n_sku": n_sku, "horizon": horizon, "lead_time": lead_time,
        "eps_reg": eps_reg, "shared_capacity": use_capacity, "solver": solver,
    }
    out.update(finite_diff_check(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_entries=24, fd_eps=1e-2, seed=seed))
    out.update(jitter_stability(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_rep=8, jitter_sd=1e-3, seed=seed))
    out.update(timing(layer, inst_t, lead_time, hold_cost, short_cost, solver, n_rep=5))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="ECOS", choices=["ECOS", "SCS"])
    parser.add_argument("--hold_cost", type=float, default=1.0)
    parser.add_argument("--short_cost", type=float, default=5.0)
    parser.add_argument("--lead_time", type=int, default=2)
    parser.add_argument("--cap_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/spike_0a_results.json")
    args = parser.parse_args()

    configs = [
        dict(n_sku=1, horizon=14, use_capacity=False, eps_reg=0.0),
        dict(n_sku=1, horizon=14, use_capacity=False, eps_reg=1e-3),
        dict(n_sku=8, horizon=14, use_capacity=True, eps_reg=0.0),
        dict(n_sku=8, horizon=14, use_capacity=True, eps_reg=1e-3),
        dict(n_sku=8, horizon=28, use_capacity=True, eps_reg=1e-3),
        dict(n_sku=32, horizon=28, use_capacity=True, eps_reg=1e-3),
    ]

    results = []
    for cfg in configs:
        try:
            res = run_config(
                cfg["n_sku"], cfg["horizon"], args.lead_time, args.hold_cost, args.short_cost,
                cfg["eps_reg"], cfg["use_capacity"], args.cap_ratio, args.solver, args.seed,
            )
        except Exception as exc:
            res = {**cfg, "error": repr(exc)}
        results.append(res)
        print(json.dumps(res))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if "error" not in r]
    if ok:
        ref = next((r for r in ok if r["n_sku"] == 32), ok[-1])
        sec = ref["sec_per_forward_backward_median"]
        calls = (200 / ref["n_sku"]) * 12 * 30 * 40
        print(json.dumps({
            "extrapolation_basis": f"{ref['n_sku']}sku_x{ref['horizon']}d",
            "assumed_calls_200sku_12fold_30epoch_40batches": int(calls),
            "estimated_training_hours": round(sec * calls / 3600.0, 2),
        }))


if __name__ == "__main__":
    main()