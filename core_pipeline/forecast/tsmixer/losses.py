"""Training losses, all computed in raw demand units (ST4-01 D1).

Callers inverse-transform model outputs with ForecastOutput.to_raw(scale)
before any loss here; targets are raw sales. Raw-unit MSE is dominated by
high-volume SKUs by construction -- that is a property of A1's objective
and is reported, not hidden.
"""

import torch

from core_pipeline.forecast.tsmixer.config import QUANTILES_9


def mse_loss_raw(yhat_raw: torch.Tensor, d_raw: torch.Tensor) -> torch.Tensor:
    """A1. yhat_raw, d_raw (B, H)."""
    return torch.mean((yhat_raw - d_raw) ** 2)


def cost_weighted_pinball_raw(
    yhat_raw: torch.Tensor, d_raw: torch.Tensor,
    h_cost: torch.Tensor, b_cost: torch.Tensor,
) -> torch.Tensor:
    """A2. Single-period newsvendor cost h_i (q - d)+ + b_i (d - q)+,
    equivalently the (h_i + b_i)-weighted pinball at tau_i = b_i / (h_i + b_i)
    (preregistration section 4). h_cost, b_cost (B,) joined by series_id;
    costs enter the loss only, never the inputs (ST4-01 d)."""
    over = torch.clamp(yhat_raw - d_raw, min=0.0)
    under = torch.clamp(d_raw - yhat_raw, min=0.0)
    return torch.mean(h_cost[:, None] * over + b_cost[:, None] * under)


def pinball_grid_raw(q_raw: torch.Tensor, d_raw: torch.Tensor) -> torch.Tensor:
    """A3. q_raw (B, H, 9) pre-rearrangement outputs (ST4-01 D2: the loss is
    applied to the free per-quantile outputs; sorting is inference-only).
    d_raw (B, H)."""
    tau = torch.as_tensor(QUANTILES_9, dtype=q_raw.dtype, device=q_raw.device).view(1, 1, -1)
    diff = d_raw.unsqueeze(-1) - q_raw
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))
