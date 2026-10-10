"""TSMixer-Ext backbone and heads (ST4-01 4-A, 4-B).

Follows the extended architecture (align -> mixing with static
conditioning), not a flattened MLP over raw inputs:
  - align: past channels are temporally projected L -> H per channel, then
    past/future features are projected to a shared width d_feat; the static
    categorical embeddings are projected to d_static;
  - mixing: K blocks of temporal mixing (linear over the H axis) and
    feature mixing conditioned on the static vector, with residuals and
    layer norm;
  - heads: point (Linear d_model -> 1, softplus) for A1/A2/B, quantile
    (Linear d_model -> 9, free sign) for A3.

Backbone sharing means identical architecture and settings across
independently trained models, not a jointly updated instance. Controlled:
information set, past_length, d_feat/d_static/d_model/ff_dim/n_blocks/
dropout, sampling, retrain cutoffs, and the seed protocol in
config.model_seed. A3 differs from the point models only in the head;
param_counts() reports backbone and head sizes separately.

Output semantics (ST4-01 P0): the forward pass returns a ForecastOutput
whose kind is fixed at construction. Point outputs are named yhat and are
never a q0.50 claim; only A3 carries a quantile grid. Non-negativity: point
heads use softplus (shared by A1/A2/B; differentiable for the Stage 5
cvxpylayers path; exact zeros are approached asymptotically, noted as a
caveat for zero-heavy series). The A3 head is unconstrained in training;
non-negativity and monotonicity are applied at inference in postproc.
"""

from dataclasses import dataclass

import torch
from torch import nn

from core_pipeline.forecast.tsmixer.config import (
    FUTURE_NUM_CHANNELS,
    PAST_NUM_CHANNELS,
    POINT_KINDS,
    QUANTILE_KIND,
    QUANTILES_9,
    TSMixerConfig,
)


@dataclass
class ForecastOutput:
    kind: str
    values_scaled: torch.Tensor  # point: (B, H); quantile: (B, H, 9)

    def to_raw(self, scale: torch.Tensor) -> torch.Tensor:
        """scale (B,) s_i > 0; returns raw demand units."""
        if self.values_scaled.dim() == 2:
            return self.values_scaled * scale[:, None]
        return self.values_scaled * scale[:, None, None]


class ConditionalMixingBlock(nn.Module):
    def __init__(self, cfg: TSMixerConfig):
        super().__init__()
        self.time_norm = nn.LayerNorm(cfg.d_model)
        self.time_lin = nn.Linear(cfg.horizon, cfg.horizon)
        self.feat_norm = nn.LayerNorm(cfg.d_model)
        self.feat_lin1 = nn.Linear(cfg.d_model + cfg.d_static, cfg.ff_dim)
        self.feat_lin2 = nn.Linear(cfg.ff_dim, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = self.time_norm(x).transpose(1, 2)
        y = torch.relu(self.time_lin(y)).transpose(1, 2)
        x = x + self.drop(y)
        y = self.feat_norm(x)
        s_b = s.unsqueeze(1).expand(-1, y.shape[1], -1)
        y = torch.relu(self.feat_lin1(torch.cat([y, s_b], dim=-1)))
        y = self.feat_lin2(self.drop(y))
        return x + self.drop(y)


class TSMixerExt(nn.Module):
    def __init__(
        self,
        cfg: TSMixerConfig,
        static_cardinalities: tuple[int, ...],
        event_cardinality: int,
        prediction_kind: str,
    ):
        super().__init__()
        if prediction_kind in POINT_KINDS:
            self.head_type = "point"
        elif prediction_kind == QUANTILE_KIND:
            self.head_type = "quantile"
        else:
            raise ValueError(f"unknown prediction_kind {prediction_kind!r}")
        self.cfg = cfg
        self.prediction_kind = prediction_kind

        self.event_emb = nn.Embedding(event_cardinality, cfg.event_emb_dim)
        self.static_embs = nn.ModuleList(
            nn.Embedding(card, cfg.static_emb_dim(card)) for card in static_cardinalities
        )
        static_width = sum(cfg.static_emb_dim(c) for c in static_cardinalities)

        c_past = len(PAST_NUM_CHANNELS) + cfg.event_emb_dim
        c_fut = len(FUTURE_NUM_CHANNELS) + cfg.event_emb_dim
        self.temporal_proj = nn.Linear(cfg.past_length, cfg.horizon)
        self.past_feat = nn.Linear(c_past, cfg.d_feat)
        self.fut_feat = nn.Linear(c_fut, cfg.d_feat)
        self.static_proj = nn.Linear(static_width, cfg.d_static)
        self.merge = nn.Linear(2 * cfg.d_feat, cfg.d_model)
        self.blocks = nn.ModuleList(ConditionalMixingBlock(cfg) for _ in range(cfg.n_blocks))

        out_dim = 1 if self.head_type == "point" else len(QUANTILES_9)
        self.head = nn.Linear(cfg.d_model, out_dim)

    def _backbone(
        self,
        x_num: torch.Tensor,    # (B, L, C_past_num)
        x_event: torch.Tensor,  # (B, L)
        z_num: torch.Tensor,    # (B, H, C_fut_num)
        z_event: torch.Tensor,  # (B, H)
        s_codes: torch.Tensor,  # (B, n_cats)
    ) -> torch.Tensor:
        x = torch.cat([x_num, self.event_emb(x_event)], dim=-1)
        x = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, H, C_past)
        x = self.past_feat(x)
        z = self.fut_feat(torch.cat([z_num, self.event_emb(z_event)], dim=-1))
        s = self.static_proj(
            torch.cat([emb(s_codes[:, k]) for k, emb in enumerate(self.static_embs)], dim=-1)
        )
        h = self.merge(torch.cat([x, z], dim=-1))
        for block in self.blocks:
            h = block(h, s)
        return h  # (B, H, d_model)

    def forward(self, x_num, x_event, z_num, z_event, s_codes) -> ForecastOutput:
        h = self._backbone(x_num, x_event, z_num, z_event, s_codes)
        out = self.head(h)
        if self.head_type == "point":
            return ForecastOutput(self.prediction_kind, nn.functional.softplus(out).squeeze(-1))
        return ForecastOutput(self.prediction_kind, out)

    def param_counts(self) -> dict[str, int]:
        head_n = sum(p.numel() for p in self.head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": total - head_n, "head": head_n}

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("head.")}
