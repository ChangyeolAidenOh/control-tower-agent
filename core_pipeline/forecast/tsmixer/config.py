"""TSMixer-Ext contract configuration (ST4-01).

Two-layer contract (ST3-01): the information-set layer (DatasetConfig /
DemandHistory / mart.known_future_calendar) stays model-common; this module
describes only the TSMixer-side adapters, backbone and heads. A1, A2 and B
share this config and architecture; A3 differs only in the output head.

Width/depth defaults below are pre-tuning placeholders. The tuning grid is
fixed in the Stage 4 tuning protocol (pre-fold-1 split, one pass) and the
chosen values are recorded in the manifest; past_length and horizon are not
tunable (ST4-01 D3).
"""

from dataclasses import dataclass

PAST_LENGTH = 56   # window [t - 55, t], origin t inclusive (ST4-01 D3)
HORIZON = 28

# A3 quantile grid, preregistration section 4 (frozen). Distinct from the
# 11-quantile LightGBM grid in forecast.common.QUANTILES.
QUANTILES_9: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# ST4-01 P0: single-output predictions are stored as `yhat` with an explicit
# prediction_kind; they are never written into quantile columns. Only A3's
# q0.50 is a median estimate.
POINT_KINDS: tuple[str, ...] = (
    "mse_point",               # A1
    "cost_weighted_point",     # A2
    "decision_focused_point",  # B (Stage 5)
)
QUANTILE_KIND = "quantile_grid"  # A3
PREDICTION_KINDS: tuple[str, ...] = POINT_KINDS + (QUANTILE_KIND,)

MODEL_IDS: tuple[str, ...] = ("a1", "a2", "a3", "b")
MODEL_KIND: dict[str, str] = {
    "a1": "mse_point",
    "a2": "cost_weighted_point",
    "a3": QUANTILE_KIND,
    "b": "decision_focused_point",
}

# Past numeric channels, fixed order. Adapters assemble exactly this order;
# the model derives its input width from this tuple. price_rel is
# price_ffill divided by the per-series train-window mean price (no
# backfill upstream, so pre-listing positions never enter a valid window:
# origin validity requires 56 days since first_active, and activity implies
# listing).
PAST_NUM_CHANNELS: tuple[str, ...] = (
    "sales_scaled",
    "price_rel",
    "is_active",
    "snap",
    "is_event",
    "month_sin",
    "month_cos",
    "wday_1", "wday_2", "wday_3", "wday_4", "wday_5", "wday_6", "wday_7",
)

# Future numeric channels: published calendar only (ST4-01 c). No price.
FUTURE_NUM_CHANNELS: tuple[str, ...] = (
    "snap",
    "is_event",
    "month_sin",
    "month_cos",
    "wday_1", "wday_2", "wday_3", "wday_4", "wday_5", "wday_6", "wday_7",
)


@dataclass(frozen=True)
class TSMixerConfig:
    past_length: int = PAST_LENGTH
    horizon: int = HORIZON
    d_feat: int = 32          # align-stage shared feature width
    d_static: int = 16        # static conditioning width
    d_model: int = 64         # mixing width
    ff_dim: int = 64          # feature-mixing hidden width
    n_blocks: int = 2
    dropout: float = 0.1
    event_emb_dim: int = 4    # event_type embedding, shared past/future
    max_static_emb_dim: int = 16

    def static_emb_dim(self, cardinality: int) -> int:
        return min(self.max_static_emb_dim, (cardinality + 1) // 2)


# Static categoricals consumed by the model: LightGBM parity set minus
# tgt_event_type (events enter via the shared embedding channel) and
# without item_id (not in the LightGBM information set either).
STATIC_CATEGORICALS: tuple[str, ...] = (
    "series_id", "store_id", "dept_id", "cat_id", "state_id",
)

ADAPTER_VERSION = "st4_adapters_v1"

# A2 (and Stage 5 B) train one checkpoint per cost ratio r because b_i =
# r * h_i moves the target quantile of the cost-weighted loss (ST4-04
# P0-01). A1/A3 are r-agnostic and reused across evaluation ratios;
# evaluation at r=10 reuses the r=5 checkpoint.
R_TRAINED: tuple[int, ...] = (3, 5)
R_MODELS: tuple[str, ...] = ("a2", "b")


def checkpoint_key(model_id: str, r: int | None = None) -> str:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model_id {model_id!r}")
    if model_id in R_MODELS:
        if r not in R_TRAINED:
            raise ValueError(f"{model_id} requires r in {R_TRAINED}, got {r!r}")
        return f"{model_id}_r{r}"
    if r is not None:
        raise ValueError(f"{model_id} is r-agnostic; got r={r!r}")
    return model_id


def model_seed(global_seed: int, model_id: str, fold: int,
               r: int | None = None) -> int:
    """Deterministic initialization protocol (ST4-01/04 parity control).

    Independent training runs per checkpoint; identical backbone config,
    seed varies only through (global_seed, model_id, fold, r). r is
    mandatory for A2/B and forbidden otherwise (checkpoint_key)."""
    checkpoint_key(model_id, r)  # validates the pairing
    return (global_seed * 10000 + MODEL_IDS.index(model_id) * 1000
            + (r or 0) * 25 + fold)


RETRAIN_FOLDS: tuple[int, ...] = (1, 4, 7, 10)
N_FOLDS = 12


def checkpoint_fold_for(fold: int) -> int:
    """Inference fold -> retrain fold whose checkpoint serves it (ST4-04:
    2,3 -> 1; 5,6 -> 4; 8,9 -> 7; 11,12 -> 10). Enforced at load time."""
    if not 1 <= fold <= N_FOLDS:
        raise ValueError(f"fold out of range: {fold}")
    return max(f for f in RETRAIN_FOLDS if f <= fold)


def input_schema_hash() -> str:
    """12-hex digest of the adapter input contract; recorded in staging
    bundles and checkpoint manifests, compared on load (hard fail)."""
    import hashlib
    import json
    payload = json.dumps([
        PAST_LENGTH, HORIZON, list(PAST_NUM_CHANNELS),
        list(FUTURE_NUM_CHANNELS), list(STATIC_CATEGORICALS),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
