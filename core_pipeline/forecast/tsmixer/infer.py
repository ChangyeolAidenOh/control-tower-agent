"""Local inference core (ST4-06). Torch phase; pure functions over a
built panel so the runner stays thin and the leakage/mapping guarantees
are testable without a database.

Origins come from FoldDatasetBuilder.iter_test_origins(fold) upstream --
the same enumeration LightGBM and Chronos use -- so cross-model frames
align by construction. Past windows read only periods <= origin through
the adapter gather; future windows read published-calendar channels only.

The 56-active-day history rule (ST4-01 D3) is enforced here as a hard
fail listing the violating (series, origin) pairs: silent dropping would
desynchronize the prediction frames across models.
"""

import numpy as np
import torch

from core_pipeline.forecast.tsmixer.adapters import (
    PanelInputs,
    StaticInputs,
    build_batch,
)
from core_pipeline.forecast.tsmixer.config import TSMixerConfig
from core_pipeline.forecast.tsmixer.model import TSMixerExt


def inference_pairs(panel: PanelInputs, origins: list[int],
                    cfg: TSMixerConfig) -> tuple[np.ndarray, np.ndarray]:
    """All series x origins, validated against the history rule."""
    origins_arr = np.asarray(sorted(origins), dtype=np.int64)
    rows = np.repeat(np.arange(panel.n_series, dtype=np.int64),
                     origins_arr.shape[0])
    orgs = np.tile(origins_arr, panel.n_series)
    ok = orgs - panel.first_active_idx[rows] + 1 >= cfg.past_length
    if not ok.all():
        bad = [(str(panel.series_ids[rows[i]]), int(orgs[i]))
               for i in np.flatnonzero(~ok)[:10]]
        raise ValueError(
            f"{int((~ok).sum())} (series, origin) pairs lack "
            f"{cfg.past_length} active-history days; first: {bad}")
    return rows, orgs


def predict(model: TSMixerExt, panel: PanelInputs, static: StaticInputs,
            rows: np.ndarray, origins: np.ndarray, cfg: TSMixerConfig,
            batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    """Raw-unit predictions: (N, 28) for point models, (N, 28, 9) for the
    quantile model (pre-rearrangement -- postproc is the caller's job)."""
    model.eval()
    outs = []
    for lo in range(0, rows.shape[0], batch_size):
        batch = build_batch(panel, static, rows[lo:lo + batch_size],
                            origins[lo:lo + batch_size], cfg)
        tensors = {
            "x_num": torch.from_numpy(batch.x_num),
            "x_event": torch.from_numpy(batch.x_event),
            "z_num": torch.from_numpy(batch.z_num),
            "z_event": torch.from_numpy(batch.z_event),
            "s_codes": torch.from_numpy(batch.s_codes),
        }
        tensors = {k: v.to(device) for k, v in tensors.items()}
        with torch.no_grad():
            out = model(tensors["x_num"], tensors["x_event"],
                        tensors["z_num"], tensors["z_event"],
                        tensors["s_codes"])
            raw = out.to_raw(torch.from_numpy(
                batch.scale.astype(np.float64)).to(device))
        outs.append(raw.cpu().numpy())
    return np.concatenate(outs, axis=0)


def long_arrays(panel: PanelInputs, rows: np.ndarray, origins: np.ndarray,
                cfg: TSMixerConfig):
    """(series_id, origin_idx, horizon) long-form keys matching a stacked
    (N, 28[, Q]) prediction block flattened over horizons."""
    h = np.arange(1, cfg.horizon + 1, dtype=np.int64)
    series_long = np.repeat(panel.series_ids[rows], cfg.horizon)
    origin_long = np.repeat(origins, cfg.horizon)
    horizon_long = np.tile(h, rows.shape[0])
    return series_long, origin_long, horizon_long
