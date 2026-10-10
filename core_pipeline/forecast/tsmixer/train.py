"""Training core for Colab (ST4-05). Torch phase; consumes staging
bundles only -- no database access anywhere in this import chain.

Stages (ST4-04 P0-02): "final" trains on origins with t + 28 <= train_end
using the final scalers; "tune" uses the tune scalers and economic
weights, splitting origins so training targets end at tune_train_end and
validation targets lie entirely inside (tune_train_end, train_end]
(D4 window: 1438..1605 on fold 1). Early stopping on the validation loss
exists for the tune stage; the tuning grid itself is fixed in the Stage 4
tuning protocol (agenda 3), not here.

Losses are computed in raw demand units (ST4-02): A1 MSE, A2 the
cost-weighted pinball with (h, b_r) from economic_params -- official
columns for the final stage, *_tune columns for the tune stage -- and A3
the 9-quantile pinball on pre-rearrangement outputs.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from core_pipeline.forecast.tsmixer.adapters import (
    PanelInputs,
    StaticInputs,
    WindowBatch,
    build_batch,
    training_origins,
)
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND,
    QUANTILES_9,
    TSMixerConfig,
    checkpoint_key,
    model_seed,
)
from core_pipeline.forecast.tsmixer.losses import (
    cost_weighted_pinball_raw,
    mse_loss_raw,
    pinball_grid_raw,
)
from core_pipeline.forecast.tsmixer.model import TSMixerExt
from core_pipeline.forecast.tsmixer.panel import (
    build_panel,
    build_static_inputs,
)
from core_pipeline.forecast.tsmixer import checkpoint as ckpt


@dataclass
class StageData:
    panel: PanelInputs
    static: StaticInputs
    train_rows: np.ndarray
    train_origins: np.ndarray
    valid_rows: np.ndarray | None
    valid_origins: np.ndarray | None
    scale: np.ndarray
    price_mean: np.ndarray
    h_cost: np.ndarray | None   # per series, stage-appropriate
    b_cost: np.ndarray | None


def prepare_stage(bundle: dict, stage: str, model_id: str,
                  r: int | None, cfg: TSMixerConfig) -> StageData:
    meta = bundle["fold_meta"]
    static_df = bundle["series_static"]
    econ = bundle["economic_params"].set_index("series_id").reindex(
        [str(s) for s in static_df["series_id"]])
    scalers = bundle["scalers"]
    train_end = int(meta["train_end_idx"])
    if stage == "final":
        scale = scalers["scale_final"]
        price_mean = scalers["price_mean_final"]
    elif stage == "tune":
        if "scale_tune" not in scalers:
            raise ValueError("tune stage requires the fold-1 bundle")
        scale = scalers["scale_tune"]
        price_mean = scalers["price_mean_tune"]
    else:
        raise ValueError(f"unknown stage {stage!r}")

    panel = build_panel(
        bundle["panel_train"], bundle["known_calendar"], static_df,
        bundle["encoder"], scale, price_mean,
        idx_start=1, idx_end=train_end)
    static = build_static_inputs(static_df, bundle["encoder"])

    rows_all, orig_all = training_origins(panel, train_end, cfg)
    if stage == "final":
        tr = (rows_all, orig_all)
        va = (None, None)
    else:
        tune_end = int(meta["tune_train_end_idx"])
        m_tr = orig_all + cfg.horizon <= tune_end
        m_va = orig_all >= tune_end  # targets t+1..t+28 in (tune_end, train_end]
        tr = (rows_all[m_tr], orig_all[m_tr])
        va = (rows_all[m_va], orig_all[m_va])

    h_cost = b_cost = None
    if checkpoint_key(model_id, r).startswith(("a2", "b")):
        h_col, b_col = (("h", f"b_r{r}") if stage == "final"
                        else ("h_tune", f"b_tune_r{r}"))
        h_cost = econ[h_col].to_numpy(dtype=np.float64)
        b_cost = econ[b_col].to_numpy(dtype=np.float64)
        if not (np.isfinite(h_cost).all() and np.isfinite(b_cost).all()):
            raise ValueError("non-finite economic parameters")
    return StageData(panel, static, tr[0], tr[1], va[0], va[1],
                     np.asarray(scale, dtype=np.float64),
                     np.asarray(price_mean, dtype=np.float64),
                     h_cost, b_cost)


def _tensors(batch: WindowBatch, device: str) -> dict:
    out = {
        "x_num": torch.from_numpy(batch.x_num),
        "x_event": torch.from_numpy(batch.x_event),
        "z_num": torch.from_numpy(batch.z_num),
        "z_event": torch.from_numpy(batch.z_event),
        "s_codes": torch.from_numpy(batch.s_codes),
    }
    return {k: v.to(device) for k, v in out.items()}


def _batch_loss(model: TSMixerExt, batch: WindowBatch, data: StageData,
                model_id: str, device: str) -> torch.Tensor:
    t = _tensors(batch, device)
    out = model(t["x_num"], t["x_event"], t["z_num"], t["z_event"],
                t["s_codes"])
    scale_t = torch.from_numpy(batch.scale.astype(np.float64)).to(device)
    y_raw = (torch.from_numpy(batch.y_scaled.astype(np.float64)).to(device)
             * scale_t[:, None])
    pred_raw = out.to_raw(scale_t)
    if model_id == "a1":
        return mse_loss_raw(pred_raw, y_raw)
    if model_id == "a3":
        return pinball_grid_raw(pred_raw, y_raw)
    h = torch.from_numpy(data.h_cost[batch.series_rows]).to(device)
    b = torch.from_numpy(data.b_cost[batch.series_rows]).to(device)
    return cost_weighted_pinball_raw(pred_raw, y_raw, h, b)


def _epoch(model, opt, data: StageData, rows, origins, model_id, cfg,
           batch_size, device, rng, train: bool) -> float:
    order = rng.permutation(rows.shape[0]) if train else np.arange(rows.shape[0])
    total, n = 0.0, 0
    model.train(mode=train)
    for lo in range(0, order.shape[0], batch_size):
        idx = order[lo:lo + batch_size]
        batch = build_batch(data.panel, data.static, rows[idx], origins[idx],
                            cfg, with_targets=True)
        if train:
            opt.zero_grad()
            loss = _batch_loss(model, batch, data, model_id, device)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                loss = _batch_loss(model, batch, data, model_id, device)
        total += float(loss.detach()) * idx.shape[0]
        n += idx.shape[0]
    return total / max(n, 1)


def _probe(model, data: StageData, cfg, device: str, k: int = 8):
    order = np.lexsort((data.train_origins, data.train_rows))[:k]
    batch = build_batch(data.panel, data.static, data.train_rows[order],
                        data.train_origins[order], cfg)
    t = _tensors(batch, device)
    model.eval()
    with torch.no_grad():
        out = model(t["x_num"], t["x_event"], t["z_num"], t["z_event"],
                    t["s_codes"]).values_scaled
    return {k2: v.cpu() for k2, v in t.items()}, out.cpu()


def train_one(bundle: dict, model_id: str, fold: int, r: int | None,
              stage: str, cfg: TSMixerConfig, out_dir, epochs: int,
              batch_size: int, lr: float, global_seed: int, device: str,
              patience: int = 5, git_commit: str = "unknown",
              measure_only: bool = False) -> dict:
    key = checkpoint_key(model_id, r)
    seed = model_seed(global_seed, model_id, fold, r)
    det = ckpt.set_training_determinism(seed)
    rng = np.random.default_rng(seed)
    data = prepare_stage(bundle, stage, model_id, r, cfg)

    model = TSMixerExt(cfg, data.static.cardinalities,
                       int(bundle["encoder"]["cardinalities"]["event_type"]),
                       prediction_kind=MODEL_KIND[model_id]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    history, best_valid, best_epoch, bad = [], float("inf"), 0, 0
    best_state = None
    run_epochs = 2 if measure_only else epochs
    for epoch in range(1, run_epochs + 1):
        t0 = time.time()
        tr_loss = _epoch(model, opt, data, data.train_rows,
                         data.train_origins, model_id, cfg, batch_size,
                         device, rng, train=True)
        line = f"epoch {epoch} train {tr_loss:.4f}"
        if stage == "tune":
            va_loss = _epoch(model, opt, data, data.valid_rows,
                             data.valid_origins, model_id, cfg, batch_size,
                             device, rng, train=False)
            line += f" valid {va_loss:.4f}"
            if va_loss < best_valid:
                best_valid, best_epoch, bad = va_loss, epoch, 0
                best_state = {k2: v.detach().cpu().clone()
                              for k2, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(time.time() - t0)
        print(line + f" {history[-1]:.1f}s")
        if stage == "tune" and bad >= patience and not measure_only:
            break

    peak_mb = (torch.cuda.max_memory_allocated() / 2**20
               if device.startswith("cuda") else 0.0)
    sec_per_epoch = float(np.mean(history))
    if measure_only:
        est_one = sec_per_epoch * epochs
        report = {
            "sec_per_epoch": round(sec_per_epoch, 1),
            "peak_vram_mb": round(peak_mb, 1),
            "train_samples": int(data.train_rows.shape[0]),
            "est_minutes_per_checkpoint": round(est_one / 60, 1),
            "est_minutes_16_checkpoints": round(16 * est_one / 60, 1),
        }
        print(f"measure: {report}")
        return report

    if stage == "tune" and best_state is not None:
        model.load_state_dict(best_state)
        selection = {"selection_metric": "valid_loss_early_stop",
                     "best_epoch": best_epoch,
                     "best_valid_loss": best_valid}
    else:
        selection = {"selection_metric": "fixed_epochs",
                     "best_epoch": run_epochs}

    probe_in, probe_out = _probe(model, data, cfg, device)
    bundle_files = bundle["manifest"]["files"]
    pt_path, mf_path = ckpt.save_checkpoint(
        out_dir, model, model_id, fold, r,
        scale=data.scale, price_mean=data.price_mean,
        probe_input=probe_in, probe_output=probe_out,
        static_cardinalities=data.static.cardinalities,
        event_cardinality=int(
            bundle["encoder"]["cardinalities"]["event_type"]),
        train_meta={"seed": seed, "epochs": run_epochs, "stage": stage,
                    "batch_size": batch_size, "lr": lr, **selection, **det},
        data_meta={
            "train_end": int(bundle["fold_meta"]["train_end_idx"]),
            "bundle_sha256": bundle_files["fold_meta.json"]["sha256"],
            "scale_sha256": bundle_files["scalers.npz"]["sha256"],
            "encoder_hash": bundle["encoder"]["hash"],
        },
        env_meta={**ckpt.environment_meta(device), "git_commit": git_commit},
    )
    print(f"saved {pt_path.name} sec/epoch {sec_per_epoch:.1f} "
          f"vram_mb {peak_mb:.0f}")
    return {"checkpoint": str(pt_path), "manifest": str(mf_path)}


def _valid_wql9(model: TSMixerExt, data: StageData, cfg: TSMixerConfig,
                batch_size: int, device: str) -> float:
    """A3 selection metric (ST4-07 T1): computed exactly as inference
    serves the quantiles -- raw units, clamped at zero, monotone
    rearranged. Formula: 2 * sum of pinball over all (sample, horizon,
    quantile) / (9 * sum |y|) over the fixed validation origins. The
    denominator is trial-invariant, so the argmin matches any
    standard-normalized WQL on the same set; the official metrics-module
    value is recomputed locally for the selected trial (cross-check)."""
    total_pin, total_abs = 0.0, 0.0
    tau = np.asarray(QUANTILES_9, dtype=np.float64).reshape(1, 1, -1)
    model.eval()
    for lo in range(0, data.valid_rows.shape[0], batch_size):
        batch = build_batch(data.panel, data.static,
                           data.valid_rows[lo:lo + batch_size],
                           data.valid_origins[lo:lo + batch_size],
                           cfg, with_targets=True)
        t = _tensors(batch, device)
        with torch.no_grad():
            out = model(t["x_num"], t["x_event"], t["z_num"], t["z_event"],
                        t["s_codes"])
            raw = out.to_raw(torch.from_numpy(
                batch.scale.astype(np.float64)).to(device)).cpu().numpy()
        q = np.sort(np.maximum(raw, 0.0), axis=-1)
        y = (batch.y_scaled.astype(np.float64)
             * batch.scale.astype(np.float64)[:, None])
        diff = y[:, :, None] - q
        total_pin += float(np.maximum(tau * diff, (tau - 1.0) * diff).sum())
        total_abs += float(np.abs(y).sum())
    return 2.0 * total_pin / (len(QUANTILES_9) * max(total_abs, 1e-12))


def tune_run(bundle: dict, model_id: str, r: int | None,
             cfg: TSMixerConfig, lr: float, seed: int, device: str,
             epochs: int = 40, batch_size: int = 1024, patience: int = 5,
             valid_metric: str = "loss") -> dict:
    """One tuning-stage run (no checkpoint written). valid_metric:
    "loss" = the model's own objective on the validation origins (A1 MSE,
    A2 cost pinball, A3 quantile pinball); "wql9" = the rearranged WQL
    path above (A3 selection, ST4-07 T1/T3). Early stopping and best_epoch
    both follow the chosen metric; per-epoch values, max-epoch hits and
    the tail trend are returned for the verdict's reporting duties."""
    det = ckpt.set_training_determinism(seed)
    rng = np.random.default_rng(seed)
    data = prepare_stage(bundle, "tune", model_id, r, cfg)
    model = TSMixerExt(cfg, data.static.cardinalities,
                       int(bundle["encoder"]["cardinalities"]["event_type"]),
                       prediction_kind=MODEL_KIND[model_id]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    metrics, times = [], []
    best, best_epoch, bad = float("inf"), 0, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        _epoch(model, opt, data, data.train_rows, data.train_origins,
               model_id, cfg, batch_size, device, rng, train=True)
        if valid_metric == "wql9":
            m = _valid_wql9(model, data, cfg, batch_size, device)
        else:
            m = _epoch(model, opt, data, data.valid_rows,
                       data.valid_origins, model_id, cfg, batch_size,
                       device, rng, train=False)
        metrics.append(float(m))
        times.append(time.time() - t0)
        if m < best:
            best, best_epoch, bad = float(m), epoch, 0
        else:
            bad += 1
        if bad >= patience:
            break
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20
               if device.startswith("cuda") else 0.0)
    tail = (metrics[-1] - metrics[-4]) / abs(metrics[-4]) \
        if len(metrics) >= 4 else None
    counts = model.param_counts()
    return {
        "best_epoch": best_epoch,
        "best_metric": best,
        "epochs_ran": len(metrics),
        "hit_max_epochs": len(metrics) == epochs and bad < patience,
        "tail_rel_trend_last3": tail,
        "epoch_metrics": metrics,
        "sec_total": round(time.time() - t_start, 1),
        "sec_per_epoch": round(float(np.mean(times)), 2),
        "peak_vram_mb": round(peak_mb, 1),
        "params": counts["backbone"] + counts["head"],
        "n_train": int(data.train_rows.shape[0]),
        "n_valid": int(data.valid_rows.shape[0]),
        "valid_metric": valid_metric,
        "precision": {k: det[k] for k in
                      ("precision_policy", "tf32_matmul", "tf32_cudnn")},
    }
