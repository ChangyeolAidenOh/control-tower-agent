"""Checkpoint contract (ST4-04 section 5). Torch phase only.

A checkpoint is a .pt payload (state_dict + scale/price_mean tensors + a
deterministic probe batch with its outputs) plus a sidecar manifest json
carrying the agreed field table. Loading is hard-fail on every pin:
schema version, checkpoint key, fold, prediction_kind, input_schema_hash
(against the local adapter code), encoder_hash, state_dict sha, and the
probe comparison -- the probe recomputes the stored batch on the local
device and enforces a numeric tolerance, which is the concrete GPU -> CPU
reproduction check. Mismatch raises; never warn-and-continue.

Reproducibility claim is deliberately limited: CPU reload fidelity is
verified here; same-weights GPU retraining is NOT claimed (CUDA kernels,
nondeterministic ops, library versions). TF32 is explicitly disabled at
training time and the flags are recorded.
"""

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from core_pipeline.forecast.tsmixer.config import (
    ADAPTER_VERSION,
    MODEL_KIND,
    TSMixerConfig,
    checkpoint_key,
    input_schema_hash,
)
from core_pipeline.forecast.tsmixer.model import TSMixerExt

CKPT_SCHEMA_VERSION = "tsmixer_ckpt_v1"
PROBE_ATOL = 1e-4
PROBE_RTOL = 1e-4


def set_training_determinism(seed: int) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        # ST4-07 verdict: record the real precision structure -- fp32
        # parameters/forward, fp64 loss computation (raw-unit path), no AMP.
        "model_parameter_dtype": "float32",
        "forward_dtype": "float32",
        "loss_compute_dtype": "float64",
        "precision_policy": "fp32_model_fp64_loss",
        "amp_enabled": False,
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul,
                                    "allow_tf32", False)),
        "tf32_cudnn": bool(getattr(torch.backends.cudnn,
                                   "allow_tf32", False)),
    }


def state_dict_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(sd):
        h.update(name.encode())
        h.update(sd[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def environment_meta(device: str) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device": device,
    }


def save_checkpoint(
    out_dir: Path, model: TSMixerExt, model_id: str, fold: int,
    r: int | None, scale: np.ndarray, price_mean: np.ndarray,
    probe_input: dict, probe_output: torch.Tensor,
    static_cardinalities: tuple[int, ...], event_cardinality: int,
    train_meta: dict, data_meta: dict, env_meta: dict,
) -> tuple[Path, Path]:
    key = checkpoint_key(model_id, r)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"tsmixer_{key}_f{fold}.pt"
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in
                       model.state_dict().items()},
        "scale": torch.from_numpy(np.asarray(scale, dtype=np.float64)),
        "price_mean": torch.from_numpy(
            np.asarray(price_mean, dtype=np.float64)),
        "probe_input": {k: v.detach().cpu() for k, v in probe_input.items()},
        "probe_output": probe_output.detach().cpu(),
    }
    torch.save(payload, pt_path)
    manifest = {
        "manifest_schema_version": CKPT_SCHEMA_VERSION,
        "checkpoint_key": key,
        "model_id": model_id,
        "fold": int(fold),
        "r": r,
        "prediction_kind": MODEL_KIND[model_id],
        "tsmixer_config": asdict(model.cfg),
        "static_cardinalities": list(static_cardinalities),
        "event_cardinality": int(event_cardinality),
        "adapter_version": ADAPTER_VERSION,
        "input_schema_hash": input_schema_hash(),
        "state_dict_sha256": state_dict_sha(payload["state_dict"]),
        "checkpoint_file": pt_path.name,
        **data_meta,   # train_end, bundle_sha256, scale_sha256, encoder_hash
        **train_meta,  # seed, epochs, best_epoch, selection_metric, tf32...
        **env_meta,    # python, torch, cuda, device, dtype, git_commit
    }
    mf_path = out_dir / f"tsmixer_{key}_f{fold}.json"
    mf_path.write_text(json.dumps(manifest, indent=2))
    return pt_path, mf_path


def load_checkpoint(
    out_dir: Path, model_id: str, retrain_fold: int, r: int | None,
    expected_encoder_hash: str, device: str = "cpu",
) -> tuple[TSMixerExt, dict, dict]:
    """Returns (model.eval(), payload-aux, manifest). Every pin hard-fails."""
    key = checkpoint_key(model_id, r)
    out_dir = Path(out_dir)
    mf_path = out_dir / f"tsmixer_{key}_f{retrain_fold}.json"
    manifest = json.loads(mf_path.read_text())
    if manifest["manifest_schema_version"] != CKPT_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest schema mismatch")
    if (manifest["checkpoint_key"], int(manifest["fold"])) != (key, retrain_fold):
        raise ValueError(
            f"checkpoint identity mismatch: manifest "
            f"{manifest['checkpoint_key']}/f{manifest['fold']} "
            f"!= requested {key}/f{retrain_fold}")
    if manifest["input_schema_hash"] != input_schema_hash():
        raise ValueError("input_schema_hash mismatch with local adapter code")
    if manifest["encoder_hash"] != expected_encoder_hash:
        raise ValueError("encoder_hash mismatch")

    payload = torch.load(out_dir / manifest["checkpoint_file"],
                         weights_only=True, map_location="cpu")
    if state_dict_sha(payload["state_dict"]) != manifest["state_dict_sha256"]:
        raise ValueError("state_dict sha256 mismatch")
    cfg = TSMixerConfig(**manifest["tsmixer_config"])
    model = TSMixerExt(cfg, tuple(manifest["static_cardinalities"]),
                       int(manifest["event_cardinality"]),
                       prediction_kind=manifest["prediction_kind"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    probe_in = {k: v.to(device) for k, v in payload["probe_input"].items()}
    with torch.no_grad():
        local = model(probe_in["x_num"], probe_in["x_event"],
                      probe_in["z_num"], probe_in["z_event"],
                      probe_in["s_codes"]).values_scaled.cpu()
    ref = payload["probe_output"]
    if not torch.allclose(local, ref, atol=PROBE_ATOL, rtol=PROBE_RTOL):
        diff = float((local - ref).abs().max())
        raise ValueError(
            f"probe tolerance exceeded (max abs diff {diff:.3e}, "
            f"atol {PROBE_ATOL}, rtol {PROBE_RTOL})")
    aux = {
        "scale": payload["scale"].numpy(),
        "price_mean": payload["price_mean"].numpy(),
        "probe_max_abs_diff": float((local - ref).abs().max()),
    }
    return model, aux, manifest
