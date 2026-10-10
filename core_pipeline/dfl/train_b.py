"""Stage 5 (ST5-03..06): B (DFL) training core.

Loss: 28-day open-loop decision regret surrogate -- the TSMixer point
head's 28-day forecast enters the shared decision-time window
(lp_window: zero day 0, cached A4 state), the compiled bundle LP layer
returns the plan, downstream_cost prices it on the realized path, and
the precomputed oracle cost is subtracted (theta-independent constant).
Sample = (bundle, origin) from the frozen schedule; minibatch loss is
the sample mean (ST5-03 (3)). fp32 model, fp64 LP/loss (promotion at
to_raw, same as train._batch_loss). Deterministic: model seed via
model_seed(global, "b", fold, r); epoch order via rng(20261010 + epoch)
over the fixed origin set (ST5-05 (3)).

Integrity gates at prep time: staging input-schema hash, schedule sha ==
state-manifest sha == oracle-meta sha, oracle per-sample invalidation
keys recomputed from (state, realized path, h, b, K_g), series order of
each state file matched against series_static, and leakage bounds
(train o+28 <= cutoff, valid o+28 <= train_end).
"""

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from core_pipeline.dfl.lp_layer import (
    SOLVER_ARGS,
    LpLayerCache,
    downstream_cost,
)
from core_pipeline.dfl.lp_window import LP_WINDOW_DAYS, to_lp_window
from core_pipeline.forecast.tsmixer.adapters import build_batch
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND,
    TSMixerConfig,
    input_schema_hash,
    model_seed,
)
from core_pipeline.forecast.tsmixer.model import TSMixerExt
from core_pipeline.forecast.tsmixer.staging import load_bundle
from core_pipeline.forecast.tsmixer.train import prepare_stage
from core_pipeline.forecast.tsmixer import checkpoint as ckpt

K_G_COLUMN = "k_g_kappa_1_0"
EPOCH_ORDER_SEED = 20261010
# softplus(-10) in scaled space: outputs below this are in the dead-unit
# regime identified by spike 4-B (equivalent metering, no model change).
SATURATION_THRESHOLD = float(np.log1p(np.exp(-10.0)))


def wide_sales(panel, series_ids):
    sub = panel[panel["series_id"].isin(series_ids)]
    wide = (sub.pivot(index="period_idx", columns="series_id", values="sales")
            .reindex(columns=series_ids).sort_index())
    return wide.to_numpy(dtype=np.float64)


def sample_key(inv0, pipe0, future, h, b, k_g) -> str:
    # Must match scripts/build_b_oracle_cache.sample_key byte for byte.
    m = hashlib.sha256()
    for arr in (inv0, pipe0, future, h, b):
        m.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    m.update(struct.pack("<d", float(k_g)))
    return m.hexdigest()[:16]


@dataclass
class BundleGroup:
    gid: str
    rows: np.ndarray                 # panel/static row indices, bundle order
    k_g: float
    h: np.ndarray
    b: np.ndarray
    h_t: torch.Tensor
    b_t: torch.Tensor
    inv: np.ndarray                  # (n_states, n_sku)
    pipe: np.ndarray                 # (n_states, n_sku, L)
    o_to_idx: dict
    sales: np.ndarray                # (n_periods, n_sku) realized demand
    train_origins: list
    oracle: dict                     # origin -> cost (train + valid)
    layer: object = field(default=None, repr=False)

    def window(self, o: int):
        i = self.o_to_idx[o]
        future = self.sales[o:o + 28]
        y29, inv0, pipe0 = to_lp_window(future, self.inv[i], self.pipe[i])
        return y29, inv0, pipe0, future


@dataclass
class BPrep:
    data: object                     # StageData from prepare_stage
    groups: list
    valid_origins: list
    cfg: TSMixerConfig
    bundle: dict
    lead: int
    schedule_sha: str
    exposures: dict


def _tensorize(batch, device):
    keys = ("x_num", "x_event", "z_num", "z_event", "s_codes")
    return {k: torch.from_numpy(getattr(batch, k).copy()).to(device)
            for k in keys}


def prepare_b(stage: str, fold: int, r: int, cfg: TSMixerConfig,
              schedule_path: Path, states_dir: Path, oracle_dir: Path,
              staging_root: Path) -> BPrep:
    schedule = json.loads(schedule_path.read_text())
    bundle = load_bundle(staging_root / f"fold_{fold:02d}",
                         expected_input_schema_hash=input_schema_hash())
    data = prepare_stage(bundle, stage, "b", r, cfg)
    static = bundle["series_static"]
    panel_df = bundle["panel_train"]
    bundles_df = bundle["bundles"].set_index("group_id")
    meta = bundle["fold_meta"]

    state_dir = states_dir / f"{stage}_f{fold:02d}_r{r}"
    state_manifest = json.loads((state_dir / "manifest.json").read_text())
    oracle = json.loads(
        (oracle_dir / f"{stage}_f{fold:02d}_r{r}.json").read_text())
    sched_sha = schedule["sha256"]
    if not (state_manifest["schedule_sha256"] == sched_sha
            and oracle["meta"]["schedule_sha256"] == sched_sha):
        raise SystemExit("schedule sha mismatch across states/oracle")
    if oracle["meta"]["state_files_sha256"] != state_manifest["files"]:
        raise SystemExit("oracle cache was built from different state files")

    if stage == "tune":
        part = schedule["tune"]["bundles"]
        valid_origins = [int(o) for o in schedule["validation"]["origins"]]
        cutoff = int(meta["tune_train_end_idx"])
        exposures = schedule["exposures"]["tune"]
    else:
        part = schedule["final"][str(fold)]["bundles"]
        valid_origins = []
        cutoff = int(meta["train_end_idx"])
        exposures = schedule["exposures"][f"final_f{fold:02d}"]
    train_end = int(meta["train_end_idx"])

    group_order = static["group_id"].to_numpy()
    series_all = static["series_id"].to_numpy()
    cache = LpLayerCache()
    groups, lead = [], None
    for gid in sorted(part):
        rows = np.flatnonzero(group_order == gid)
        sids = [str(s) for s in series_all[rows]]
        npz = np.load(state_dir / f"{gid}.npz", allow_pickle=False)
        if [str(s) for s in npz["series_ids"]] != sids:
            raise SystemExit(f"{gid}: state series order != series_static")
        sales = wide_sales(panel_df, sids)
        h = np.asarray(data.h_cost[rows], dtype=np.float64).copy()
        b = np.asarray(data.b_cost[rows], dtype=np.float64).copy()
        k_g = float(bundles_df.loc[gid, K_G_COLUMN])
        inv = np.asarray(npz["inv"], dtype=np.float64).copy()
        pipe = np.asarray(npz["pipeline"], dtype=np.float64).copy()
        lead = pipe.shape[2]
        o_to_idx = {int(o): i for i, o in enumerate(npz["origins"])}
        train_origins = [int(o) for o in part[gid]["origins"]]

        needed = train_origins + valid_origins
        if max(train_origins) + 28 > cutoff:
            raise SystemExit(f"{gid}: train origin leaks past cutoff")
        if valid_origins and max(valid_origins) + 28 > train_end:
            raise SystemExit(f"{gid}: valid origin leaks past train_end")
        oracle_g = {}
        for o in needed:
            rec = oracle["bundles"][gid][str(o)]
            key = sample_key(inv[o_to_idx[o]], pipe[o_to_idx[o]],
                             sales[o:o + 28], h, b, k_g)
            if key != rec["key"]:
                raise SystemExit(f"{gid} o={o}: oracle cache invalidated "
                                 "(state/path/cost changed) -- rebuild it")
            oracle_g[o] = float(rec["cost"])

        layer = cache.get((gid, r), h, b, np.ones(len(sids)), k_g,
                          LP_WINDOW_DAYS, lead)
        groups.append(BundleGroup(
            gid=gid, rows=rows, k_g=k_g, h=h, b=b,
            h_t=torch.from_numpy(h), b_t=torch.from_numpy(b),
            inv=inv, pipe=pipe, o_to_idx=o_to_idx, sales=sales,
            train_origins=train_origins, oracle=oracle_g, layer=layer))
    return BPrep(data=data, groups=groups, valid_origins=valid_origins,
                 cfg=cfg, bundle=bundle, lead=lead,
                 schedule_sha=sched_sha, exposures=exposures)


def sample_regret(model, prep: BPrep, group: BundleGroup, o: int,
                  device: str, probe_lp_grad: bool = False):
    """One (bundle, origin) sample. Returns (regret tensor, stats dict,
    d29 ref when probing the LP-input gradient)."""
    batch = build_batch(prep.data.panel, prep.data.static, group.rows,
                        np.full(group.rows.shape[0], o), prep.cfg)
    t = _tensorize(batch, device)
    out = model(t["x_num"], t["x_event"], t["z_num"], t["z_event"],
                t["s_codes"])
    scale_t = torch.from_numpy(batch.scale.astype(np.float64).copy()).to(device)
    pred_raw = out.to_raw(scale_t)                     # (n_g, 28), fp64
    d28 = pred_raw.transpose(0, 1)
    zero = torch.zeros((1, d28.shape[1]), dtype=torch.float64, device=device)
    d29 = torch.cat([zero, d28], dim=0)
    if probe_lp_grad:
        d29.retain_grad()

    y29, inv0, pipe0, _ = group.window(o)
    inv_t = torch.from_numpy(inv0)
    pipe_t = torch.from_numpy(pipe0)
    (q,) = group.layer(d29, inv_t, pipe_t, solver_args=SOLVER_ARGS)
    cost = downstream_cost(q, torch.from_numpy(y29), inv_t, pipe_t,
                           group.h_t, group.b_t)
    regret = cost - group.oracle[o]
    stats = {"saturation": float(
        (out.values_scaled.detach() < SATURATION_THRESHOLD)
        .double().mean().item())}
    return regret, stats, (d29 if probe_lp_grad else None)


def run_epoch(model, opt, prep: BPrep, epoch: int, batch_size: int,
              device: str, train: bool):
    samples = [(gi, o) for gi, g in enumerate(prep.groups)
               for o in (g.train_origins if train else prep.valid_origins)]
    if train:
        order = np.random.default_rng(EPOCH_ORDER_SEED + epoch).permutation(
            len(samples))
    else:
        order = np.arange(len(samples))
    model.train(mode=train)
    total, sat_sum, n = 0.0, 0.0, 0
    min_regret = float("inf")
    lp_grad_norm = backbone_grad_norm = None
    head_param_ids = {id(p) for p in model.head.parameters()}
    for lo in range(0, len(order), batch_size):
        idx = order[lo:lo + batch_size]
        probe = train and lo == 0
        regrets, d29_ref = [], None
        for j, flat in enumerate(idx):
            gi, o = samples[int(flat)]
            want_probe = probe and j == 0
            if train:
                regret, stats, d29 = sample_regret(
                    model, prep, prep.groups[gi], o, device,
                    probe_lp_grad=want_probe)
                if want_probe:
                    d29_ref = d29
            else:
                with torch.no_grad():
                    regret, stats, _ = sample_regret(
                        model, prep, prep.groups[gi], o, device)
            regrets.append(regret)
            sat_sum += stats["saturation"]
            min_regret = min(min_regret, float(regret.detach()))
        loss = torch.stack(regrets).mean()
        if train:
            opt.zero_grad()
            loss.backward()
            if probe:
                if d29_ref is not None and d29_ref.grad is not None:
                    lp_grad_norm = float(d29_ref.grad.norm())
                sq = sum(float(p.grad.detach().pow(2).sum())
                         for p in model.parameters()
                         if p.grad is not None
                         and id(p) not in head_param_ids)
                backbone_grad_norm = float(np.sqrt(sq))
            opt.step()
        total += float(loss.detach()) * len(idx)
        n += len(idx)
    out = {"regret_mean": total / max(n, 1),
           "min_sample_regret": min_regret,
           "saturation_share": sat_sum / max(n, 1),
           "n_samples": n}
    if train:
        out["lp_input_grad_norm"] = lp_grad_norm
        out["backbone_grad_norm"] = backbone_grad_norm
        out["optimizer_steps"] = int(np.ceil(len(order) / batch_size))
    return out


def make_model(prep: BPrep, seed: int, device: str) -> TSMixerExt:
    det = ckpt.set_training_determinism(seed)
    model = TSMixerExt(
        prep.cfg, prep.data.static.cardinalities,
        int(prep.bundle["encoder"]["cardinalities"]["event_type"]),
        prediction_kind=MODEL_KIND["b"]).to(device)
    return model, det


def probe_pair(model, prep: BPrep, device: str, k: int = 8):
    g = prep.groups[0]
    rows = g.rows[:k]
    origin = g.train_origins[0]
    batch = build_batch(prep.data.panel, prep.data.static, rows,
                        np.full(rows.shape[0], origin), prep.cfg)
    t = _tensorize(batch, device)
    model.eval()
    with torch.no_grad():
        out = model(t["x_num"], t["x_event"], t["z_num"], t["z_event"],
                    t["s_codes"]).values_scaled
    return {k2: v.cpu() for k2, v in t.items()}, out.cpu()
