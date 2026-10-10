"""ST4-01 contract tests (4-A). Five of the six required checks; the
gradient test (TSMixer -> inverse scale -> LP -> downstream loss) lands in
4-B after the architecture is verified.

All data here is synthetic; no DB access (ST3-06 phase discipline)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from core_pipeline.forecast.tsmixer.adapters import (
    PanelInputs,
    StaticInputs,
    build_batch,
    training_origins,
)
from core_pipeline.forecast.tsmixer.config import (
    MODEL_KIND,
    QUANTILES_9,
    TSMixerConfig,
    model_seed,
)
from core_pipeline.forecast.tsmixer.losses import (
    cost_weighted_pinball_raw,
    mse_loss_raw,
    pinball_grid_raw,
)
from core_pipeline.forecast.tsmixer.model import TSMixerExt
from core_pipeline.forecast.tsmixer.postproc import finalize_quantiles
from core_pipeline.forecast.tsmixer.scaler import SeriesMeanScaler

S, T, IDX_START = 6, 200, 101
CFG = TSMixerConfig()
STATIC_CARDS = (6, 3, 2)
EVENT_CARD = 5


def _panel(seed: int = 0) -> PanelInputs:
    rng = np.random.default_rng(seed)
    sales = rng.poisson(3.0, size=(S, T)).astype(np.float64)
    first_active = np.array([IDX_START, IDX_START, IDX_START + 10,
                             IDX_START + 40, IDX_START, IDX_START + 5])
    is_active = np.zeros((S, T))
    for i, fa in enumerate(first_active):
        is_active[i, fa - IDX_START:] = 1.0
    sales *= is_active
    scale = np.maximum(sales.mean(axis=1), 1.0)
    return PanelInputs(
        idx_start=IDX_START,
        series_ids=np.array([f"s{i}" for i in range(S)]),
        sales_scaled=sales / scale[:, None],
        price_rel=rng.uniform(0.8, 1.2, size=(S, T)),
        is_active=is_active,
        snap=rng.integers(0, 2, size=(S, T)).astype(np.float64),
        is_event=rng.integers(0, 2, size=(S, T)).astype(np.float64),
        event_code=rng.integers(0, EVENT_CARD, size=(S, T)),
        wday=(np.arange(T) % 7) + 1,
        month=(np.arange(T) // 30) % 12 + 1,
        first_active_idx=first_active,
        scale=scale,
    )


def _static() -> StaticInputs:
    rng = np.random.default_rng(1)
    codes = np.stack([np.arange(S),
                      rng.integers(0, 3, size=S),
                      rng.integers(0, 2, size=S)], axis=1)
    return StaticInputs(("series_id", "store_id", "state_id"), codes, STATIC_CARDS)


def _model(kind: str, seed: int = 7) -> TSMixerExt:
    torch.manual_seed(seed)
    return TSMixerExt(CFG, STATIC_CARDS, EVENT_CARD, prediction_kind=kind)


def _tensors(batch):
    return (torch.from_numpy(batch.x_num), torch.from_numpy(batch.x_event),
            torch.from_numpy(batch.z_num), torch.from_numpy(batch.z_event),
            torch.from_numpy(batch.s_codes))


def test_shapes_and_origin_validity():
    panel, static = _panel(), _static()
    train_end = IDX_START + T - 1 - 10
    rows, origins = training_origins(panel, train_end, CFG)
    assert rows.size > 0
    # every origin respects grid, horizon and the 56-active-day rule
    assert (origins - panel.first_active_idx[rows] + 1 >= CFG.past_length).all()
    assert (origins + CFG.horizon <= train_end).all()
    # the late starter (first_active + 40) loses exactly its early origins
    late = rows == 3
    assert origins[late].min() == panel.first_active_idx[3] + CFG.past_length - 1

    batch = build_batch(panel, static, rows[:32], origins[:32], CFG, with_targets=True)
    assert batch.x_num.shape == (32, 56, 14)
    assert batch.z_num.shape == (32, 28, 11)
    assert batch.y_scaled.shape == (32, 28)

    for kind, out_shape in (("mse_point", (32, 28)), ("quantile_grid", (32, 28, 9))):
        out = _model(kind)(*_tensors(batch))
        assert out.kind == kind
        assert tuple(out.values_scaled.shape) == out_shape
        raw = out.to_raw(torch.from_numpy(batch.scale))
        assert tuple(raw.shape) == out_shape


def test_temporal_leakage():
    panel, static = _panel(), _static()
    origin = IDX_START + 120
    rows = np.array([0, 1, 2, 4, 5])  # series 3 starts too late for this origin
    origins = np.full(rows.shape, origin)
    batch_a = build_batch(panel, static, rows, origins, CFG)

    poisoned = _panel()
    j = origin - IDX_START
    for name in ("sales_scaled", "price_rel", "is_active", "snap", "is_event"):
        getattr(poisoned, name)[:, j + 1:] += 99.0
    poisoned.event_code[:, j + 1:] = 0  # future event codes feed z by design
    batch_b = build_batch(poisoned, static, rows, origins, CFG)

    for name in ("x_num", "x_event", "s_codes", "scale"):
        assert np.array_equal(getattr(batch_a, name), getattr(batch_b, name)), name
    # and the model output from past-only channels is bitwise identical
    model = _model("mse_point")
    model.eval()
    with torch.no_grad():
        out_a = model(*_tensors(batch_a)).values_scaled
        xb = _tensors(batch_b)
        out_b = model(xb[0], xb[1], *_tensors(batch_a)[2:]).values_scaled
    assert torch.equal(out_a, out_b)


def test_scaling_and_raw_losses():
    rng = np.random.default_rng(2)
    sales = rng.poisson(4.0, size=(4, 100)).astype(np.float64)
    sales[3] = 0.0  # dead series -> eps floor
    mask = np.zeros_like(sales, dtype=bool)
    mask[:, :60] = True  # train span only
    sc = SeriesMeanScaler(eps=1.0).fit(sales, mask)
    poisoned = sales.copy()
    poisoned[:, 60:] += 1000.0
    sc2 = SeriesMeanScaler(eps=1.0).fit(poisoned, mask)
    assert np.allclose(sc.scale_, sc2.scale_)  # cutoff respected
    assert sc.scale_[3] == 1.0
    assert np.allclose(sc.inverse(sc.transform(sales)), sales)

    # raw-unit newsvendor identity: yhat=3, d=1, h=2, b=6 -> 2 * (3-1) = 4
    yhat = torch.full((1, 2), 3.0)
    d = torch.full((1, 2), 1.0)
    loss = cost_weighted_pinball_raw(yhat, d, torch.tensor([2.0]), torch.tensor([6.0]))
    assert torch.isclose(loss, torch.tensor(4.0))
    assert torch.isclose(mse_loss_raw(yhat, d), torch.tensor(4.0))
    # pinball grid at d == q collapses to zero
    q = d.unsqueeze(-1).expand(1, 2, 9).clone()
    assert torch.isclose(pinball_grid_raw(q, d), torch.tensor(0.0))


def test_model_parity():
    seed = model_seed(42, "a1", 1)
    sds = {}
    for mid in ("a1", "a2", "b"):
        sds[mid] = _model(MODEL_KIND[mid], seed=seed).state_dict()
    for mid in ("a2", "b"):
        assert sds["a1"].keys() == sds[mid].keys()
        for k in sds["a1"]:
            assert torch.equal(sds["a1"][k], sds[mid][k]), k
    # A3: identical backbone structure, head differs only in width
    a1 = _model("mse_point", seed=seed)
    a3 = _model("quantile_grid", seed=seed)
    b1, b3 = a1.backbone_state_dict(), a3.backbone_state_dict()
    assert b1.keys() == b3.keys()
    for k in b1:
        assert b1[k].shape == b3[k].shape, k
    assert a1.param_counts()["backbone"] == a3.param_counts()["backbone"]
    assert a3.param_counts()["head"] == a1.param_counts()["head"] * 9
    # seeds differ across (model, r, fold) by protocol (ST4-04: 16
    # Stage 4 checkpoints; A2 trains per r because b = r * h moves tau)
    stage4 = [("a1", None), ("a2", 3), ("a2", 5), ("a3", None)]
    assert len({model_seed(42, m, f, r) for m, r in stage4
                for f in (1, 4, 7, 10)}) == 16


def test_output_semantics():
    panel, static = _panel(), _static()
    rows = np.array([0, 1])
    origins = np.full(2, IDX_START + 150)
    batch = build_batch(panel, static, rows, origins, CFG)

    with pytest.raises(ValueError):
        TSMixerExt(CFG, STATIC_CARDS, EVENT_CARD, prediction_kind="median")

    for mid in ("a1", "a2", "b"):
        out = _model(MODEL_KIND[mid])(*_tensors(batch))
        assert out.kind == MODEL_KIND[mid]
        assert out.values_scaled.dim() == 2      # yhat, not a quantile grid
        assert (out.values_scaled >= 0).all()    # shared softplus transform

    out = _model("quantile_grid")(*_tensors(batch))
    assert out.kind == "quantile_grid"
    raw = out.to_raw(torch.from_numpy(batch.scale)).detach().numpy()
    q_final, crossing = finalize_quantiles(raw)
    assert 0.0 <= crossing <= 1.0
    assert (q_final >= 0).all()
    assert (np.diff(q_final, axis=-1) >= 0).all()
    # crossing measured pre-rearrangement on a known inversion
    toy = np.zeros((1, 1, 9)); toy[0, 0] = np.arange(9)[::-1]
    _, c = finalize_quantiles(toy)
    assert c == 1.0
    assert len(QUANTILES_9) == 9


# ---- ST4-05 training-side contract (bundle -> train -> checkpoint) ----

from core_pipeline.forecast.tsmixer import staging
from core_pipeline.forecast.tsmixer import checkpoint as ckpt_mod
from core_pipeline.forecast.tsmixer.train import prepare_stage, train_one

TINY = TSMixerConfig(d_feat=8, d_static=4, d_model=16, ff_dim=16,
                     n_blocks=1, dropout=0.0, event_emb_dim=2,
                     max_static_emb_dim=4)
TRAIN_END, TUNE_END_B = 150, 100


def _bundle(tmp_path):
    rng = np.random.default_rng(3)
    series = [f"s{i}" for i in range(4)]
    rows = []
    for i, sid in enumerate(series):
        first = 1 if i < 3 else 30
        for t in range(1, TRAIN_END + 1):
            active = t >= first
            rows.append((sid, t,
                         float(rng.poisson(4.0)) if active else 0.0,
                         float(2.0 + i) if active else np.nan, active))
    dp = pd.DataFrame(rows, columns=["series_id", "period_idx", "sales",
                                     "price_ffill", "is_active"])
    cal = []
    for st in ("CA", "TX"):
        for t in range(1, TRAIN_END + 1):
            ev = "none" if t % 9 else "holiday"
            cal.append((st, t, t % 7 + 1, (t // 30) % 12 + 1,
                        float(t % 3 == 0), ev, float(ev != "none")))
    cal = pd.DataFrame(cal, columns=["state_id", "period_idx", "wday",
                                     "month", "snap", "event_type",
                                     "is_event"])
    static = pd.DataFrame({
        "series_id": series, "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
        "state_id": ["CA", "CA", "TX", "TX"],
        "dept_id": ["d1", "d2", "d1", "d2"],
        "cat_id": ["c1", "c1", "c2", "c2"],
        "group_id": ["g1", "g1", "g2", "g2"],
        "first_active_idx": [1, 1, 1, 30]})
    h = np.array([1.0, 1.5, 2.0, 0.5])
    econ = pd.DataFrame({"series_id": series, "p_bar": 2 * h,
                         "listed_days": 140, "h": h, "b_r3": 3 * h,
                         "b_r5": 5 * h, "b_r10": 10 * h})
    bundles = pd.DataFrame({"group_id": ["g1", "g2"],
                            "k_g_kappa_1_0": [10.0, 12.0]})
    fold_row = {"fold": 1, "train_end_idx": TRAIN_END,
                "test_start_idx": TRAIN_END + 1,
                "test_end_idx": TRAIN_END + 28,
                "warmup_start_idx": TRAIN_END - 13, "is_retrain": 1}
    enc = staging.build_encoder(cal, static)
    tune = {"tune_end": TUNE_END_B,
            "econ": staging.compute_tune_econ(dp, series, TUNE_END_B)}
    staging.write_bundle(tmp_path / "fold_01", fold_row, dp, cal, static,
                         econ, bundles, enc, tune)
    return staging.load_bundle(tmp_path / "fold_01")


def test_tune_stage_split_and_scale_selection(tmp_path):
    bundle = _bundle(tmp_path)
    tune = prepare_stage(bundle, "tune", "a2", 3, TINY)
    final = prepare_stage(bundle, "final", "a2", 3, TINY)
    assert tune.train_origins.size and tune.valid_origins.size
    assert (tune.train_origins + TINY.horizon <= TUNE_END_B).all()
    assert (tune.valid_origins >= TUNE_END_B).all()
    assert (tune.valid_origins + TINY.horizon <= TRAIN_END).all()
    assert np.array_equal(tune.scale, bundle["scalers"]["scale_tune"])
    assert np.array_equal(final.scale, bundle["scalers"]["scale_final"])
    assert np.allclose(tune.h_cost,
                       bundle["economic_params"]["h_tune"].to_numpy())
    assert np.allclose(final.b_cost,
                       bundle["economic_params"]["b_r3"].to_numpy())


def test_train_checkpoint_cycle(tmp_path):
    bundle = _bundle(tmp_path)
    out = tmp_path / "ckpt"
    res = train_one(bundle, "a2", 1, 3, "final", TINY, out, epochs=2,
                    batch_size=64, lr=1e-3, global_seed=42, device="cpu")
    model, aux, manifest = ckpt_mod.load_checkpoint(
        out, "a2", 1, 3, expected_encoder_hash=bundle["encoder"]["hash"])
    assert model.prediction_kind == "cost_weighted_point"
    assert aux["probe_max_abs_diff"] <= ckpt_mod.PROBE_ATOL
    assert np.array_equal(aux["scale"], bundle["scalers"]["scale_final"])
    required = ["manifest_schema_version", "checkpoint_key", "model_id",
                "fold", "r", "prediction_kind", "tsmixer_config",
                "adapter_version", "input_schema_hash", "state_dict_sha256",
                "train_end", "bundle_sha256", "scale_sha256",
                "encoder_hash", "seed", "epochs", "best_epoch",
                "selection_metric", "tf32_matmul", "tf32_cudnn", "python",
                "torch", "cuda", "device", "git_commit",
                "model_parameter_dtype", "forward_dtype",
                "loss_compute_dtype", "precision_policy", "amp_enabled"]
    missing = [k for k in required if k not in manifest]
    assert not missing, missing
    assert manifest["scale_sha256"] == (
        bundle["manifest"]["files"]["scalers.npz"]["sha256"])
    # hard fails: wrong encoder, tampered sha, wrong identity
    with pytest.raises(ValueError, match="encoder_hash"):
        ckpt_mod.load_checkpoint(out, "a2", 1, 3,
                                 expected_encoder_hash="deadbeef0000")
    mf = out / "tsmixer_a2_r3_f1.json"
    meta = json.loads(mf.read_text())
    meta["state_dict_sha256"] = "0" * 16
    mf.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        ckpt_mod.load_checkpoint(out, "a2", 1, 3,
                                 expected_encoder_hash=bundle["encoder"]["hash"])
    assert "checkpoint" in res and "manifest" in res


# ---- ST4-06 inference-side contract ----

from core_pipeline.forecast import point_store, prediction_store
from core_pipeline.forecast.tsmixer.config import QUANTILES_9
from core_pipeline.forecast.tsmixer.infer import (
    inference_pairs, long_arrays, predict,
)
from core_pipeline.forecast.tsmixer.panel import (
    build_panel, build_static_inputs,
)
from core_pipeline.forecast.tsmixer.postproc import finalize_quantiles
from core_pipeline.forecast.tsmixer.checkpoint import load_checkpoint


def _inference_setup(tmp_path, model_id, r):
    bundle = _bundle(tmp_path)
    out = tmp_path / "ckpt"
    train_one(bundle, model_id, 1, r, "final", TINY, out, epochs=1,
              batch_size=64, lr=1e-3, global_seed=42, device="cpu")
    model, aux, manifest = load_checkpoint(
        out, model_id, 1, r, expected_encoder_hash=bundle["encoder"]["hash"])
    # inference panel: demand to TRAIN_END, calendar extended past it
    dp = bundle["panel_train"]
    cal = bundle["known_calendar"]
    ext = []
    for st in ("CA", "TX"):
        for t in range(TRAIN_END + 1, TRAIN_END + 29):
            ev = "none" if t % 9 else "holiday"
            ext.append((st, t, t % 7 + 1, (t // 30) % 12 + 1,
                        float(t % 3 == 0), ev, float(ev != "none")))
    cal_full = pd.concat([cal, pd.DataFrame(ext, columns=cal.columns)],
                         ignore_index=True)
    panel = build_panel(dp, cal_full, bundle["series_static"],
                        bundle["encoder"], aux["scale"], aux["price_mean"],
                        idx_start=1, idx_end=TRAIN_END + 28)
    static_in = build_static_inputs(bundle["series_static"],
                                    bundle["encoder"])
    return bundle, model, aux, panel, static_in, dp, cal_full


def test_inference_point_frame_and_leakage(tmp_path):
    bundle, model, aux, panel, static_in, dp, cal_full = _inference_setup(
        tmp_path, "a2", 3)
    origins = [TRAIN_END - 1, TRAIN_END]  # second origin: pure future calendar
    rows, orgs = inference_pairs(panel, origins, TINY)
    raw = predict(model, panel, static_in, rows, orgs, TINY, batch_size=32)
    assert raw.shape == (rows.shape[0], 28) and (raw >= 0).all()
    sid, org_l, hor = long_arrays(panel, rows, orgs, TINY)
    frame = point_store.build_point_frame(sid, org_l, hor, raw.reshape(-1))
    stats = point_store.validate_point_frame(frame, "cost_weighted_point")
    assert stats["origins"] == 2 and stats["series"] == 4
    # input leakage: demand after the origins cannot move predictions
    dp2 = dp.copy()
    mask = dp2["period_idx"] > TRAIN_END - 1
    dp2.loc[mask, "sales"] = dp2.loc[mask, "sales"] + 999.0
    panel2 = build_panel(dp2, cal_full, bundle["series_static"],
                         bundle["encoder"], aux["scale"],
                         aux["price_mean"], idx_start=1,
                         idx_end=TRAIN_END + 28)
    rows1, orgs1 = inference_pairs(panel2, [TRAIN_END - 1], TINY)
    raw_a = predict(model, panel, static_in, rows1, orgs1, TINY)
    raw_b = predict(model, panel2, static_in, rows1, orgs1, TINY)
    assert np.array_equal(raw_a, raw_b)
    # history rule hard-fails instead of silently dropping
    with pytest.raises(ValueError, match="active-history"):
        inference_pairs(panel, [40], TINY)


def test_inference_quantile_frame(tmp_path):
    bundle, model, aux, panel, static_in, _, _ = _inference_setup(
        tmp_path, "a3", None)
    rows, orgs = inference_pairs(panel, [TRAIN_END], TINY)
    raw = predict(model, panel, static_in, rows, orgs, TINY)
    assert raw.shape == (rows.shape[0], 28, 9)
    fixed, crossing = finalize_quantiles(raw)
    assert 0.0 <= crossing <= 1.0
    sid, org_l, hor = long_arrays(panel, rows, orgs, TINY)
    frame = prediction_store.build_frame(
        sid, org_l, hor, fixed.reshape(-1, 9), QUANTILES_9)
    stats = prediction_store.validate_frame(frame, QUANTILES_9)
    assert stats["rows"] == rows.shape[0] * 28


# ---- ST4-07 tuning-side contract ----

from core_pipeline.forecast.tsmixer.train import (
    _batch_loss, prepare_stage as _prep, tune_run,
)
from core_pipeline.forecast.tsmixer.adapters import build_batch as _bb


def test_precision_fields_and_loss_dtype(tmp_path):
    bundle = _bundle(tmp_path)
    out = tmp_path / "ckpt"
    train_one(bundle, "a1", 1, None, "final", TINY, out, epochs=1,
              batch_size=64, lr=1e-3, global_seed=42, device="cpu")
    manifest = json.loads((out / "tsmixer_a1_f1.json").read_text())
    assert manifest["precision_policy"] == "fp32_model_fp64_loss"
    assert manifest["model_parameter_dtype"] == "float32"
    assert manifest["loss_compute_dtype"] == "float64"
    assert manifest["amp_enabled"] is False
    assert "dtype" not in manifest
    # verified in code, not just recorded (verdict): fp32 params,
    # fp64 at the loss
    data = _prep(bundle, "final", "a2", 3, TINY)
    model = TSMixerExt(TINY, data.static.cardinalities, 2,
                       prediction_kind="cost_weighted_point")
    assert next(model.parameters()).dtype == torch.float32
    batch = _bb(data.panel, data.static, data.train_rows[:8],
                data.train_origins[:8], TINY, with_targets=True)
    loss = _batch_loss(model, batch, data, "a2", "cpu")
    assert loss.dtype == torch.float64


def test_tune_run_contract(tmp_path):
    bundle = _bundle(tmp_path)
    res = tune_run(bundle, "a3", None, TINY, lr=1e-3, seed=90000,
                   device="cpu", epochs=3, batch_size=128, patience=2,
                   valid_metric="wql9")
    assert 1 <= res["best_epoch"] <= res["epochs_ran"] <= 3
    assert np.isfinite(res["best_metric"]) and res["best_metric"] > 0
    assert len(res["epoch_metrics"]) == res["epochs_ran"]
    assert res["valid_metric"] == "wql9"
    assert res["precision"]["precision_policy"] == "fp32_model_fp64_loss"
    assert res["n_valid"] > 0 and res["params"] > 0
    res_a2 = tune_run(bundle, "a2", 3, TINY, lr=1e-3, seed=90001,
                      device="cpu", epochs=2, batch_size=128, patience=2,
                      valid_metric="loss")
    assert np.isfinite(res_a2["best_metric"])


import pytest
pytestmark = pytest.mark.torch
