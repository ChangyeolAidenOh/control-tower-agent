"""ST4-04 staging tests. Torch-free: main suite. A synthetic
PanelSource-shaped object stands in for the mart; no DB access."""

import json

import numpy as np
import pandas as pd
import pytest

from core_pipeline.forecast.tsmixer import staging
from core_pipeline.forecast.tsmixer.config import (
    checkpoint_key,
    input_schema_hash,
    model_seed,
)

S, T = 4, 160
SERIES = [f"s{i}" for i in range(S)]
TUNE_END = 100


class FakeSource:
    """Duck-typed PanelSource over a synthetic panel. load_demand_price is
    the single chokepoint: rows past the cap never leave this object."""

    def __init__(self, seed: int = 0, bump_after: int | None = None):
        rng = np.random.default_rng(seed)
        rows = []
        for i, sid in enumerate(SERIES):
            first = 1 if i < 3 else 30
            for t in range(1, T + 1):
                active = t >= first
                sales = float(rng.poisson(0.4 if i == 2 else 4.0)) if active else 0.0
                price = float(2.0 + i + 0.1 * np.sin(t / 7)) if active else np.nan
                if bump_after is not None and t > bump_after and active:
                    sales += 1000.0
                    price += 1000.0
                rows.append((sid, t, sales, price, active))
        self._dp = pd.DataFrame(
            rows, columns=["series_id", "period_idx", "sales",
                           "price_ffill", "is_active"])
        cal = []
        for st in ("CA", "TX"):
            for t in range(1, T + 1):
                ev = "none" if t % 11 else ("holiday" if t % 22 else "sport")
                cal.append((st, t, t % 7 + 1, (t // 30) % 12 + 1,
                            float(t % 3 == 0), ev, float(ev != "none")))
        self._cal = pd.DataFrame(
            cal, columns=["state_id", "period_idx", "wday", "month",
                          "snap", "event_type", "is_event"])
        self._static = pd.DataFrame({
            "series_id": SERIES,
            "item_id": [f"it{i}" for i in range(S)],
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "state_id": ["CA", "CA", "TX", "TX"],
            "dept_id": ["d1", "d2", "d1", "d2"],
            "cat_id": ["c1", "c1", "c2", "c2"],
            "group_id": ["g1", "g1", "g2", "g2"],
            "first_active_idx": [1, 1, 1, 30],
        })
        self._folds = pd.DataFrame({
            "fold": [1, 2, 3],
            "test_start_idx": [121, 141, 151],
            "test_end_idx": [148, 150, 160],
            "train_end_idx": [120, 140, 150],
            "warmup_start_idx": [107, 127, 137],
            "is_retrain": [1, 0, 1],
        })

    def load_static(self):
        return self._static.copy()

    def load_folds(self):
        return self._folds.copy()

    def load_demand_price(self, max_period_idx: int):
        return self._dp[self._dp["period_idx"] <= max_period_idx].copy()

    def load_calendar(self, start_idx: int, end_idx: int):
        m = self._cal["period_idx"].between(start_idx, end_idx)
        return self._cal[m].copy()

    def max_calendar_idx(self) -> int:
        return T


def _econ():
    p_bar = np.array([2.0, 3.0, 4.0, 5.0])
    h = p_bar / np.median(p_bar)
    return pd.DataFrame({
        "series_id": SERIES, "p_bar": p_bar, "listed_days": 150,
        "h": h, "b_r3": 3 * h, "b_r5": 5 * h, "b_r10": 10 * h,
    })


def _bundles():
    return pd.DataFrame({"group_id": ["g1", "g2"], "n_sku": [2, 2],
                         "k_g_kappa_1_0": [10.0, 12.0]})


def _export(tmp_path, source):
    return staging.export_staging(
        source, _econ(), _bundles(), tmp_path / "stg",
        tune_train_end=TUNE_END)


def test_physical_truncation_and_integrity(tmp_path):
    manifests = _export(tmp_path, FakeSource())
    assert [m["fold"] for m in manifests] == [1, 3]
    for m in manifests:
        bdir = tmp_path / "stg" / f"fold_{m['fold']:02d}"
        b = staging.load_bundle(bdir)  # re-hashes every file, hard fail
        for key in ("panel_train", "known_calendar"):
            assert int(b[key]["period_idx"].max()) <= m["train_end_idx"]
        assert set(m["files"]) == set(staging.BUNDLE_FILES)
        assert b["manifest"]["input_schema_hash"] == input_schema_hash()
    # tune artifacts only in the first retrain fold
    b1 = staging.load_bundle(tmp_path / "stg" / "fold_01")
    b3 = staging.load_bundle(tmp_path / "stg" / "fold_03")
    assert "scale_tune" in b1["scalers"] and "scale_tune" not in b3["scalers"]
    assert "h_tune" in b1["economic_params"].columns
    assert "h_tune" not in b3["economic_params"].columns
    assert b1["fold_meta"]["tune_train_end_idx"] == TUNE_END


def test_tuning_asof_leakage(tmp_path):
    _export(tmp_path / "a", FakeSource(seed=0))
    _export(tmp_path / "b", FakeSource(seed=0, bump_after=TUNE_END))
    a = staging.load_bundle(tmp_path / "a" / "stg" / "fold_01")
    b = staging.load_bundle(tmp_path / "b" / "stg" / "fold_01")
    # everything as-of tune_end is invariant to post-tune_end changes
    for key in ("scale_tune", "price_mean_tune", "floor_mask_tune"):
        assert np.array_equal(a["scalers"][key], b["scalers"][key]), key
    ea = a["economic_params"].sort_values("series_id")
    eb = b["economic_params"].sort_values("series_id")
    for col in ("p_bar_tune", "rho_tune", "h_tune", "b_tune_r3", "b_tune_r5"):
        assert np.allclose(ea[col], eb[col]), col
    # while the final (train_end) statistics do move
    assert not np.allclose(a["scalers"]["scale_final"],
                           b["scalers"]["scale_final"])
    assert not np.allclose(a["scalers"]["price_mean_final"],
                           b["scalers"]["price_mean_final"])


def test_scaler_floor_and_price_mask(tmp_path):
    _export(tmp_path, FakeSource())
    b = staging.load_bundle(tmp_path / "stg" / "fold_01")
    sc, meta = b["scalers"], b["fold_meta"]["scalers"]["final"]
    assert sc["floor_mask_final"][2]  # low-rate series floored at 1.0
    assert sc["scale_final"][2] == staging.SCALE_FLOOR
    assert meta["floor_count"] == int(sc["floor_mask_final"].sum())
    assert 0.0 < meta["floor_ratio"] <= 1.0
    # late starter: price mean over listed days only (NaN before day 30)
    assert np.isfinite(sc["price_mean_final"]).all()
    assert sc["price_mean_final"][3] > 4.9


def test_encoder_contract(tmp_path):
    _export(tmp_path, FakeSource())
    b1 = staging.load_bundle(tmp_path / "stg" / "fold_01")
    b3 = staging.load_bundle(tmp_path / "stg" / "fold_03",
                             expected_encoder_hash=b1["encoder"]["hash"])
    assert b1["encoder"]["event_type"]["none"] == 0
    assert b1["encoder"]["hash"] == b3["encoder"]["hash"]
    assert set(b1["encoder"]["static"]) == {
        "series_id", "store_id", "dept_id", "cat_id", "state_id"}
    assert b1["encoder"]["cardinalities"]["series_id"] == S


def test_load_bundle_hard_fail(tmp_path):
    _export(tmp_path, FakeSource())
    bdir = tmp_path / "stg" / "fold_01"
    path = bdir / "scalers.npz"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        staging.load_bundle(bdir)
    with pytest.raises(ValueError, match="encoder hash"):
        staging.load_bundle(tmp_path / "stg" / "fold_03",
                            expected_encoder_hash="deadbeef0000")


def test_checkpoint_keys_and_seeds():
    assert checkpoint_key("a1") == "a1"
    assert checkpoint_key("a2", 3) == "a2_r3"
    assert checkpoint_key("b", 5) == "b_r5"
    with pytest.raises(ValueError):
        checkpoint_key("a2")          # r mandatory for A2
    with pytest.raises(ValueError):
        checkpoint_key("a1", 3)       # r forbidden for A1
    with pytest.raises(ValueError):
        checkpoint_key("a2", 10)      # r=10 is evaluation-only (reuse r=5)
    # Stage 4: 16 final checkpoints; with Stage 5 B variants, 24 seeds
    stage4 = [("a1", None), ("a2", 3), ("a2", 5), ("a3", None)]
    seeds = {model_seed(42, m, f, r)
             for m, r in stage4 + [("b", 3), ("b", 5)]
             for f in (1, 4, 7, 10)}
    assert len(seeds) == 24
    assert len({checkpoint_key(m, r) for m, r in stage4}) * 4 == 16


def test_checkpoint_fold_mapping():
    from core_pipeline.forecast.tsmixer.config import checkpoint_fold_for
    expected = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4,
                7: 7, 8: 7, 9: 7, 10: 10, 11: 10, 12: 10}
    assert {f: checkpoint_fold_for(f) for f in range(1, 13)} == expected
    with pytest.raises(ValueError):
        checkpoint_fold_for(13)


def test_tuning_grid_and_selection():
    from core_pipeline.forecast.tsmixer import tuning
    trials = tuning.build_trials()
    assert len(trials) == 24
    assert all(t["ff_dim"] == t["d_model"] and
               t["d_feat"] == t["d_model"] // 2 for t in trials)
    assert len({t["seed"] for t in trials}) == 24
    order = tuning.execution_order()
    assert sorted(order) == list(range(24))
    assert order == tuning.execution_order()  # deterministic shuffle
    assert order != list(range(24))
    rows = [
        {"trial_id": 0, "best_metric": 0.4000, "params": 900},
        {"trial_id": 1, "best_metric": 0.4003, "params": 500},  # tie, smaller
        {"trial_id": 2, "best_metric": 0.4003, "params": 500},  # tie, higher id
        {"trial_id": 3, "best_metric": 0.4200, "params": 100},  # outside tie
    ]
    sel = tuning.select_trial(rows)
    assert sel["trial_id"] == 1 and sel["tie_rule_applied"]
    assert sel["tie_group"] == [0, 1, 2]
    with pytest.raises(ValueError):
        tuning.select_trial([])
