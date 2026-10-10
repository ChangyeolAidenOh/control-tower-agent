"""ST4-07 tuning orchestrator (Colab, bundle-only, no DB in this chain).

Protocol (verdict-fixed):
  Step 1 (--step grid): 24-trial exhaustive grid on A3 with the tune
    scalers/economics of the fold-1 bundle; trials run in the
    deterministically pre-shuffled order; selection = argmin of the
    rearranged, non-negative, raw-unit validation WQL(9); tie within
    1e-3 relative of the minimum -> fewest parameters -> lowest trial_id.
    The common HP is A3-favoring by design and reported as such.
  Step 2 (--step epochs): under the selected HP, one validation-based
    best_epoch per training objective (a1: valid MSE; a2_r3 / a2_r5:
    valid cost-weighted pinball; a3: valid WQL9). Folds 1/4/7/10 then
    reuse these epochs verbatim -- no per-fold early stopping or
    retuning. This is one HP search plus one per-objective stopping
    correction, not "no tuning".
  --step all runs both.

Budget (provisional until measured): --measure-extremes times 2 epochs
on the smallest (32, 2) and largest (128, 4) configurations first. A
pre-fixed stop rule applies during the grid: before starting a trial,
if elapsed time plus the mean of completed trial times would exceed
--max-minutes, the search stops; evaluated trial count is recorded and
reported -- never a performance-based cut.

Output: <out>/tuning_results.json -- grid, execution order, per-trial
rows (metric, best_epoch, epochs_ran, hit_max, tail trend, sec, vram,
params), selection block, epochs_by_key, environment. Keep it on Drive.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from core_pipeline.forecast.tsmixer.config import (
    TSMixerConfig,
    checkpoint_key,
    input_schema_hash,
    model_seed,
)
from core_pipeline.forecast.tsmixer.staging import load_bundle
from core_pipeline.forecast.tsmixer import tuning
from core_pipeline.forecast.tsmixer.train import tune_run

EPOCH_METRIC = {"a1": "loss", "a2_r3": "loss", "a2_r5": "loss",
                "a3": "wql9"}


def trial_config(t: dict) -> TSMixerConfig:
    return TSMixerConfig(d_model=t["d_model"], n_blocks=t["n_blocks"],
                         dropout=t["dropout"], ff_dim=t["ff_dim"],
                         d_feat=t["d_feat"])


def run_grid(bundle, device, args, trials, order) -> dict:
    rows, elapsed = [], 0.0
    stopped_early = False
    for pos, tid in enumerate(order):
        if rows:
            mean_s = sum(r["sec_total"] for r in rows) / len(rows)
            if (elapsed + mean_s) > args.max_minutes * 60:
                stopped_early = True
                break
        t = trials[tid]
        res = tune_run(bundle, "a3", None, trial_config(t), lr=t["lr"],
                       seed=t["seed"], device=device,
                       epochs=tuning.MAX_EPOCHS,
                       batch_size=args.batch_size,
                       patience=tuning.PATIENCE, valid_metric="wql9")
        row = {**t, **{k: res[k] for k in (
            "best_metric", "best_epoch", "epochs_ran", "hit_max_epochs",
            "tail_rel_trend_last3", "sec_total", "sec_per_epoch",
            "peak_vram_mb", "params")}}
        rows.append(row)
        elapsed += res["sec_total"]
        print(f"[{pos + 1}/{len(order)}] trial {tid:02d} "
              f"d{t['d_model']} b{t['n_blocks']} do{t['dropout']} "
              f"lr{t['lr']} -> wql9 {res['best_metric']:.5f} "
              f"ep {res['best_epoch']}/{res['epochs_ran']} "
              f"{res['sec_total']:.0f}s")
    selection = tuning.select_trial(rows)
    hit_max = sum(1 for r in rows if r["hit_max_epochs"])
    print(f"selected trial {selection['trial_id']} "
          f"(tie_group {selection['tie_group']}) "
          f"evaluated {len(rows)}/24 hit_max {hit_max}")
    return {"rows": rows, "selection": selection,
            "evaluated": len(rows), "stopped_early": stopped_early,
            "hit_max_count": hit_max}


def run_epochs(bundle, device, args, hp: dict) -> dict:
    cfg = trial_config(hp)
    out = {}
    for key in ("a1", "a2_r3", "a2_r5", "a3"):
        mid, r = (key.split("_r") + [None])[:2]
        r = int(r) if r else None
        checkpoint_key(mid, r)
        res = tune_run(bundle, mid, r, cfg, lr=hp["lr"],
                       seed=model_seed(args.global_seed, mid, 0, r),
                       device=device, epochs=tuning.MAX_EPOCHS,
                       batch_size=args.batch_size,
                       patience=tuning.PATIENCE,
                       valid_metric=EPOCH_METRIC[key])
        out[key] = {k: res[k] for k in (
            "best_epoch", "best_metric", "valid_metric", "epochs_ran",
            "hit_max_epochs", "sec_total")}
        print(f"epochs[{key}] best_epoch {res['best_epoch']} "
              f"({res['valid_metric']} {res['best_metric']:.5f}) "
              f"{res['sec_total']:.0f}s")
    return out


def measure_extremes(bundle, device, args) -> dict:
    out = {}
    for label, (dm, nb) in (("min", (32, 2)), ("max", (128, 4))):
        cfg = TSMixerConfig(d_model=dm, n_blocks=nb, ff_dim=dm,
                            d_feat=dm // 2)
        res = tune_run(bundle, "a3", None, cfg, lr=1e-3,
                       seed=tuning.TRIAL_SEED_BASE - 1, device=device,
                       epochs=2, batch_size=args.batch_size,
                       patience=99, valid_metric="wql9")
        out[label] = {"d_model": dm, "n_blocks": nb,
                      "sec_per_epoch": res["sec_per_epoch"],
                      "peak_vram_mb": res["peak_vram_mb"]}
        print(f"extreme {label}: {out[label]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="fold_01 bundle directory (tune scalers inside)")
    ap.add_argument("--out", default="outputs/tuning")
    ap.add_argument("--step", choices=["grid", "epochs", "all"],
                    default="all")
    ap.add_argument("--measure-extremes", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--batch-size", type=int, default=tuning.BATCH_SIZE)
    ap.add_argument("--global-seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_bundle(Path(args.bundle),
                         expected_input_schema_hash=input_schema_hash())
    if "scale_tune" not in bundle["scalers"]:
        raise ValueError("tuning requires the fold-1 bundle (tune scalers)")
    trials = tuning.build_trials()
    order = tuning.execution_order(len(trials))
    print(f"device {device} encoder {bundle['encoder']['hash']} "
          f"order {order}")

    result = {
        "protocol": "ST4-07",
        "grid": trials,
        "execution_order": order,
        "device": device,
        "torch": torch.__version__,
        "input_schema_hash": input_schema_hash(),
        "encoder_hash": bundle["encoder"]["hash"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.measure_extremes:
        result["extremes"] = measure_extremes(bundle, device, args)
    if args.step in ("grid", "all"):
        result["grid_run"] = run_grid(bundle, device, args, trials, order)
        selected = trials[result["grid_run"]["selection"]["trial_id"]]
        result["selected_hp"] = selected
    if args.step in ("epochs", "all"):
        if "selected_hp" not in result:
            prev = json.loads(
                (Path(args.out) / "tuning_results.json").read_text())
            result["selected_hp"] = prev["selected_hp"]
            result["grid_run"] = prev.get("grid_run")
        result["epochs_by_key"] = run_epochs(
            bundle, device, args, result["selected_hp"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tuning_results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"written {path}")


if __name__ == "__main__":
    main()
