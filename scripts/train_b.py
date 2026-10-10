"""Stage 5: B (DFL) training runner.

tune:  fold-1 tune stage; early stopping on the fixed validation regret
       (patience 5, cap 20); writes/merges the selected best_epoch into
       outputs/stage5_b_tuning.json with the full manifest (ST5-03 (3)).
final: loads best_epoch from stage5_b_tuning.json only (no validation
       anywhere), trains the fixed epochs and saves the checkpoint via
       ckpt.save_checkpoint (probe + sha hard-fail on reload, as ST4).
--measure: one training epoch + one validation pass, timing only,
       writes nothing.

HP source: outputs/stage4_tsmixer_tuning.json selected_hp (trial 3,
common architecture + lr -- parity with A1/A2/A3); epochs_by_key is NOT
read (B's epochs come from its own tune run).

Run:
  python -m scripts.train_b --stage tune --r 3 --measure
  caffeinate -i python -m scripts.train_b --stage tune --r 3
  caffeinate -i python -m scripts.train_b --stage final --fold 1 --r 3
"""

import argparse
import json
import time
from pathlib import Path

import torch

from core_pipeline.dfl.train_b import (
    make_model,
    prepare_b,
    probe_pair,
    run_epoch,
)
from core_pipeline.forecast.tsmixer.config import TSMixerConfig, model_seed
from core_pipeline.forecast.tsmixer import checkpoint as ckpt


def load_hp(path: Path):
    tuned = json.loads(path.read_text())
    hp = tuned["selected_hp"]
    cfg = TSMixerConfig(d_model=hp["d_model"], n_blocks=hp["n_blocks"],
                        dropout=hp["dropout"], ff_dim=hp["ff_dim"],
                        d_feat=hp["d_feat"])
    return cfg, float(hp["lr"]), hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--r", type=int, required=True, choices=[3, 5])
    parser.add_argument("--tuning-json",
                        default="outputs/stage4_tsmixer_tuning.json")
    parser.add_argument("--b-tuning", default="outputs/stage5_b_tuning.json")
    parser.add_argument("--schedule", default="outputs/b_sample_schedule.json")
    parser.add_argument("--states-dir", default="outputs/b_warmup_states")
    parser.add_argument("--oracle-dir", default="outputs/b_oracle")
    parser.add_argument("--staging-root", default="outputs/staging/tsmixer")
    parser.add_argument("--ckpt-dir", default="outputs/checkpoints")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs-cap", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.stage == "tune" and args.fold != 1:
        raise SystemExit("tune stage runs on fold 1")

    cfg, lr, hp = load_hp(Path(args.tuning_json))
    prep = prepare_b(args.stage, args.fold, args.r, cfg,
                     Path(args.schedule), Path(args.states_dir),
                     Path(args.oracle_dir), Path(args.staging_root))
    seed = model_seed(args.global_seed, "b", args.fold, args.r)
    model, det = make_model(prep, seed, args.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    manifest_common = {
        "sample_schedule_sha": prep.schedule_sha,
        "origins_per_bundle": len(prep.groups[0].train_origins),
        "sku_origin_exposures": prep.exposures["sku_origin_exposures"],
        "batch_size": args.batch_size, "lr": lr, "hp": hp,
        "seed": seed, **det,
    }

    if args.stage == "tune":
        b_epochs = 2 if args.measure else args.epochs_cap
        history, best, best_epoch, bad = [], float("inf"), 0, 0
        best_state = None
        steps_total, t_start = 0, time.time()
        for epoch in range(1, b_epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, opt, prep, epoch, args.batch_size,
                           args.device, train=True)
            va = run_epoch(model, opt, prep, epoch, args.batch_size,
                           args.device, train=False)
            steps_total += tr["optimizer_steps"]
            rec = {"epoch": epoch, "sec": round(time.time() - t0, 1),
                   "train": tr, "valid": va}
            history.append(rec)
            print(json.dumps(rec))
            if args.measure:
                continue
            if va["regret_mean"] < best:
                best, best_epoch, bad = va["regret_mean"], epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if args.measure:
            sec = history[-1]["sec"]
            print(json.dumps({"measure": True, "sec_per_epoch": sec,
                              "est_hours_cap": round(
                                  args.epochs_cap * sec / 3600.0, 2)}))
            return
        if best_state is None:
            raise SystemExit("no best epoch recorded")
        entry = {"best_epoch": best_epoch, "best_val_regret": best,
                 "epochs_ran": len(history),
                 "optimizer_steps": steps_total,
                 "wall_time_sec": round(time.time() - t_start, 1),
                 "epoch_history": history, **manifest_common}
        out_path = Path(args.b_tuning)
        merged = (json.loads(out_path.read_text())
                  if out_path.exists() else {})
        merged[str(args.r)] = entry
        out_path.write_text(json.dumps(merged, indent=1))
        print(json.dumps({"r": args.r, "best_epoch": best_epoch,
                          "best_val_regret": best, "wrote": str(out_path)}))
        return

    # final stage: fixed epochs from the B tuning JSON, no validation.
    b_tuning = json.loads(Path(args.b_tuning).read_text())
    epochs = int(b_tuning[str(args.r)]["best_epoch"])
    run_epochs = 1 if args.measure else epochs
    steps_total, t_start = 0, time.time()
    for epoch in range(1, run_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, opt, prep, epoch, args.batch_size,
                       args.device, train=True)
        steps_total += tr["optimizer_steps"]
        print(json.dumps({"epoch": epoch,
                          "sec": round(time.time() - t0, 1), "train": tr}))
    if args.measure:
        return
    probe_in, probe_out = probe_pair(model, prep, args.device)
    files = prep.bundle["manifest"]["files"]
    pt_path, mf_path = ckpt.save_checkpoint(
        Path(args.ckpt_dir), model, "b", args.fold, args.r,
        scale=prep.data.scale, price_mean=prep.data.price_mean,
        probe_input=probe_in, probe_output=probe_out,
        static_cardinalities=prep.data.static.cardinalities,
        event_cardinality=int(
            prep.bundle["encoder"]["cardinalities"]["event_type"]),
        train_meta={"stage": "final", "epochs": run_epochs,
                    "selection_metric": "fixed_epochs_from_b_tuning",
                    "optimizer_steps": steps_total,
                    "wall_time_sec": round(time.time() - t_start, 1),
                    **manifest_common},
        data_meta={
            "train_end": int(prep.bundle["fold_meta"]["train_end_idx"]),
            "bundle_sha256": files["fold_meta.json"]["sha256"],
            "scale_sha256": files["scalers.npz"]["sha256"],
            "encoder_hash": prep.bundle["encoder"]["hash"],
        },
        env_meta={**ckpt.environment_meta(args.device),
                  "git_commit": args.git_commit},
    )
    print(json.dumps({"checkpoint": str(pt_path),
                      "manifest": str(mf_path)}))


if __name__ == "__main__":
    main()
