"""TSMixer-Ext training on Colab (ST4-05). Bundle-only: no database
access in this import chain; torch loads freely.

Budget discipline (ST4-04 approval condition, run_chronos lineage):
measure BEFORE committing --
  python -m scripts.train_tsmixer_colab --bundle <fold_01> --model a1 \
      --fold 1 --stage final --measure
runs 2 epochs and extrapolates minutes per checkpoint and for all 16
Stage 4 checkpoints at the configured --epochs. If the extrapolated
budget exceeds the free-runtime allowance, STOP and report; a reduced
protocol must be fixed in advance, never chosen after seeing results.

Colab setup (one notebook cell):
  from google.colab import drive; drive.mount('/content/drive')
  !unzip -o /content/drive/MyDrive/ct_agent/tsmixer_code.zip -d /content
  %cd /content
then run this module with --bundle pointing into Drive and --out on
Drive so checkpoints and manifests persist (ST4-04: Drive is the durable
store; git keeps only a sha index).

A2 and (Stage 5) B require --r in {3, 5}; A1/A3 refuse --r. Evaluation
at r=10 reuses the r=5 checkpoint and is never trained (enforced).
"""

import argparse
from pathlib import Path

import torch

from core_pipeline.forecast.tsmixer.config import (
    TSMixerConfig,
    checkpoint_key,
    input_schema_hash,
)
from core_pipeline.forecast.tsmixer.staging import load_bundle
from core_pipeline.forecast.tsmixer.train import train_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True,
                        help="path to one fold_XX bundle directory")
    parser.add_argument("--out", default="outputs/checkpoints")
    parser.add_argument("--model", required=True,
                        choices=["a1", "a2", "a3", "b"])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--stage", choices=["final", "tune"],
                        default="final")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epochs-json", default=None,
                        help="tuning_results.json; overrides --epochs with "
                             "epochs_by_key[<checkpoint key>] (ST4-07 T3)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()

    key = checkpoint_key(args.model, args.r)  # validates model/r pairing
    if args.epochs_json:
        import json
        tuned = json.loads(Path(args.epochs_json).read_text())
        args.epochs = int(tuned["epochs_by_key"][key]["best_epoch"])
        print(f"epochs from tuning: {args.epochs} ({key})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_bundle(
        Path(args.bundle),
        expected_input_schema_hash=input_schema_hash())
    if int(bundle["fold_meta"]["fold"]) != args.fold:
        raise ValueError(
            f"bundle is fold {bundle['fold_meta']['fold']}, "
            f"requested {args.fold}")
    print(f"{key} fold {args.fold} stage {args.stage} device {device} "
          f"encoder {bundle['encoder']['hash']}")
    train_one(bundle, args.model, args.fold, args.r, args.stage,
              TSMixerConfig(), Path(args.out), epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              global_seed=args.global_seed, device=device,
              patience=args.patience, git_commit=args.git_commit,
              measure_only=args.measure)


if __name__ == "__main__":
    main()
