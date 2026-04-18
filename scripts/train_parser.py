from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.train_eval import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "processed" / "train_v1.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data" / "processed" / "val_v1.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "qwen06b_v1"))
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = train_model(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        train_backbone=args.train_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
