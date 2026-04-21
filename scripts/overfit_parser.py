from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.io_utils import write_jsonl
from oracle_translator.train_eval import train_model


def _default_backbone() -> str:
    candidate = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-0___6B-Base"
    if candidate.exists():
        return str(candidate)
    return "Qwen/Qwen3-0.6B-Base"


def _load_success_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["status"] == "success" and row.get("runtime_b") is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data" / "processed" / "train_v1.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "overfit_success32"))
    parser.add_argument("--backbone", default=_default_backbone())
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--big-head", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    lora_target_modules = tuple(part.strip() for part in args.lora_target_modules.split(",") if part.strip())

    success_rows = _load_success_rows(Path(args.source))
    if args.subset_size > len(success_rows):
        raise ValueError(
            f"subset_size={args.subset_size} is larger than available success rows={len(success_rows)}"
        )

    rng = random.Random(args.seed)
    subset = rng.sample(success_rows, args.subset_size)
    output_dir = Path(args.output_dir)
    train_subset_path = output_dir / "train_overfit_subset.jsonl"
    val_subset_path = output_dir / "val_overfit_subset.jsonl"
    write_jsonl(train_subset_path, subset)
    write_jsonl(val_subset_path, subset)

    summary = train_model(
        train_path=str(train_subset_path),
        val_path=str(val_subset_path),
        output_dir=str(output_dir),
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        big_head=args.big_head,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        train_backbone=args.train_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
