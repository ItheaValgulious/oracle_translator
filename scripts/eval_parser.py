from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.dataset import SpellCollator, SpellDataset
from oracle_translator.io_utils import write_json
from oracle_translator.model import build_model
from oracle_translator.train_eval import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default=str(ROOT / "data" / "processed" / "val_v1.jsonl"))
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "qwen06b_v1" / "best_model.pt"))
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "qwen06b_v1" / "eval_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SpellDataset.from_jsonl(args.val)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=SpellCollator(tokenizer, max_length=args.max_length))
    model = build_model(backbone_name=args.backbone).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    metrics = evaluate_model(model, dataloader, device=device)
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
