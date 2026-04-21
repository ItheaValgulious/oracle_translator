from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.model import build_model


def _default_backbone() -> str:
    candidate = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-0___6B-Base"
    if candidate.exists():
        return str(candidate)
    return "Qwen/Qwen3-0.6B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=_default_backbone())
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = args.backbone
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(backbone_name=backbone).to(device)
    model.eval()
    batch = tokenizer(
        ["圣火昭昭, 涤荡前路.", "太阳第四课行星对应之秘, 在我命盘中点燃."],
        return_tensors="pt",
        padding=True,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        t1 = time.time()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        print("backbone", backbone)
        print("device", device)
        print("forward_seconds", time.time() - t1)
    print("status", outputs.status_logits.shape)
    print("status_confidence", outputs.status_confidence.shape)
    print("categorical heads", len(outputs.categorical_logits))
    print("binned heads", len(outputs.binned_logits))
    print("style heads", len(outputs.style_logits))


if __name__ == "__main__":
    main()
