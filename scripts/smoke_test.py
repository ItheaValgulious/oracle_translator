from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.model import build_model


def main() -> None:
    backbone = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(backbone_name=backbone)
    model.eval()
    batch = tokenizer(
        ["圣火昭昭, 涤荡前路.", "太阳第四课行星对应之秘, 在我命盘中点燃."],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        t1=time.time()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        print(time.time()-t1)
    print("status", outputs.status_logits.shape)
    print("status_confidence", outputs.status_confidence.shape)
    print("categorical heads", len(outputs.categorical_logits))
    print("binned heads", len(outputs.binned_logits))
    print("style heads", len(outputs.style_logits))


if __name__ == "__main__":
    main()
