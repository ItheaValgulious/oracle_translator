from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.model import build_model
from oracle_translator.ontology import (
    BINNED_SPECS,
    CATEGORICAL_SPECS,
    REACTION_MASK_LABELS,
    STATUS_LABELS,
    STYLE_SPECS,
    set_nested,
)


def _default_backbone() -> str:
    candidate = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-0___6B-Base"
    if candidate.exists():
        return str(candidate)
    return "Qwen/Qwen3-0.6B-Base"


def _default_checkpoint() -> str:
    preferred = ROOT / "artifacts" / "qwen06b_v1_val_loss_es" / "best_model.pt"
    if preferred.exists():
        return str(preferred)
    checkpoints = sorted(
        ROOT.glob("artifacts/**/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return str(checkpoints[0])
    return str(preferred)


def _load_runtime(
    *,
    checkpoint: str,
    backbone: str,
    device: str | None,
) -> tuple[AutoTokenizer, torch.nn.Module, torch.device]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(backbone_name=backbone).to(device_obj)
    state_dict = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return tokenizer, model, device_obj


def _decode_label(logits: torch.Tensor, labels: list[str]) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    best_index = int(probs.argmax(dim=-1).item())
    return labels[best_index], float(probs[best_index].item())


def _predict_text(
    *,
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    reaction_threshold: float,
) -> dict[str, Any]:
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])

    status_probs = torch.softmax(outputs.status_logits[0], dim=-1).detach().cpu()
    status_index = int(status_probs.argmax(dim=-1).item())
    status_label = STATUS_LABELS[status_index]
    runtime_preview: dict[str, Any] = {
        "subject_kind": "summon_material",
        "subject": {},
        "release": {},
        "motion": {},
        "targeting": {},
        "expression": {},
    }
    field_confidence: dict[str, float] = {}

    for spec in CATEGORICAL_SPECS:
        public_name = ".".join(spec.path)
        label, confidence = _decode_label(outputs.categorical_logits[public_name][0], spec.labels)
        set_nested(runtime_preview, spec.path, label)
        field_confidence[public_name] = confidence

    for spec in BINNED_SPECS:
        public_name = ".".join(spec.path)
        label, confidence = _decode_label(outputs.binned_logits[public_name][0], spec.labels)
        set_nested(runtime_preview, spec.path, label)
        field_confidence[public_name] = confidence

    for spec in STYLE_SPECS:
        public_name = ".".join(spec.path)
        label, confidence = _decode_label(outputs.style_logits[public_name][0], spec.labels)
        set_nested(runtime_preview, spec.path, label)
        field_confidence[public_name] = confidence

    reaction_probs = torch.sigmoid(outputs.reaction_mask_logits[0]).detach().cpu()
    reaction_mask = [
        label
        for label, score in zip(REACTION_MASK_LABELS, reaction_probs.tolist(), strict=True)
        if score >= reaction_threshold
    ]
    runtime_preview["subject"]["reaction_mask"] = reaction_mask

    low_confidence_fields = {
        name: score
        for name, score in field_confidence.items()
        if score < 0.6
    }
    preview_is_reliable = status_label == "success"

    return {
        "text": text,
        "status": status_label,
        "status_confidence": float(outputs.status_confidence[0].detach().cpu().item()),
        "status_probabilities": {
            label: float(score)
            for label, score in zip(STATUS_LABELS, status_probs.tolist(), strict=True)
        },
        "ready_to_cast": status_label == "success",
        "preview_is_reliable": preview_is_reliable,
        "runtime_preview": runtime_preview,
        "field_confidence": field_confidence,
        "low_confidence_fields": low_confidence_fields,
        "reaction_mask_scores": {
            label: float(score)
            for label, score in zip(REACTION_MASK_LABELS, reaction_probs.tolist(), strict=True)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=_default_checkpoint())
    parser.add_argument("--backbone", default=_default_backbone())
    parser.add_argument("--text", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--reaction-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    tokenizer, model, device = _load_runtime(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        device=args.device,
    )
    print(f"checkpoint: {args.checkpoint}")
    print(f"backbone: {args.backbone}")
    print(f"device: {device}")

    if args.text is not None:
        result = _predict_text(
            text=args.text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            reaction_threshold=args.reaction_threshold,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("enter spell text. type 'exit' or 'quit' to stop.")
    while True:
        try:
            text = input("spell> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q"}:
            break
        result = _predict_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            reaction_threshold=args.reaction_threshold,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
