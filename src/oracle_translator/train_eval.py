from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset import SpellCollator, SpellDataset
from .io_utils import ensure_parent, write_json
from .losses import compute_loss
from .metrics import build_prediction_records, compute_epoch_metrics
from .model import build_model
from .ontology import (
    BINNED_SPECS,
    CATEGORICAL_SPECS,
    STATUS_LABELS,
    STYLE_SPECS,
)

LOSS_PART_KEYS = ("status", "categorical", "binned", "style", "reaction_mask")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _ontology_lookup() -> dict[str, list[str]]:
    lookup = {"status": STATUS_LABELS}
    for spec in CATEGORICAL_SPECS:
        lookup[".".join(spec.path)] = spec.labels
    for spec in BINNED_SPECS:
        lookup[".".join(spec.path)] = spec.labels
    for spec in STYLE_SPECS:
        lookup[".".join(spec.path)] = spec.labels
    return lookup


def _init_loss_part_totals() -> dict[str, float]:
    return {key: 0.0 for key in LOSS_PART_KEYS}


def _accumulate_loss_parts(totals: dict[str, float], parts: dict[str, float]) -> None:
    for key in LOSS_PART_KEYS:
        totals[key] += float(parts.get(key, 0.0))


def _average_loss_parts(totals: dict[str, float], count: int) -> dict[str, float]:
    denom = max(1, count)
    return {key: totals[key] / denom for key in LOSS_PART_KEYS}


def _format_loss_parts(parts: dict[str, float]) -> str:
    return ", ".join(f"{key}={parts[key]:.4f}" for key in LOSS_PART_KEYS)


def evaluate_model(
    model,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_records: list[dict[str, Any]] = []
    running_loss = 0.0
    loss_part_totals = _init_loss_part_totals()
    batches = 0
    lookup = _ontology_lookup()
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss_bundle = compute_loss(outputs, batch)
            running_loss += float(loss_bundle.total.detach().cpu().item())
            _accumulate_loss_parts(loss_part_totals, loss_bundle.parts)
            batches += 1
            prediction_records.extend(build_prediction_records(batch, outputs, lookup))
    metrics = compute_epoch_metrics(prediction_records)
    metrics["loss"] = running_loss / max(1, batches)
    metrics["loss_parts"] = _average_loss_parts(loss_part_totals, batches)
    return metrics


def train_model(
    *,
    train_path: str,
    val_path: str,
    output_dir: str,
    backbone_name: str = "Qwen/Qwen3-0.6B-Base",
    batch_size: int = 2,
    max_epochs: int = 300,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.0,
    learning_rate: float = 2e-4,
    max_length: int = 128,
    weight_decay: float = 0.01,
    big_head: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    train_backbone: bool = False,
    unfreeze_last_n_layers: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SpellDataset.from_jsonl(train_path)
    val_dataset = SpellDataset.from_jsonl(val_path)
    collator = SpellCollator(tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model = build_model(
        backbone_name=backbone_name,
        big_head=big_head,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        train_backbone=train_backbone,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
    ).to(device_obj)
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    best_state_path = Path(output_dir) / "best_model.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    stop_reason = f"reached max_epochs={max_epochs}"
    stopped_early = False
    for epoch in range(1, max_epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs}", leave=False)
        epoch_loss = 0.0
        train_loss_part_totals = _init_loss_part_totals()
        steps = 0
        for batch in progress:
            batch = _move_batch(batch, device_obj)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss_bundle = compute_loss(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss_bundle.total.backward()
            optimizer.step()
            epoch_loss += float(loss_bundle.total.detach().cpu().item())
            _accumulate_loss_parts(train_loss_part_totals, loss_bundle.parts)
            steps += 1
            progress.set_postfix(loss=f"{epoch_loss / max(1, steps):.4f}")
        val_metrics = evaluate_model(model, val_loader, device=device_obj)
        train_loss = epoch_loss / max(1, steps)
        train_loss_parts = _average_loss_parts(train_loss_part_totals, steps)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_parts": train_loss_parts,
            "val": val_metrics,
        }
        history.append(record)
        val_loss = float(val_metrics["loss"])
        tqdm.write(
            f"epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, status_acc={val_metrics['status_accuracy']:.4f}, "
            f"success_exact_match={val_metrics['success_exact_match']:.4f}"
        )
        tqdm.write(f"  train_parts: {_format_loss_parts(train_loss_parts)}")
        tqdm.write(f"  val_parts: {_format_loss_parts(val_metrics['loss_parts'])}")
        # Early stopping monitors val.loss directly and keeps the lowest-loss checkpoint.
        if best_val_loss - val_loss > early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            ensure_parent(best_state_path)
            torch.save(model.state_dict(), best_state_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                stop_reason = (
                    "val.loss did not improve for "
                    f"{early_stopping_patience} consecutive epochs"
                )
                break
    metrics_path = Path(output_dir) / "metrics.json"
    summary = {
        "backbone_name": backbone_name,
        "device": str(device_obj),
        "max_epochs": max_epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "big_head": big_head,
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": list(lora_target_modules),
        },
        "train_backbone": train_backbone,
        "unfreeze_last_n_layers": unfreeze_last_n_layers,
        "early_stopping": {
            "monitor": "val.loss",
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "best_model_path": str(best_state_path),
    }
    write_json(metrics_path, summary)
    return summary
