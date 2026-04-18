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


def evaluate_model(
    model,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_records: list[dict[str, Any]] = []
    running_loss = 0.0
    batches = 0
    lookup = _ontology_lookup()
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss_bundle = compute_loss(outputs, batch)
            running_loss += float(loss_bundle.total.detach().cpu().item())
            batches += 1
            prediction_records.extend(build_prediction_records(batch, outputs, lookup))
    metrics = compute_epoch_metrics(prediction_records)
    metrics["loss"] = running_loss / max(1, batches)
    return metrics


def train_model(
    *,
    train_path: str,
    val_path: str,
    output_dir: str,
    backbone_name: str = "Qwen/Qwen3-0.6B-Base",
    batch_size: int = 2,
    epochs: int = 2,
    learning_rate: float = 2e-4,
    max_length: int = 128,
    weight_decay: float = 0.01,
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
        train_backbone=train_backbone,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
    ).to(device_obj)
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    best_metric = -1.0
    best_state_path = Path(output_dir) / "best_model.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        epoch_loss = 0.0
        steps = 0
        for batch in progress:
            batch = _move_batch(batch, device_obj)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss_bundle = compute_loss(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss_bundle.total.backward()
            optimizer.step()
            epoch_loss += float(loss_bundle.total.detach().cpu().item())
            steps += 1
            progress.set_postfix(loss=f"{epoch_loss / max(1, steps):.4f}")
        val_metrics = evaluate_model(model, val_loader, device=device_obj)
        train_loss = epoch_loss / max(1, steps)
        record = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history.append(record)
        # We select checkpoints by a simple parse-quality score instead of loss alone.
        score = 0.5 * val_metrics["status_accuracy"] + 0.5 * val_metrics["success_exact_match"]
        if score > best_metric:
            best_metric = score
            ensure_parent(best_state_path)
            torch.save(model.state_dict(), best_state_path)
    metrics_path = Path(output_dir) / "metrics.json"
    summary = {
        "backbone_name": backbone_name,
        "device": str(device_obj),
        "epochs": epochs,
        "batch_size": batch_size,
        "history": history,
        "best_model_path": str(best_state_path),
    }
    write_json(metrics_path, summary)
    return summary
