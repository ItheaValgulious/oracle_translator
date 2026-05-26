from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.dataset import (
    CATEGORICAL_SPECS,
    ModelSocketDataset,
    build_train_val_splits,
    collate_fn,
)
from oracle_translator.model import OracleTranslatorModel

logger = logging.getLogger(__name__)


def _head_name(spec_path: tuple[str, str]) -> str:
    return f"{spec_path[0]}.{spec_path[1]}"


def compute_loss(
    model: OracleTranslatorModel,
    batch: dict[str, Any],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    categorical_logits, politeness_logit = model(input_ids, attention_mask)

    losses: dict[str, float] = {}
    total = torch.tensor(0.0, device=device)

    ce = nn.CrossEntropyLoss()
    for spec in CATEGORICAL_SPECS:
        name = _head_name(spec.path)
        labels = batch[spec.path].to(device)
        loss = ce(categorical_logits[name], labels)
        w = loss_weights.get(name, 1.0)
        total = total + w * loss
        losses[name] = loss.item()

    bce = nn.BCEWithLogitsLoss()
    politeness_labels = batch["politeness"].to(device)
    p_loss = bce(politeness_logit, politeness_labels)
    w = loss_weights.get("politeness", 1.0)
    total = total + w * p_loss
    losses["politeness"] = p_loss.item()

    return total, losses


@torch.no_grad()
def evaluate(
    model: OracleTranslatorModel,
    loader: DataLoader,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    per_head_losses: dict[str, float] = {}
    per_head_acc: dict[str, float] = {}
    politeness_mae = 0.0
    total_samples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        categorical_logits, politeness_logit = model(input_ids, attention_mask)

        ce = nn.CrossEntropyLoss()
        for spec in CATEGORICAL_SPECS:
            name = _head_name(spec.path)
            labels = batch[spec.path].to(device)
            loss = ce(categorical_logits[name], labels)
            w = loss_weights.get(name, 1.0)
            total_loss += w * loss.item()
            per_head_losses[name] = per_head_losses.get(name, 0.0) + loss.item()
            preds = categorical_logits[name].argmax(dim=-1)
            acc = (preds == labels).float().sum().item()
            per_head_acc[name] = per_head_acc.get(name, 0.0) + acc

        bce = nn.BCEWithLogitsLoss()
        politeness_labels = batch["politeness"].to(device)
        p_loss = bce(politeness_logit, politeness_labels)
        total_loss += loss_weights.get("politeness", 1.0) * p_loss.item()
        per_head_losses["politeness"] = per_head_losses.get("politeness", 0.0) + p_loss.item()
        politeness_preds = torch.sigmoid(politeness_logit)
        politeness_mae += (politeness_preds - politeness_labels).abs().sum().item()

        total_samples += len(batch["input_ids"])

    n_batches = max(1, len(loader))
    metrics = {
        "loss": total_loss / n_batches,
        "politeness_mae": politeness_mae / max(1, total_samples),
    }
    for name in per_head_losses:
        metrics[f"loss_{name}"] = per_head_losses[name] / n_batches
    for name in per_head_acc:
        metrics[f"acc_{name}"] = per_head_acc[name] / max(1, total_samples)

    model.train()
    return metrics


def train(config: dict, config_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    backbone_name = config["backbone_name"]
    max_length = config.get("max_length", 128)
    batch_size = config.get("batch_size", 2)
    max_epochs = config.get("max_epochs", 300)
    learning_rate = config.get("learning_rate", 2e-4)
    weight_decay = config.get("weight_decay", 0.01)
    rng_seed = config.get("rng_seed", 42)
    early_stopping_patience = config.get("early_stopping_patience", 3)
    early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0)
    val_ratio = config.get("val_ratio", 0.1)
    layer_mix_count = config.get("layer_mix_count", 4)
    dropout = config.get("dropout", 0.1)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    output_dir = Path(config.get("output_dir", str(ROOT / "artifacts")))

    loss_weights: dict[str, float] = {}
    for spec in CATEGORICAL_SPECS:
        loss_weights[_head_name(spec.path)] = config.get("loss_weights", {}).get(
            _head_name(spec.path), 1.0
        )
    loss_weights["politeness"] = config.get("loss_weights", {}).get("politeness", 1.0)

    torch.manual_seed(rng_seed)

    data_paths = config["dataset"]["source_paths"]
    train_records, val_records = build_train_val_splits(
        data_paths=data_paths,
        val_ratio=val_ratio,
        rng_seed=rng_seed,
    )
    logger.info(f"train samples: {len(train_records)}, val samples: {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ModelSocketDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = ModelSocketDataset(val_records, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = OracleTranslatorModel(
        backbone_name=backbone_name,
        layer_mix_count=layer_mix_count,
        dropout=dropout,
        train_backbone=config.get("train_backbone", True),
        unfreeze_last_n_layers=config.get("unfreeze_last_n_layers", 0),
        big_head=config.get("big_head", False),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            loss, head_losses = compute_loss(model, batch, loss_weights)
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        val_metrics = evaluate(model, val_loader, loss_weights)
        val_loss = val_metrics["loss"]
        logger.info(
            f"epoch {epoch} | train_loss: {epoch_loss / max(1, len(train_loader)):.4f}"
            f" | val_loss: {val_loss:.4f} | val_politeness_mae: {val_metrics['politeness_mae']:.4f}"
        )

        improved = val_loss < best_loss - early_stopping_min_delta
        if improved:
            best_loss = val_loss
            patience_counter = 0
            save_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                save_path,
            )
            logger.info(f"saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"early stopping after {epoch} epochs")
                break

    logger.info(f"training finished, best val_loss: {best_loss:.4f}")


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "parser_v1.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config, config_path)


if __name__ == "__main__":
    main()