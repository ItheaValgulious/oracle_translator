from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .model import ParserModelOutput


@dataclass
class LossBundle:
    total: torch.Tensor
    parts: dict[str, float]


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_item = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_item * mask).sum() / denom


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_item = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_item * mask).sum() / denom


def compute_loss(
    outputs: ParserModelOutput,
    batch: dict[str, Any],
    *,
    status_weight: float = 1.0,
    categorical_weight: float = 1.0,
    binned_weight: float = 0.8,
    style_weight: float = 0.5,
    reaction_mask_weight: float = 0.7,
) -> LossBundle:
    losses: dict[str, torch.Tensor] = {}
    losses["status"] = F.cross_entropy(outputs.status_logits, batch["status_labels"])
    categorical_losses: list[torch.Tensor] = []
    for name, logits in outputs.categorical_logits.items():
        categorical_losses.append(_masked_ce(logits, batch[f"cat::{name}"], batch[f"cat_mask::{name}"]))
    losses["categorical"] = torch.stack(categorical_losses).mean() if categorical_losses else outputs.status_logits.new_tensor(0.0)
    binned_losses: list[torch.Tensor] = []
    for name, logits in outputs.binned_logits.items():
        binned_losses.append(_masked_ce(logits, batch[f"bin::{name}"], batch[f"bin_mask::{name}"]))
    losses["binned"] = torch.stack(binned_losses).mean() if binned_losses else outputs.status_logits.new_tensor(0.0)
    style_losses: list[torch.Tensor] = []
    for name, logits in outputs.style_logits.items():
        style_losses.append(_masked_ce(logits, batch[f"style::{name}"], batch[f"style_mask::{name}"]))
    losses["style"] = torch.stack(style_losses).mean() if style_losses else outputs.status_logits.new_tensor(0.0)
    losses["reaction_mask"] = _masked_bce(
        outputs.reaction_mask_logits,
        batch["reaction_mask_targets"],
        batch["reaction_mask_valid"],
    )
    # First-stage objective focuses on reliable slot learning. Confidence is derived at inference time.
    total = (
        status_weight * losses["status"]
        + categorical_weight * losses["categorical"]
        + binned_weight * losses["binned"]
        + style_weight * losses["style"]
        + reaction_mask_weight * losses["reaction_mask"]
    )
    return LossBundle(
        total=total,
        parts={key: float(value.detach().cpu().item()) for key, value in losses.items()},
    )
