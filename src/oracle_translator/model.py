from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .ontology import BINNED_SPECS, CATEGORICAL_SPECS, REACTION_MASK_LABELS, STYLE_SPECS


def _key(path: tuple[str, str]) -> str:
    return ".".join(path)


def _module_key(path: tuple[str, str]) -> str:
    return "__".join(path)


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bth,h->bt", hidden_states, self.query)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,bth->bh", weights, hidden_states)


class SlotAttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bth,h->bt", hidden_states, self.query)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,bth->bh", weights, hidden_states)


class MLPHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ParserModelOutput:
    status_logits: torch.Tensor
    status_confidence: torch.Tensor
    categorical_logits: dict[str, torch.Tensor]
    binned_logits: dict[str, torch.Tensor]
    style_logits: dict[str, torch.Tensor]
    reaction_mask_logits: torch.Tensor


class SpellParserModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen3-0.6B-Base",
        *,
        trust_remote_code: bool = True,
        dropout: float = 0.1,
        train_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
        # We only need token hidden states, so we load the backbone model rather than a full LM head.
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=trust_remote_code,
        )
        self.hidden_size = config.hidden_size
        self.layer_mix = nn.Parameter(torch.zeros(4))
        self.global_pool = AttnPool(self.hidden_size)
        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.status_head = MLPHead(self.hidden_size, 3, dropout)
        self.slot_pools = nn.ModuleDict()
        self.categorical_heads = nn.ModuleDict()
        self.binned_heads = nn.ModuleDict()
        self.style_heads = nn.ModuleDict()
        self.categorical_name_map: dict[str, str] = {}
        self.binned_name_map: dict[str, str] = {}
        self.style_name_map: dict[str, str] = {}
        for spec in CATEGORICAL_SPECS:
            public_name = _key(spec.path)
            module_name = _module_key(spec.path)
            self.categorical_name_map[public_name] = module_name
            self.slot_pools[module_name] = SlotAttentionPool(self.hidden_size)
            self.categorical_heads[module_name] = MLPHead(self.hidden_size, len(spec.labels), dropout)
        for spec in BINNED_SPECS:
            public_name = _key(spec.path)
            module_name = _module_key(spec.path)
            self.binned_name_map[public_name] = module_name
            self.slot_pools[module_name] = SlotAttentionPool(self.hidden_size)
            self.binned_heads[module_name] = MLPHead(self.hidden_size, len(spec.labels), dropout)
        for spec in STYLE_SPECS:
            public_name = _key(spec.path)
            module_name = _module_key(spec.path)
            self.style_name_map[public_name] = module_name
            self.slot_pools[module_name] = SlotAttentionPool(self.hidden_size)
            self.style_heads[module_name] = MLPHead(self.hidden_size, len(spec.labels), dropout)
        self.reaction_mask_pool = SlotAttentionPool(self.hidden_size)
        self.reaction_mask_head = MLPHead(self.hidden_size, len(REACTION_MASK_LABELS), dropout)
        self._configure_training(train_backbone=train_backbone, unfreeze_last_n_layers=unfreeze_last_n_layers)

    def _configure_training(self, *, train_backbone: bool, unfreeze_last_n_layers: int) -> None:
        if train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_last_n_layers <= 0:
            return
        layers = None
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            layers = self.backbone.model.layers
        if layers is None:
            raise RuntimeError("cannot locate backbone layers for partial unfreeze")
        for layer in layers[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _mix_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # A small learned layer mix is cheaper than adding another large fusion block.
        weights = torch.softmax(self.layer_mix, dim=0)
        selected = hidden_states[-4:]
        mixed = sum(weight * tensor for weight, tensor in zip(weights, selected, strict=True))
        return mixed

    @staticmethod
    def _derive_status_confidence(status_logits: torch.Tensor) -> torch.Tensor:
        # First-stage confidence is derived from status separation, not separately supervised.
        probs = torch.softmax(status_logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        max_prob = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        return 0.5 * max_prob + 0.5 * margin

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ParserModelOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = self._mix_hidden_states(outputs.hidden_states).float()
        last_indices = attention_mask.sum(dim=-1).clamp_min(1) - 1
        last_hidden = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]
        pooled_hidden = self.global_pool(hidden_states, attention_mask)
        global_hidden = self.global_proj(torch.cat([last_hidden, pooled_hidden], dim=-1))
        categorical_logits = {}
        for public_name, module_name in self.categorical_name_map.items():
            categorical_logits[public_name] = self.categorical_heads[module_name](
                self.slot_pools[module_name](hidden_states, attention_mask)
            )
        binned_logits = {}
        for public_name, module_name in self.binned_name_map.items():
            binned_logits[public_name] = self.binned_heads[module_name](
                self.slot_pools[module_name](hidden_states, attention_mask)
            )
        style_logits = {}
        for public_name, module_name in self.style_name_map.items():
            style_logits[public_name] = self.style_heads[module_name](
                self.slot_pools[module_name](hidden_states, attention_mask)
            )
        reaction_mask_logits = self.reaction_mask_head(self.reaction_mask_pool(hidden_states, attention_mask))
        status_logits = self.status_head(global_hidden)
        return ParserModelOutput(
            status_logits=status_logits,
            status_confidence=self._derive_status_confidence(status_logits),
            categorical_logits=categorical_logits,
            binned_logits=binned_logits,
            style_logits=style_logits,
            reaction_mask_logits=reaction_mask_logits,
        )


def build_model(**kwargs: Any) -> SpellParserModel:
    return SpellParserModel(**kwargs)
