from __future__ import annotations

from dataclasses import dataclass
import math
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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return self.base(x) + delta

    def lora_parameters(self):
        yield from self.lora_a.parameters()
        yield from self.lora_b.parameters()


class MLPHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout: float,
        *,
        hidden_multiplier: int = 1,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        expanded_hidden = hidden_size * hidden_multiplier
        layers: list[nn.Module] = []
        input_size = hidden_size
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_size, expanded_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = expanded_hidden
        layers.append(nn.Linear(input_size, output_size))
        self.net = nn.Sequential(*layers)

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
        big_head: bool = False,
        train_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
        # We only need token hidden states, so we load the backbone model rather than a full LM head.
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=trust_remote_code,
        )
        self.hidden_size = config.hidden_size
        self.head_hidden_multiplier = 4 if big_head else 1
        self.head_hidden_layers = 2 if big_head else 1
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_module_names: list[str] = []
        self.layer_mix = nn.Parameter(torch.zeros(4))
        self.global_pool = AttnPool(self.hidden_size)
        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.status_head = MLPHead(
            self.hidden_size,
            3,
            dropout,
            hidden_multiplier=self.head_hidden_multiplier,
            hidden_layers=self.head_hidden_layers,
        )
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
            self.categorical_heads[module_name] = MLPHead(
                self.hidden_size,
                len(spec.labels),
                dropout,
                hidden_multiplier=self.head_hidden_multiplier,
                hidden_layers=self.head_hidden_layers,
            )
        for spec in BINNED_SPECS:
            public_name = _key(spec.path)
            module_name = _module_key(spec.path)
            self.binned_name_map[public_name] = module_name
            self.slot_pools[module_name] = SlotAttentionPool(self.hidden_size)
            self.binned_heads[module_name] = MLPHead(
                self.hidden_size,
                len(spec.labels),
                dropout,
                hidden_multiplier=self.head_hidden_multiplier,
                hidden_layers=self.head_hidden_layers,
            )
        for spec in STYLE_SPECS:
            public_name = _key(spec.path)
            module_name = _module_key(spec.path)
            self.style_name_map[public_name] = module_name
            self.slot_pools[module_name] = SlotAttentionPool(self.hidden_size)
            self.style_heads[module_name] = MLPHead(
                self.hidden_size,
                len(spec.labels),
                dropout,
                hidden_multiplier=self.head_hidden_multiplier,
                hidden_layers=self.head_hidden_layers,
            )
        self.reaction_mask_pool = SlotAttentionPool(self.hidden_size)
        self.reaction_mask_head = MLPHead(
            self.hidden_size,
            len(REACTION_MASK_LABELS),
            dropout,
            hidden_multiplier=self.head_hidden_multiplier,
            hidden_layers=self.head_hidden_layers,
        )
        if self.lora_rank > 0:
            self._apply_lora()
        self._configure_training(train_backbone=train_backbone, unfreeze_last_n_layers=unfreeze_last_n_layers)

    def _locate_backbone_layers(self):
        candidates = [
            getattr(self.backbone, "layers", None),
            getattr(getattr(self.backbone, "model", None), "layers", None),
        ]
        for layers in candidates:
            if layers is not None:
                return layers
        raise RuntimeError("cannot locate backbone layers for partial unfreeze")

    def _replace_backbone_module(self, module_name: str, replacement: nn.Module) -> None:
        parent = self.backbone
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)

    def _apply_lora(self) -> None:
        for module_name, module in list(self.backbone.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(module_name.endswith(target) for target in self.lora_target_modules):
                continue
            replacement = LoRALinear(
                module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            self._replace_backbone_module(module_name, replacement)
            self.lora_module_names.append(module_name)
        if not self.lora_module_names:
            raise RuntimeError(
                f"no backbone linear modules matched lora targets: {self.lora_target_modules}"
            )

    def _enable_lora_training(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                for param in module.lora_parameters():
                    param.requires_grad = True

    def _configure_training(self, *, train_backbone: bool, unfreeze_last_n_layers: int) -> None:
        if train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self._enable_lora_training()
            return
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_last_n_layers <= 0:
            self._enable_lora_training()
            return
        layers = self._locate_backbone_layers()
        for layer in layers[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self._enable_lora_training()

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
