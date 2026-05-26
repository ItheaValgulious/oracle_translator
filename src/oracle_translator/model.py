from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel

from slm.model_socket_schema import CATEGORICAL_SPECS


class OracleTranslatorModel(nn.Module):
    """Qwen3-0.6B backbone + classification heads for model socket prediction."""

    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen3-0.6B-Base",
        layer_mix_count: int = 4,
        dropout: float = 0.1,
        train_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        big_head: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.layer_mix_count = layer_mix_count
        self.dropout = nn.Dropout(dropout)

        total_layers = self.backbone.config.num_hidden_layers
        self.mix_layers = list(range(total_layers - layer_mix_count, total_layers))

        if not train_backbone:
            if unfreeze_last_n_layers > 0:
                for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                    for param in self.backbone.layers[layer_idx].parameters():
                        param.requires_grad = True
                for param in self.backbone.embed_tokens.parameters():
                    param.requires_grad = False
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        self.categorical_heads = nn.ModuleDict()
        self.num_classes: dict[tuple[str, str], int] = {}
        for spec in CATEGORICAL_SPECS:
            key = f"{spec.path[0]}_{spec.path[1]}"
            num_class = len(spec.labels)
            self.num_classes[spec.path] = num_class
            self.categorical_heads[key] = nn.Linear(hidden_size, num_class)

        self.politeness_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size // 2)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("output", nn.Linear(hidden_size // 2, 1)),
                ]
            )
        )

    def _pool_hidden_states(self, outputs, attention_mask):
        hidden_states = outputs.hidden_states
        selected = [hidden_states[i] for i in self.mix_layers]
        stacked = torch.stack(selected, dim=0)  # (layers, B, L, H)
        layer_mean = stacked.mean(dim=0)  # (B, L, H)

        expanded_mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        masked = layer_mean * expanded_mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = expanded_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = summed / counts  # (B, H)

        last_hidden = hidden_states[-1][:, -1, :]  # (B, H)
        return self.dropout(pooled + last_hidden)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        z_global = self._pool_hidden_states(outputs, attention_mask)

        logits: dict[str, torch.Tensor] = {}
        for spec in CATEGORICAL_SPECS:
            key = f"{spec.path[0]}_{spec.path[1]}"
            logits[key] = self.categorical_heads[key](z_global)

        politeness_logit = self.politeness_head(z_global).squeeze(-1)
        return logits, politeness_logit