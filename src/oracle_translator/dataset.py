from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .io_utils import read_jsonl
from .ontology import (
    BINNED_SPECS,
    CATEGORICAL_SPECS,
    REACTION_MASK_LABELS,
    STATUS_LABELS,
    STYLE_SPECS,
    get_nested,
)


IGNORE_INDEX = -100


@dataclass
class EncodedSample:
    text: str
    status_id: int
    categorical: dict[str, int]
    categorical_mask: dict[str, int]
    binned: dict[str, int]
    binned_mask: dict[str, int]
    style: dict[str, int]
    style_mask: dict[str, int]
    reaction_mask: list[int]
    reaction_mask_valid: int
    raw: dict[str, Any]


def _key(path: tuple[str, str]) -> str:
    return ".".join(path)


def _encode_with_mask(value: str | None, labels: list[str]) -> tuple[int, int]:
    if value is None:
        return IGNORE_INDEX, 0
    return labels.index(value), 1


def encode_row(row: dict[str, Any]) -> EncodedSample:
    status_id = STATUS_LABELS.index(row["status"])
    runtime_b = row.get("runtime_b")
    categorical: dict[str, int] = {}
    categorical_mask: dict[str, int] = {}
    for spec in CATEGORICAL_SPECS:
        # Success rows supervise semantic slots. Unstable/backfire rows only supervise status/style.
        value = get_nested(runtime_b, spec.path)
        if row["status"] != "success":
            value = None
        encoded, mask = _encode_with_mask(value, spec.labels)
        categorical[_key(spec.path)] = encoded
        categorical_mask[_key(spec.path)] = mask
    binned: dict[str, int] = {}
    binned_mask: dict[str, int] = {}
    for spec in BINNED_SPECS:
        value = get_nested(runtime_b, spec.path)
        if row["status"] != "success":
            value = None
        encoded, mask = _encode_with_mask(value, spec.labels)
        binned[_key(spec.path)] = encoded
        binned_mask[_key(spec.path)] = mask
    style: dict[str, int] = {}
    style_mask: dict[str, int] = {}
    for spec in STYLE_SPECS:
        value = None
        if runtime_b is not None:
            value = get_nested(runtime_b, spec.path)
        else:
            meta_expression = row.get("meta", {}).get("expression", {})
            value = meta_expression.get(spec.path[1])
        encoded, mask = _encode_with_mask(value, spec.labels)
        style[_key(spec.path)] = encoded
        style_mask[_key(spec.path)] = mask
    reaction_mask = [0] * len(REACTION_MASK_LABELS)
    reaction_mask_valid = 0
    if runtime_b is not None and row["status"] == "success":
        # reaction_mask is multi-label, so we materialize a dense bit vector here.
        labels = get_nested(runtime_b, ("subject", "reaction_mask"))
        if labels is not None:
            reaction_mask_valid = 1
            for item in labels:
                reaction_mask[REACTION_MASK_LABELS.index(item)] = 1
    return EncodedSample(
        text=row["text"],
        status_id=status_id,
        categorical=categorical,
        categorical_mask=categorical_mask,
        binned=binned,
        binned_mask=binned_mask,
        style=style,
        style_mask=style_mask,
        reaction_mask=reaction_mask,
        reaction_mask_valid=reaction_mask_valid,
        raw=row,
    )


class SpellDataset(Dataset[EncodedSample]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.encoded = [encode_row(row) for row in rows]

    @classmethod
    def from_jsonl(cls, path: str) -> "SpellDataset":
        return cls(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.encoded[index]


class SpellCollator:
    def __init__(self, tokenizer, *, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[EncodedSample]) -> dict[str, Any]:
        texts = [item.text for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Keep a flat tensor dictionary so training code can stay model-agnostic.
        result: dict[str, Any] = {
            "texts": texts,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "status_labels": torch.tensor([item.status_id for item in batch], dtype=torch.long),
            "reaction_mask_targets": torch.tensor([item.reaction_mask for item in batch], dtype=torch.float32),
            "reaction_mask_valid": torch.tensor([item.reaction_mask_valid for item in batch], dtype=torch.float32),
            "raw": [item.raw for item in batch],
        }
        for key in batch[0].categorical:
            result[f"cat::{key}"] = torch.tensor([item.categorical[key] for item in batch], dtype=torch.long)
            result[f"cat_mask::{key}"] = torch.tensor(
                [item.categorical_mask[key] for item in batch], dtype=torch.float32
            )
        for key in batch[0].binned:
            result[f"bin::{key}"] = torch.tensor([item.binned[key] for item in batch], dtype=torch.long)
            result[f"bin_mask::{key}"] = torch.tensor([item.binned_mask[key] for item in batch], dtype=torch.float32)
        for key in batch[0].style:
            result[f"style::{key}"] = torch.tensor([item.style[key] for item in batch], dtype=torch.long)
            result[f"style_mask::{key}"] = torch.tensor([item.style_mask[key] for item in batch], dtype=torch.float32)
        return result
