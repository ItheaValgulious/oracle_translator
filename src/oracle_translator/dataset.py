from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from slm.model_socket_schema import CATEGORICAL_SPECS, get_nested


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_model_socket(sample: dict) -> dict | None:
    socket = sample.get("model_socket")
    if socket is not None:
        return socket
    return None


class ModelSocketDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2id: dict[tuple[str, str], dict[str, int]] = {}
        self.num_classes: dict[tuple[str, str], int] = {}
        for spec in CATEGORICAL_SPECS:
            self.label2id[spec.path] = {label: i for i, label in enumerate(spec.labels)}
            self.num_classes[spec.path] = len(spec.labels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        text = rec["text"]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        socket = _extract_model_socket(rec)
        if socket is None:
            raise ValueError(f"missing model_socket in record {rec.get('id', idx)}")

        for spec in CATEGORICAL_SPECS:
            value = get_nested(socket, spec.path)
            if value is None or value not in self.label2id[spec.path]:
                raise ValueError(
                    f"invalid label '{value}' for {spec.path} in record {rec.get('id', idx)}"
                )
            item[spec.path] = torch.tensor(self.label2id[spec.path][value], dtype=torch.long)

        politeness = get_nested(socket, ("expression", "politeness"))
        if politeness is None:
            raise ValueError(f"missing politeness in record {rec.get('id', idx)}")
        item["politeness"] = torch.tensor(float(politeness), dtype=torch.float)

        return item


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "politeness": torch.stack([b["politeness"] for b in batch]),
    }
    for spec in CATEGORICAL_SPECS:
        collated[spec.path] = torch.stack([b[spec.path] for b in batch])
    return collated


def build_train_val_splits(
    *,
    data_paths: list[str],
    val_ratio: float = 0.1,
    rng_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    all_records: list[dict] = []
    for path in data_paths:
        raw = _load_jsonl(Path(path))
        for rec in raw:
            socket = _extract_model_socket(rec)
            if socket is None:
                continue
            try:
                float(get_nested(socket, ("expression", "politeness")))
            except (TypeError, ValueError):
                continue
            all_records.append(rec)

    rng = random.Random(rng_seed)
    rng.shuffle(all_records)
    split = max(1, int(len(all_records) * val_ratio))
    val = all_records[:split]
    train = all_records[split:]
    return train, val