from __future__ import annotations

from typing import Any

import torch

from .dataset import IGNORE_INDEX


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_epoch_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    status_correct = 0
    status_total = 0
    exact_success = 0
    exact_success_total = 0
    categorical_correct = 0
    categorical_total = 0
    binned_correct = 0
    binned_total = 0
    style_abs_error = 0.0
    style_total = 0
    reaction_tp = 0.0
    reaction_fp = 0.0
    reaction_fn = 0.0
    for item in predictions:
        status_total += 1
        status_correct += int(item["status_pred"] == item["status_gold"])
        if item["status_gold"] == "success":
            exact_success_total += 1
            exact_success += int(item["all_success_fields_match"])
        for pred, gold, mask in item["categorical"]:
            if mask:
                categorical_total += 1
                categorical_correct += int(pred == gold)
        for pred, gold, mask in item["binned"]:
            if mask:
                binned_total += 1
                binned_correct += int(pred == gold)
        for pred, gold, mask in item["style"]:
            if mask:
                style_total += 1
                style_abs_error += abs(pred - gold)
        if item["reaction_valid"]:
            pred_vec = item["reaction_pred"]
            gold_vec = item["reaction_gold"]
            for pred_bit, gold_bit in zip(pred_vec, gold_vec, strict=True):
                if pred_bit and gold_bit:
                    reaction_tp += 1
                elif pred_bit and not gold_bit:
                    reaction_fp += 1
                elif gold_bit and not pred_bit:
                    reaction_fn += 1
    precision = _safe_div(reaction_tp, reaction_tp + reaction_fp)
    recall = _safe_div(reaction_tp, reaction_tp + reaction_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "status_accuracy": _safe_div(status_correct, status_total),
        "success_exact_match": _safe_div(exact_success, exact_success_total),
        "categorical_accuracy": _safe_div(categorical_correct, categorical_total),
        "binned_accuracy": _safe_div(binned_correct, binned_total),
        "style_mae": _safe_div(style_abs_error, style_total),
        "reaction_mask_f1": f1,
    }


def build_prediction_records(
    batch: dict[str, Any],
    outputs,
    ontology_lookup: dict[str, list[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    status_preds = outputs.status_logits.argmax(dim=-1).detach().cpu()
    status_golds = batch["status_labels"].detach().cpu()
    reaction_preds = (torch.sigmoid(outputs.reaction_mask_logits) > 0.5).int().detach().cpu()
    reaction_golds = batch["reaction_mask_targets"].int().detach().cpu()
    reaction_valid = batch["reaction_mask_valid"].int().detach().cpu()
    cat_preds = {name: logits.argmax(dim=-1).detach().cpu() for name, logits in outputs.categorical_logits.items()}
    bin_preds = {name: logits.argmax(dim=-1).detach().cpu() for name, logits in outputs.binned_logits.items()}
    style_preds = {name: logits.argmax(dim=-1).detach().cpu() for name, logits in outputs.style_logits.items()}
    batch_size = status_preds.size(0)
    for idx in range(batch_size):
        categorical_items: list[tuple[int, int, int]] = []
        binned_items: list[tuple[int, int, int]] = []
        style_items: list[tuple[int, int, int]] = []
        all_match = True
        for name, pred_tensor in cat_preds.items():
            gold = int(batch[f"cat::{name}"][idx].item())
            mask = int(batch[f"cat_mask::{name}"][idx].item())
            pred = int(pred_tensor[idx].item())
            categorical_items.append((pred, gold, mask))
            if mask and pred != gold:
                all_match = False
        for name, pred_tensor in bin_preds.items():
            gold = int(batch[f"bin::{name}"][idx].item())
            mask = int(batch[f"bin_mask::{name}"][idx].item())
            pred = int(pred_tensor[idx].item())
            binned_items.append((pred, gold, mask))
            if mask and pred != gold:
                all_match = False
        for name, pred_tensor in style_preds.items():
            gold = int(batch[f"style::{name}"][idx].item())
            mask = int(batch[f"style_mask::{name}"][idx].item())
            pred = int(pred_tensor[idx].item())
            style_items.append((pred, gold, mask))
        records.append(
            {
                "status_pred": ontology_lookup["status"][int(status_preds[idx].item())],
                "status_gold": ontology_lookup["status"][int(status_golds[idx].item())],
                "all_success_fields_match": all_match,
                "categorical": categorical_items,
                "binned": binned_items,
                "style": style_items,
                "reaction_pred": reaction_preds[idx].tolist(),
                "reaction_gold": reaction_golds[idx].tolist(),
                "reaction_valid": int(reaction_valid[idx].item()),
            }
        )
    return records
