"""Trainable v5 core/residual substrate utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

import torch
from torch import nn

from core_mem.v2.encoder import _lexical_features


BANK_LABELS = ("core", "residual")
WRITE_ACTIONS = ("new", "merge", "overwrite", "stale", "promote_to_core")


@dataclass(frozen=True)
class SubstrateExample:
    features: list[float]
    bank_label: int
    action_label: int


def _role_from_row(row: dict[str, Any]) -> str:
    return str(row.get("target_role") or row.get("masked_role") or "unknown").lower()


def _text_from_row(row: dict[str, Any]) -> str:
    return str(row.get("target_text", ""))


def _contains_any(text: str, terms: set[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    return bool(tokens & terms)


def infer_bank_label(row: dict[str, Any]) -> int:
    role = _role_from_row(row)
    text = _text_from_row(row).lower()
    if role == "system" or text.startswith("current user persona:"):
        return BANK_LABELS.index("core")
    if _contains_any(text, {"always", "usually", "love", "prefer", "favorite", "enjoy"}) and not _contains_any(
        text,
        {"recently", "now", "currently", "decided", "stopped", "started"},
    ):
        return BANK_LABELS.index("core")
    return BANK_LABELS.index("residual")


def infer_write_action(row: dict[str, Any]) -> int:
    role = _role_from_row(row)
    text = _text_from_row(row).lower()
    if role == "system" or text.startswith("current user persona:"):
        return WRITE_ACTIONS.index("promote_to_core")
    if _contains_any(text, {"stopped", "quit", "gave", "lost", "skip", "skipped", "no", "not", "never", "don't"}):
        return WRITE_ACTIONS.index("stale")
    if _contains_any(text, {"now", "currently", "recently", "decided", "started", "shift", "changed"}):
        return WRITE_ACTIONS.index("overwrite")
    if _contains_any(text, {"love", "prefer", "favorite", "enjoy", "appreciate", "passion"}):
        return WRITE_ACTIONS.index("merge")
    return WRITE_ACTIONS.index("new")


def featurize_substrate_row(row: dict[str, Any]) -> list[float]:
    role = _role_from_row(row)
    text = _text_from_row(row)
    role_features = [
        float(role == "system"),
        float(role == "user"),
        float(role == "assistant"),
        float(str(row.get("task")) == "masked_turn_reconstruction"),
    ]
    return [*role_features, *_lexical_features(text)]


def build_substrate_examples(rows: list[dict[str, Any]]) -> list[SubstrateExample]:
    return [
        SubstrateExample(
            features=featurize_substrate_row(row),
            bank_label=infer_bank_label(row),
            action_label=infer_write_action(row),
        )
        for row in rows
    ]


def _accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((predicted == labels).float().mean().item())


def _majority_accuracy(labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    values, counts = torch.unique(labels, return_counts=True)
    majority = values[int(torch.argmax(counts).item())]
    return _accuracy(torch.full_like(labels, int(majority.item())), labels)


def train_v5_substrate_heads(
    train_examples: list[SubstrateExample],
    eval_examples: list[SubstrateExample],
    *,
    device: str = "cpu",
    epochs: int = 80,
    lr: float = 0.05,
) -> dict[str, Any]:
    if not train_examples or not eval_examples:
        raise ValueError("v5 substrate training requires non-empty train and eval examples.")
    torch.manual_seed(52)
    target_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    input_dim = len(train_examples[0].features)
    bank_head = nn.Linear(input_dim, len(BANK_LABELS)).to(target_device)
    action_head = nn.Linear(input_dim, len(WRITE_ACTIONS)).to(target_device)
    optimizer = torch.optim.AdamW([*bank_head.parameters(), *action_head.parameters()], lr=lr)
    train_x = torch.tensor([example.features for example in train_examples], dtype=torch.float32, device=target_device)
    train_bank_y = torch.tensor([example.bank_label for example in train_examples], dtype=torch.long, device=target_device)
    train_action_y = torch.tensor([example.action_label for example in train_examples], dtype=torch.long, device=target_device)
    eval_x = torch.tensor([example.features for example in eval_examples], dtype=torch.float32, device=target_device)
    eval_bank_y = torch.tensor([example.bank_label for example in eval_examples], dtype=torch.long, device=target_device)
    eval_action_y = torch.tensor([example.action_label for example in eval_examples], dtype=torch.long, device=target_device)

    started = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        bank_loss = nn.functional.cross_entropy(bank_head(train_x), train_bank_y)
        action_loss = nn.functional.cross_entropy(action_head(train_x), train_action_y)
        loss = bank_loss + action_loss
        loss.backward()
        optimizer.step()
    train_seconds = time.perf_counter() - started

    with torch.no_grad():
        bank_pred = torch.argmax(bank_head(eval_x), dim=-1)
        action_pred = torch.argmax(action_head(eval_x), dim=-1)
    bank_accuracy = _accuracy(bank_pred, eval_bank_y)
    action_accuracy = _accuracy(action_pred, eval_action_y)
    bank_baseline = _majority_accuracy(eval_bank_y)
    action_baseline = _majority_accuracy(eval_action_y)
    covered_actions = sorted({WRITE_ACTIONS[example.action_label] for example in [*train_examples, *eval_examples]})
    return {
        "device": str(target_device),
        "train_seconds": train_seconds,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "bank_accuracy": bank_accuracy,
        "bank_majority_baseline_accuracy": bank_baseline,
        "bank_gain": bank_accuracy - bank_baseline,
        "action_accuracy": action_accuracy,
        "disabled_controller_accuracy": action_baseline,
        "action_gain": action_accuracy - action_baseline,
        "covered_actions": covered_actions,
        "positive_gain": bank_accuracy > bank_baseline and action_accuracy > action_baseline,
    }
