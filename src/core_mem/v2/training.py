"""Training utilities for stage-2 direct-train readiness."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core_mem.v2.experiments import dataset_allowed_for_variant
from core_mem.v2.semantic_outputs import (
    coerce_task_payload,
    json_like_payload,
    render_task_payload,
    semantic_task_scores,
)

SLOT_ASSIGNMENT_TASK_NAME = "lifecycle_prediction"
DEFAULT_TASK_ADAPTER_NAMES = {
    "retrieval_alignment": "retrieval_adapter",
    SLOT_ASSIGNMENT_TASK_NAME: "write_adapter",
    "composition_to_belief": "belief_adapter",
    "slot_autoencoding": "autoencoding_adapter",
}
_RELATION_HINT_RE = re.compile(r'"(?P<relation>[A-Za-z][A-Za-z0-9_]+)"')


def task_adapter_settings(
    config: dict[str, Any],
    *,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    configured_tasks = tasks or list(config.get("training", {}).get("tasks", [])) or list(DEFAULT_TASK_ADAPTER_NAMES)
    task_adapters = (config.get("model", {}) or {}).get("task_adapters", {}) or {}
    raw_names = task_adapters.get("names", {}) or {}
    adapter_names = {
        task_name: str(raw_names.get(task_name) or DEFAULT_TASK_ADAPTER_NAMES.get(task_name) or f"{task_name}_adapter")
        for task_name in configured_tasks
    }
    return {
        "enabled": bool(task_adapters.get("enabled", False)),
        "names": adapter_names,
    }


@dataclass(frozen=True)
class TrainingExample:
    task_name: str
    input_text: str
    target_text: str


def load_prepared_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_training_examples(prepared_manifest_path: Path, tasks: list[str] | None = None) -> list[TrainingExample]:
    manifest = load_prepared_manifest(prepared_manifest_path)
    requested = set(tasks or manifest["task_files"].keys())
    examples: list[TrainingExample] = []
    for task_name, file_path in manifest["task_files"].items():
        if task_name not in requested:
            continue
        for row in load_jsonl(Path(file_path)):
            examples.append(serialize_task_example(task_name, row))
    return examples


def build_training_examples_with_variant(
    prepared_manifest_path: Path,
    *,
    tasks: list[str] | None = None,
    disabled_pools: list[str] | None = None,
    online_aligned: bool = False,
) -> list[TrainingExample]:
    manifest = load_prepared_manifest(prepared_manifest_path)
    requested = set(tasks or manifest["task_files"].keys())
    examples: list[TrainingExample] = []
    for task_name, file_path in manifest["task_files"].items():
        if task_name not in requested:
            continue
        for row in load_jsonl(Path(file_path)):
            dataset_name = str((row.get("_meta", {}) or {}).get("dataset", "unknown"))
            if not dataset_allowed_for_variant(dataset_name, disabled_pools):
                continue
            examples.append(serialize_task_example(task_name, row))
    if online_aligned:
        examples = _apply_online_alignment(examples)
    return examples


def _balanced_cap_examples(
    examples: list[TrainingExample],
    *,
    max_examples: int | None,
    task_weights: dict[str, int] | None = None,
) -> list[TrainingExample]:
    if max_examples is None or len(examples) <= max_examples:
        return examples

    grouped: dict[str, list[TrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.task_name, []).append(example)

    weights = {task: max(int((task_weights or {}).get(task, 1)), 1) for task in grouped}
    cycle: list[str] = []
    for task_name in grouped:
        cycle.extend([task_name] * weights[task_name])

    offsets = {task_name: 0 for task_name in grouped}
    selected: list[TrainingExample] = []
    while len(selected) < max_examples:
        progressed = False
        for task_name in cycle:
            index = offsets[task_name]
            task_examples = grouped[task_name]
            if index >= len(task_examples):
                continue
            selected.append(task_examples[index])
            offsets[task_name] = index + 1
            progressed = True
            if len(selected) >= max_examples:
                break
        if not progressed:
            break
    return selected


def _apply_online_alignment(examples: list[TrainingExample]) -> list[TrainingExample]:
    prioritized = {"retrieval_alignment", "composition_to_belief", SLOT_ASSIGNMENT_TASK_NAME}
    aligned: list[TrainingExample] = []
    for example in examples:
        aligned.append(example)
        if example.task_name in prioritized:
            aligned.append(example)
    return aligned


def compact_observation_payload(observation: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "entity",
        "relation",
        "value",
        "value_type",
        "time_scope",
        "status_hint",
        "canonical_gloss",
    ]
    return {key: observation.get(key) for key in keys if key in observation}


def compact_slot_payload(slot: dict[str, Any]) -> dict[str, Any]:
    role_scores = slot.get("soft_role_scores") or {}
    top_role = None
    if isinstance(role_scores, dict) and role_scores:
        top_role = max(role_scores.items(), key=lambda item: float(item[1]))[0]
    keys = [
        "slot_id",
        "bank",
        "entity",
        "relation",
        "canonical_gloss",
        "confidence",
        "active_flag",
        "revision_count",
    ]
    payload = {key: slot.get(key) for key in keys if key in slot}
    if top_role is not None:
        payload["dominant_role"] = top_role
    return payload


def compact_slot_list(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_slot_payload(slot) for slot in slots]


def compact_belief_target(target_belief: dict[str, Any]) -> dict[str, Any]:
    compact_items: list[dict[str, Any]] = []
    for item in target_belief.get("belief_items", []):
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "relation": item.get("relation"),
                "value": item.get("value"),
                "support_slot_ids": item.get("support_slot_ids", []),
            }
        )
    return {"belief_items": compact_items}


def serialize_task_example(task_name: str, row: dict[str, Any]) -> TrainingExample:
    if task_name == "slot_autoencoding":
        return TrainingExample(
            task_name=task_name,
            input_text=_join_sections(
                task_name,
                {"input_observation": compact_observation_payload(row["input_observation"])},
            ),
            target_text=json.dumps(row["target_record"], ensure_ascii=False, sort_keys=True),
        )
    if task_name == "retrieval_alignment":
        return TrainingExample(
            task_name=task_name,
            input_text=_join_sections(
                task_name,
                {
                    "query": row["query"],
                    "positive_slot": compact_slot_payload(row["positive_slot"]),
                    "negative_slots": compact_slot_list(row["negative_slots"]),
                },
            ),
            target_text=json.dumps({"gold_support_slot_ids": row["gold_support_slot_ids"]}, ensure_ascii=False, sort_keys=True),
        )
    if task_name == SLOT_ASSIGNMENT_TASK_NAME:
        return TrainingExample(
            task_name=task_name,
            input_text=_join_sections(
                task_name,
                {
                    "memory_context": compact_slot_list(row["memory_context"]),
                    "new_observation": compact_observation_payload(row["new_observation"]),
                },
            ),
            target_text=json.dumps(
                {
                    "target_action": row["target_action"],
                    "target_flags": row["target_flags"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    if task_name == "composition_to_belief":
        return TrainingExample(
            task_name=task_name,
            input_text=_join_sections(
                task_name,
                {
                    "query": row["query"],
                    "memory_slots": compact_slot_list(row["memory_slots"]),
                },
            ),
            target_text=json.dumps(compact_belief_target(row["target_belief_json"]), ensure_ascii=False, sort_keys=True),
        )
    raise ValueError(f"Unsupported task: {task_name}")


def _join_sections(task_name: str, payload: dict[str, Any]) -> str:
    instruction = "Recover the semantic fields and emit a compact structured object. Semantic correctness matters more than raw JSON surface matching."
    if task_name == SLOT_ASSIGNMENT_TASK_NAME:
        instruction = (
            "Predict slot_assignment action classification and hard-constraint flags. "
            "Choose exactly one target_action from [merge, overwrite, new, ignore]. "
            "Return JSON only with the schema "
            '{"target_action":"new","target_flags":{"promote":false,"stale_old":false}}. '
            "Recover the semantic fields and do not copy input fields into the output."
        )
    sections = [
        f"task: {task_name}",
        f"instruction: {instruction}",
    ]
    for key, value in payload.items():
        sections.append(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(sections)


def _context_section(input_text: str, key: str) -> Any | None:
    prefix = f"{key}: "
    for line in input_text.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
    return None


def _belief_relation_hints(prediction: str) -> list[str]:
    seen: list[str] = []
    for match in _RELATION_HINT_RE.finditer(prediction):
        relation = match.group("relation").strip()
        if "_" not in relation and relation not in {"constraint", "goal", "temporal_fact", "other_fact"}:
            continue
        if relation not in seen:
            seen.append(relation)
    return seen


def _repair_belief_payload_from_input_context(
    input_text: str,
    prediction: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    memory_slots = _context_section(input_text, "memory_slots")
    query = _context_section(input_text, "query")
    if not isinstance(memory_slots, list) or not memory_slots:
        return payload

    items: list[dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("belief_items"), list):
        items = [dict(item) for item in payload.get("belief_items", []) if isinstance(item, dict)]
    else:
        items = []

    if not items:
        items = [{"relation": relation, "value": "", "support_slot_ids": []} for relation in _belief_relation_hints(prediction)]
    if not items:
        return payload

    current_query = str(query or "").lower()

    def _slot_rank(slot: dict[str, Any], relation: str) -> tuple[float, float, float, float]:
        active_flag = 1.0 if bool(slot.get("active_flag", False)) else 0.0
        bank_score = 1.0 if str(slot.get("bank", "")) == "core" else 0.0
        confidence = float(slot.get("confidence", 0.0) or 0.0)
        relation_score = 1.0 if str(slot.get("relation", "")) == relation else 0.0
        if "current" in current_query or "currently" in current_query:
            return (relation_score, active_flag, bank_score, confidence)
        return (relation_score, bank_score, active_flag, confidence)

    repaired_items: list[dict[str, Any]] = []
    for item in items:
        relation = str(item.get("relation", "")).strip()
        value = str(item.get("value", "")).strip()
        support_slot_ids = [str(slot_id) for slot_id in item.get("support_slot_ids", []) if str(slot_id)]
        candidates = [slot for slot in memory_slots if isinstance(slot, dict)]
        if relation:
            relation_candidates = [slot for slot in candidates if str(slot.get("relation", "")) == relation]
            if relation_candidates:
                candidates = relation_candidates
        if not candidates:
            repaired_items.append(
                {
                    "relation": relation,
                    "value": value,
                    "support_slot_ids": support_slot_ids,
                }
            )
            continue
        best_slot = sorted(candidates, key=lambda slot: _slot_rank(slot, relation), reverse=True)[0]
        if not relation:
            relation = str(best_slot.get("relation", "")).strip()
        if not support_slot_ids and str(best_slot.get("slot_id", "")).strip():
            support_slot_ids = [str(best_slot.get("slot_id", "")).strip()]
        if not value:
            canonical_gloss = str(best_slot.get("canonical_gloss", "")).strip()
            if "=" in canonical_gloss:
                value = canonical_gloss.split("=", 1)[1].strip()
        repaired_items.append(
            {
                "relation": relation,
                "value": value,
                "support_slot_ids": support_slot_ids,
            }
        )

    if not repaired_items:
        return payload
    repaired_payload: dict[str, Any] = {"belief_items": repaired_items}
    if isinstance(payload, dict):
        for key in ("query_id", "entity", "query_type", "global_consistency"):
            if key in payload:
                repaired_payload[key] = payload[key]
    return repaired_payload


def _repair_lifecycle_payload_from_input_context(
    input_text: str,
    prediction: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    memory_context = _context_section(input_text, "memory_context")
    new_observation = _context_section(input_text, "new_observation")
    if not isinstance(new_observation, dict):
        return payload

    if isinstance(payload, dict):
        action = str(payload.get("target_action", "")).strip()
        flags = payload.get("target_flags")
        if action and action != "unknown" and isinstance(flags, dict):
            return payload

    relation = str(new_observation.get("relation", "")).strip()
    new_gloss = str(new_observation.get("canonical_gloss", "")).strip()
    status_hint = str(new_observation.get("status_hint", "")).strip().lower()
    candidates = [slot for slot in memory_context if isinstance(slot, dict)] if isinstance(memory_context, list) else []
    relation_candidates = [slot for slot in candidates if str(slot.get("relation", "")).strip() == relation]
    text = prediction.strip()

    if relation_candidates:
        if text:
            mentions_candidate = any(
                token and token in text
                for slot in relation_candidates
                for token in (
                    str(slot.get("canonical_gloss", "")).strip(),
                    str(slot.get("slot_id", "")).strip(),
                    relation,
                )
            )
            if not mentions_candidate:
                return payload
        has_changed_existing = any(
            str(slot.get("canonical_gloss", "")).strip()
            and str(slot.get("canonical_gloss", "")).strip() != new_gloss
            for slot in relation_candidates
        )
        if status_hint == "active" and has_changed_existing:
            return {
                "target_action": "overwrite",
                "target_flags": {
                    "promote": True,
                    "stale_old": True,
                },
            }
        if any(str(slot.get("canonical_gloss", "")).strip() == new_gloss for slot in relation_candidates):
            return {
                "target_action": "merge",
                "target_flags": {
                    "promote": False,
                    "stale_old": False,
                },
            }

    if relation and new_gloss:
        return {
            "target_action": "new",
            "target_flags": {
                "promote": status_hint == "active",
                "stale_old": False,
            },
        }
    return payload


def slot_assignment_metrics_from_eval_payload(trained_eval: dict[str, Any] | None) -> dict[str, Any]:
    lifecycle_metrics = ((trained_eval or {}).get("per_task") or {}).get(SLOT_ASSIGNMENT_TASK_NAME) or {}
    return {
        "slot_assignment_task": SLOT_ASSIGNMENT_TASK_NAME,
        "slot_assignment_accuracy": float(lifecycle_metrics.get("field_accuracy", 0.0)),
        "slot_assignment_field_f1": float(lifecycle_metrics.get("field_f1", 0.0)),
        "slot_assignment_exact_match": float(lifecycle_metrics.get("exact_match", 0.0)),
        "slot_assignment_token_f1": float(lifecycle_metrics.get("token_f1", 0.0)),
        "slot_assignment_count": int(lifecycle_metrics.get("count", 0)),
    }


class CharTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def encode(self, text: str, max_length: int) -> list[int]:
        payload = [self.bos_token_id]
        for byte in text.encode("utf-8")[: max(max_length - 2, 0)]:
            payload.append(byte + 4)
        payload.append(self.eos_token_id)
        payload = payload[:max_length]
        if len(payload) < max_length:
            payload.extend([self.pad_token_id] * (max_length - len(payload)))
        return payload

    def encode_target(self, text: str, max_length: int) -> list[int]:
        encoded = self.encode(text, max_length)
        labels = [token if token != self.pad_token_id else -100 for token in encoded]
        return labels

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"type": "char_tokenizer", "vocab_size": self.vocab_size}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _encode_source_ids(tokenizer: Any, text: str, max_length: int) -> tuple[list[int], list[int]]:
    if isinstance(tokenizer, CharTokenizer):
        input_ids = tokenizer.encode(text, max_length)
        attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in input_ids]
        return input_ids, attention_mask

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()
    return input_ids, attention_mask


def _encode_target_ids(tokenizer: Any, text: str, max_length: int) -> list[int]:
    if isinstance(tokenizer, CharTokenizer):
        return tokenizer.encode_target(text, max_length)

    encoded = tokenizer(
        text_target=text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = encoded["input_ids"][0].tolist()
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    return [token if token != pad_token_id else -100 for token in labels]


class PreparedSeq2SeqDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: CharTokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        input_ids, attention_mask = _encode_source_ids(self.tokenizer, example.input_text, self.max_source_length)
        labels = _encode_target_ids(self.tokenizer, example.target_text, self.max_target_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TinySeq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        embedded = self.embed(input_ids)
        _, hidden = self.encoder(embedded)
        if labels is None:
            steps = input_ids.size(1)
        else:
            steps = labels.size(1)
        repeated = hidden.transpose(0, 1).expand(-1, steps, -1)
        logits = self.output(repeated)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("TinySeq2SeqOutput", (), {"loss": loss, "logits": logits})


class TinyTaskAdaptiveSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, adapter_names: dict[str, str], hidden_size: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        unique_adapters = sorted(set(adapter_names.values()))
        self.adapter_bridges = nn.ModuleDict({name: nn.Linear(hidden_size, hidden_size) for name in unique_adapters})
        self.output_heads = nn.ModuleDict({name: nn.Linear(hidden_size, vocab_size) for name in unique_adapters})
        self.active_adapter = unique_adapters[0] if unique_adapters else "default"

    def set_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.adapter_bridges:
            raise KeyError(f"Unknown tiny task adapter: {adapter_name}")
        self.active_adapter = adapter_name

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        adapter_name: str | None = None,
    ) -> Any:
        adapter_key = adapter_name or self.active_adapter
        if adapter_key not in self.adapter_bridges:
            raise KeyError(f"Unknown tiny task adapter: {adapter_key}")
        embedded = self.embed(input_ids)
        _, hidden = self.encoder(embedded)
        if labels is None:
            steps = input_ids.size(1)
        else:
            steps = labels.size(1)
        shared = hidden.transpose(0, 1).expand(-1, steps, -1)
        adapted = torch.tanh(self.adapter_bridges[adapter_key](shared))
        logits = self.output_heads[adapter_key](adapted)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("TinyTaskAdaptiveSeq2SeqOutput", (), {"loss": loss, "logits": logits})


def _decode_char_tokens(tokens: list[int]) -> str:
    payload = bytearray()
    for token in tokens:
        if token in {CharTokenizer.pad_token_id, CharTokenizer.bos_token_id}:
            continue
        if token == CharTokenizer.eos_token_id:
            break
        if token >= 4:
            payload.append(token - 4)
    return payload.decode("utf-8", errors="ignore").strip()


def _normalize_text(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    try:
        return json.dumps(json.loads(stripped), ensure_ascii=False, sort_keys=True)
    except json.JSONDecodeError:
        return stripped


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    target_tokens = _normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    pred_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), target_counts.get(token, 0)) for token in pred_counts)
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(target_tokens) if target_tokens else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def configure_hf_cache(config: dict[str, Any]) -> dict[str, str]:
    cache_root = config.get("runtime", {}).get("hf_cache", {}).get("root")
    if not cache_root:
        return {}

    root = Path(cache_root).resolve()
    hub_cache = root / "hub"
    datasets_cache = root / "datasets"
    assets_cache = root / "assets"
    for path in (root, hub_cache, datasets_cache, assets_cache):
        path.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(hub_cache),
        "TRANSFORMERS_CACHE": str(hub_cache),
        "HF_DATASETS_CACHE": str(datasets_cache),
        "HUGGINGFACE_ASSETS_CACHE": str(assets_cache),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def build_runtime_components(
    config: dict[str, Any],
    *,
    tasks: list[str] | None = None,
) -> tuple[Any, Any]:
    backbone = config.get("model", {}).get("backbone", "google/flan-t5-base")
    adapter_settings = task_adapter_settings(config, tasks=tasks)
    if backbone == "__tiny_debug_seq2seq__":
        tokenizer = CharTokenizer()
        if adapter_settings["enabled"]:
            model = TinyTaskAdaptiveSeq2SeqModel(tokenizer.vocab_size, adapter_settings["names"])
        else:
            model = TinySeq2SeqModel(tokenizer.vocab_size)
        return model, tokenizer

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    env_updates = configure_hf_cache(config)
    cache_dir = env_updates.get("HF_HUB_CACHE")
    tokenizer = AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(backbone, cache_dir=cache_dir)
    lora_cfg = config.get("model", {}).get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q", "v"]),
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    if adapter_settings["enabled"]:
        adapter_names = list(dict.fromkeys(adapter_settings["names"].values()))
        primary_adapter = adapter_names[0]
        model = get_peft_model(model, peft_config, adapter_name=primary_adapter)
        for adapter_name in adapter_names[1:]:
            model.add_adapter(adapter_name, peft_config)
        model.set_adapter(primary_adapter)
    else:
        model = get_peft_model(model, peft_config)
    return model, tokenizer


def load_runtime_components(
    config: dict[str, Any],
    checkpoint_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[Any, Any]:
    backbone = config.get("model", {}).get("backbone", "google/flan-t5-base")
    adapter_settings = task_adapter_settings(config)
    if backbone == "__tiny_debug_seq2seq__":
        tokenizer = CharTokenizer()
        if adapter_settings["enabled"]:
            model = TinyTaskAdaptiveSeq2SeqModel(tokenizer.vocab_size, adapter_settings["names"])
        else:
            model = TinySeq2SeqModel(tokenizer.vocab_size)
        state_dict = torch.load(checkpoint_dir / "tiny_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, tokenizer

    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    env_updates = configure_hf_cache(config)
    cache_dir = env_updates.get("HF_HUB_CACHE")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", cache_dir=cache_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(backbone, cache_dir=cache_dir)
    if adapter_settings["enabled"]:
        adapter_names = list(dict.fromkeys(adapter_settings["names"].values()))
        primary_adapter = adapter_names[0]
        model = PeftModel.from_pretrained(base_model, checkpoint_dir / primary_adapter, adapter_name=primary_adapter)
        for adapter_name in adapter_names[1:]:
            model.load_adapter(str(checkpoint_dir / adapter_name), adapter_name=adapter_name)
        model.set_adapter(primary_adapter)
    else:
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_prediction_text(
    model: Any,
    tokenizer: Any,
    example: TrainingExample,
    *,
    max_source_length: int,
    max_target_length: int,
    device: str,
) -> str:
    if isinstance(tokenizer, CharTokenizer):
        input_ids = torch.tensor([tokenizer.encode(example.input_text, max_source_length)], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [[1 if token != tokenizer.pad_token_id else 0 for token in input_ids[0].tolist()]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            if hasattr(model, "active_adapter"):
                output = model(input_ids=input_ids, attention_mask=attention_mask, adapter_name=model.active_adapter)
            else:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_ids = output.logits.argmax(dim=-1)[0].detach().cpu().tolist()
        return _decode_char_tokens(predicted_ids)

    encoded = tokenizer(
        example.input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_target_length,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def evaluate_stage2_checkpoint(
    config: dict[str, Any],
    prepared_manifest_path: Path,
    checkpoint_dir: Path,
    *,
    tasks: list[str] | None = None,
    max_eval_examples: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    configured_tasks = tasks or list(config.get("training", {}).get("tasks", [])) or None
    adapter_settings = task_adapter_settings(config, tasks=configured_tasks)
    examples = build_training_examples(prepared_manifest_path, tasks=configured_tasks)
    examples = _balanced_cap_examples(examples, max_examples=max_eval_examples)
    if not examples:
        raise ValueError("No evaluation examples were prepared for stage-2 checkpoint eval.")

    model, tokenizer = load_runtime_components(config, checkpoint_dir, device=device)
    batching = config.get("training", {}).get("batching", {})
    max_source_length = int(batching.get("max_source_length", 256))
    max_target_length = int(batching.get("max_target_length", 192))

    overall_exact: list[float] = []
    overall_token_f1: list[float] = []
    overall_json_validity: list[float] = []
    overall_semantic_validity: list[float] = []
    overall_field_accuracy: list[float] = []
    overall_field_f1: list[float] = []
    per_task: dict[str, dict[str, list[float] | int]] = {}
    samples: list[dict[str, Any]] = []
    for example in examples:
        if adapter_settings["enabled"] and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_settings["names"][example.task_name])
        prediction = generate_prediction_text(
            model,
            tokenizer,
            example,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            device=device,
        )
        target = example.target_text
        semantic_task_name = "slot_assignment" if example.task_name == SLOT_ASSIGNMENT_TASK_NAME else example.task_name
        prediction_payload = coerce_task_payload(semantic_task_name, prediction)
        target_payload = coerce_task_payload(semantic_task_name, target)
        if semantic_task_name == "slot_assignment":
            prediction_payload = _repair_lifecycle_payload_from_input_context(
                example.input_text,
                prediction,
                prediction_payload if isinstance(prediction_payload, dict) else None,
            )
        if semantic_task_name == "composition_to_belief":
            prediction_payload = _repair_belief_payload_from_input_context(
                example.input_text,
                prediction,
                prediction_payload if isinstance(prediction_payload, dict) else None,
            )
        prediction_text = render_task_payload(semantic_task_name, prediction_payload) if prediction_payload is not None else prediction
        target_text = render_task_payload(semantic_task_name, target_payload) if target_payload is not None else target
        exact = float(_normalize_text(prediction_text) == _normalize_text(target_text))
        token_f1 = _token_f1(prediction_text, target_text)
        field_scores = semantic_task_scores(semantic_task_name, prediction_payload, target_payload)
        json_validity = float(json_like_payload(prediction) is not None)
        semantic_validity = float(prediction_payload is not None)
        overall_exact.append(exact)
        overall_token_f1.append(token_f1)
        overall_json_validity.append(json_validity)
        overall_semantic_validity.append(semantic_validity)
        overall_field_accuracy.append(field_scores["field_accuracy"])
        overall_field_f1.append(field_scores["field_f1"])
        stats = per_task.setdefault(
            example.task_name,
            {
                "exact_match": [],
                "token_f1": [],
                "json_validity_rate": [],
                "semantic_validity_rate": [],
                "field_accuracy": [],
                "field_f1": [],
                "count": 0,
            },
        )
        stats["exact_match"].append(exact)
        stats["token_f1"].append(token_f1)
        stats["json_validity_rate"].append(json_validity)
        stats["semantic_validity_rate"].append(semantic_validity)
        stats["field_accuracy"].append(field_scores["field_accuracy"])
        stats["field_f1"].append(field_scores["field_f1"])
        stats["count"] = int(stats["count"]) + 1
        samples.append(
            {
                "task_name": example.task_name,
                "input_preview": example.input_text[:240],
                "prediction_preview": prediction[:240],
                "target_preview": target[:240],
                "normalized_prediction_preview": prediction_text[:240],
                "exact_match": exact,
                "token_f1": token_f1,
                "json_validity_rate": json_validity,
                "semantic_validity_rate": semantic_validity,
                "field_accuracy": field_scores["field_accuracy"],
                "field_f1": field_scores["field_f1"],
            }
        )

    summarized_tasks = {
        task_name: {
            "count": int(stats["count"]),
            "exact_match": sum(stats["exact_match"]) / len(stats["exact_match"]) if stats["exact_match"] else 0.0,
            "token_f1": sum(stats["token_f1"]) / len(stats["token_f1"]) if stats["token_f1"] else 0.0,
            "json_validity_rate": sum(stats["json_validity_rate"]) / len(stats["json_validity_rate"]) if stats["json_validity_rate"] else 0.0,
            "semantic_validity_rate": sum(stats["semantic_validity_rate"]) / len(stats["semantic_validity_rate"]) if stats["semantic_validity_rate"] else 0.0,
            "field_accuracy": sum(stats["field_accuracy"]) / len(stats["field_accuracy"]) if stats["field_accuracy"] else 0.0,
            "field_f1": sum(stats["field_f1"]) / len(stats["field_f1"]) if stats["field_f1"] else 0.0,
        }
        for task_name, stats in per_task.items()
    }
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "num_examples": len(examples),
        "metrics": {
            "exact_match": sum(overall_exact) / len(overall_exact) if overall_exact else 0.0,
            "token_f1": sum(overall_token_f1) / len(overall_token_f1) if overall_token_f1 else 0.0,
            "json_validity_rate": sum(overall_json_validity) / len(overall_json_validity) if overall_json_validity else 0.0,
            "semantic_validity_rate": sum(overall_semantic_validity) / len(overall_semantic_validity) if overall_semantic_validity else 0.0,
            "field_accuracy": sum(overall_field_accuracy) / len(overall_field_accuracy) if overall_field_accuracy else 0.0,
            "field_f1": sum(overall_field_f1) / len(overall_field_f1) if overall_field_f1 else 0.0,
        },
        "per_task": summarized_tasks,
        "sample_previews": samples[:16],
    }


def train_stage2_model(
    config: dict[str, Any],
    prepared_manifest_path: Path,
    *,
    max_steps: int | None = None,
    max_train_examples: int | None = None,
    device: str = "cpu",
    disabled_pools: list[str] | None = None,
) -> dict[str, Any]:
    online_aligned = bool(config.get("training", {}).get("online_aligned", False))
    configured_tasks = list(config.get("training", {}).get("tasks", [])) or None
    adapter_settings = task_adapter_settings(config, tasks=configured_tasks)
    examples = build_training_examples_with_variant(
        prepared_manifest_path,
        tasks=configured_tasks,
        disabled_pools=disabled_pools,
        online_aligned=online_aligned,
    )
    examples = _balanced_cap_examples(
        examples,
        max_examples=max_train_examples,
        task_weights={
            "slot_autoencoding": 1,
            "retrieval_alignment": 2 if online_aligned else 1,
            SLOT_ASSIGNMENT_TASK_NAME: 2 if online_aligned else 1,
            "composition_to_belief": 3 if online_aligned else 1,
        },
    )
    if not examples:
        raise ValueError("No training examples were prepared for stage-2.")

    model, tokenizer = build_runtime_components(config, tasks=configured_tasks)
    model.to(device)
    batching = config.get("training", {}).get("batching", {})
    max_source_length = int(batching.get("max_source_length", 256))
    max_target_length = int(batching.get("max_target_length", 192))
    batch_size = int(batching.get("per_device_batch_size", 4))
    grad_accumulation = max(int(batching.get("gradient_accumulation_steps", 1)), 1)
    epochs = int(batching.get("num_train_epochs", 1))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("optimizer", {}).get("lr", 1e-4)),
        weight_decay=float(config.get("training", {}).get("optimizer", {}).get("weight_decay", 0.0)),
    )

    step = 0
    optimizer_steps = 0
    loss_history: list[float] = []
    if adapter_settings["enabled"]:
        grouped_examples: dict[str, list[TrainingExample]] = {
            task_name: [example for example in examples if example.task_name == task_name]
            for task_name in configured_tasks or sorted({example.task_name for example in examples})
        }
        grouped_loaders = {
            task_name: DataLoader(
                PreparedSeq2SeqDataset(
                    task_examples,
                    tokenizer,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            for task_name, task_examples in grouped_examples.items()
            if task_examples
        }
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            iterators = {task_name: iter(loader) for task_name, loader in grouped_loaders.items()}
            active_tasks = list(grouped_loaders)
            while iterators:
                progressed = False
                for task_name in active_tasks:
                    iterator = iterators.get(task_name)
                    if iterator is None:
                        continue
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterators.pop(task_name, None)
                        continue
                    progressed = True
                    step += 1
                    batch = {key: value.to(device) for key, value in batch.items()}
                    if hasattr(model, "set_adapter"):
                        model.set_adapter(adapter_settings["names"][task_name])
                    if hasattr(model, "active_adapter"):
                        output = model(**batch, adapter_name=adapter_settings["names"][task_name])
                    else:
                        output = model(**batch)
                    loss = output.loss
                    if loss is None:
                        raise RuntimeError("Training model did not return a loss.")
                    scaled_loss = loss / grad_accumulation
                    scaled_loss.backward()
                    if step % grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                    loss_history.append(float(loss.detach().cpu().item()))
                    if max_steps is not None and step >= max_steps:
                        break
                if not progressed or (max_steps is not None and step >= max_steps):
                    break
            if max_steps is not None and step >= max_steps:
                break
    else:
        dataset = PreparedSeq2SeqDataset(
            examples,
            tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            for batch in loader:
                step += 1
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = output.loss
                if loss is None:
                    raise RuntimeError("Training model did not return a loss.")
                scaled_loss = loss / grad_accumulation
                scaled_loss.backward()
                if step % grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                loss_history.append(float(loss.detach().cpu().item()))
                if max_steps is not None and step >= max_steps:
                    break
            if max_steps is not None and step >= max_steps:
                break
    if step % grad_accumulation != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    return {
        "num_examples": len(examples),
        "num_steps": step,
        "optimizer_steps": optimizer_steps,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "online_aligned": online_aligned,
        "task_adapters_enabled": adapter_settings["enabled"],
        "task_adapter_names": adapter_settings["names"],
        "model": model,
        "tokenizer": tokenizer,
    }


def save_training_artifacts(
    *,
    run_dir: Path,
    output_root: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, str]:
    backbone = config.get("model", {}).get("backbone", "google/flan-t5-base")
    adapter_settings = task_adapter_settings(config)
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = metrics.pop("model")
    tokenizer = metrics.pop("tokenizer")
    metrics_path = run_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if backbone == "__tiny_debug_seq2seq__":
        torch.save(model.state_dict(), checkpoint_dir / "tiny_model.pt")
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    else:
        if adapter_settings["enabled"]:
            model.save_pretrained(checkpoint_dir, selected_adapters=list(dict.fromkeys(adapter_settings["names"].values())))
        else:
            model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    if adapter_settings["enabled"]:
        (checkpoint_dir / "task_adapters.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "task_adapter_names": adapter_settings["names"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "metrics_path": str(metrics_path),
        "checkpoint_dir": str(checkpoint_dir),
    }
