"""Prepare stage-2 task manifests and tiny demo samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2 import STAGE2_TASK_DATASETS, STAGE2_TASK_MODULES, build_demo_prepared_payload, build_task_registry
from core_mem.config import load_project_config
from core_mem.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider
from core_mem.v2.datasets import (
    STAGE2_DATA_POOLS,
    build_prepared_payload_from_sources,
    build_source_record_split_from_sources,
    summarize_stage2_sources,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_by_sample_id(path: Path, *, allowed_sample_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows_by_sample: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        rows_by_sample[sample_id] = row
    return rows_by_sample


def _materialize_prepared_payload(
    output_root: Path,
    *,
    prepared_payload: dict[str, list[dict]],
    source_config: Path | None,
    source_summary: dict[str, object] | None,
    max_rows_per_dataset: int | None,
    mode: str,
    namespace: str | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    registry_entries = [entry.to_dict() for entry in build_task_registry()]
    artifact_root = output_root / "artifacts"
    cache_root = output_root / "cache_manifests"
    if namespace:
        artifact_root = artifact_root / namespace
        cache_root = cache_root / namespace

    cache_manifest_path = cache_root / "stage2_dataset_registry.json"
    task_registry_path = artifact_root / "stage2_task_registry.json"
    prepared_manifest_path = artifact_root / "stage2_prepared_samples_manifest.json"

    _write_json(
        cache_manifest_path,
        {
            "data_pools": STAGE2_DATA_POOLS,
            "task_datasets": STAGE2_TASK_DATASETS,
            "task_modules": STAGE2_TASK_MODULES,
            "source_config": str(source_config) if source_config is not None else None,
            "source_summary": source_summary,
            "max_rows_per_dataset": max_rows_per_dataset,
        },
    )
    _write_json(task_registry_path, {"tasks": registry_entries})

    task_files: dict[str, str] = {}
    task_counts: dict[str, int] = {}
    for task_name, rows in prepared_payload.items():
        task_path = artifact_root / f"{task_name}.jsonl"
        _write_jsonl(task_path, rows)
        task_files[task_name] = str(task_path)
        task_counts[task_name] = len(rows)

    manifest = {
        "prepared_at": "2026-04-07T00:00:00Z",
        "mode": mode,
        "task_files": task_files,
        "task_counts": task_counts,
        "task_registry": str(task_registry_path),
        "dataset_registry": str(cache_manifest_path),
        "source_config": str(source_config) if source_config is not None else None,
        "source_summary": source_summary,
        "max_rows_per_dataset": max_rows_per_dataset,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(prepared_manifest_path, manifest)
    return {
        "prepared_manifest": str(prepared_manifest_path),
        "task_registry": str(task_registry_path),
        "dataset_registry": str(cache_manifest_path),
        "task_counts": task_counts,
    }


def prepare_stage2_data(
    output_root: Path,
    *,
    source_config: Path | None = None,
    strict_sources: bool = False,
    max_rows_per_dataset: int | None = None,
) -> dict[str, object]:
    source_summary = None
    if source_config is not None:
        source_summary = summarize_stage2_sources(source_config)
        prepared_payload = build_prepared_payload_from_sources(
            source_config,
            max_rows_per_dataset=max_rows_per_dataset,
        )
    else:
        prepared_payload = {}
    if not any(prepared_payload.values()):
        if strict_sources:
            raise FileNotFoundError("No stage-2 source datasets were available under the configured paths.")
        prepared_payload = build_demo_prepared_payload()
        mode = "demo_fallback"
    else:
        mode = "configured_sources"
    return _materialize_prepared_payload(
        output_root,
        prepared_payload=prepared_payload,
        source_config=source_config,
        source_summary=source_summary,
        max_rows_per_dataset=max_rows_per_dataset,
        mode=mode,
    )


def _selection_map(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    selection: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        selection[str(row["dataset_key"])].add(str(row["sample_id"]))
    return dict(selection)


def _top_counter(rows: list[dict[str, object]], field: str, *, limit: int = 10) -> list[dict[str, object]]:
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    return [{"value": value, "count": count} for value, count in counts.most_common(limit)]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else "unknown"


def _provider_from_config(config_path: Path) -> tuple[dict[str, Any], OpenAICompatibleProvider]:
    config = load_project_config(config_path)
    return (
        {
            "config_path": str(config_path),
            "provider": config.llm.provider,
            "api_key_env": config.llm.api_key_env,
            "base_url": config.llm.base_url,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key_env=config.llm.api_key_env,
                base_url=config.llm.base_url,
                model=config.llm.model,
                timeout_seconds=config.llm.timeout_seconds,
                max_retries=config.llm.max_retries,
                retry_backoff_seconds=config.llm.retry_backoff_seconds,
                min_request_interval_seconds=config.llm.min_request_interval_seconds,
                max_retry_delay_seconds=config.llm.max_retry_delay_seconds,
            )
        ),
    )


def _extract_json_payload(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.startswith("```")]
        candidate = "\n".join(lines).strip()
    for payload in (candidate,):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = candidate.find(start_char)
        end = candidate.rfind(end_char)
        if start == -1 or end <= start:
            continue
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("Teacher response did not contain valid JSON.")


def _compact_observation_payload(observation: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "obs_id",
        "entity",
        "relation",
        "value",
        "value_type",
        "time_scope",
        "status_hint",
        "polarity",
        "canonical_gloss",
        "evidence_text",
        "metadata",
    ]
    return {key: observation.get(key) for key in keys if key in observation}


def _observation_teacher_payload(observation: dict[str, Any]) -> dict[str, Any]:
    metadata = observation.get("metadata") or {}
    allowed_metadata = {
        key: metadata.get(key)
        for key in ("service", "slot_name", "source_type", "domain", "speaker_role")
        if key in metadata
    }
    keys = [
        "obs_id",
        "source_dataset",
        "source_dialogue_id",
        "source_turn_id",
        "session_id",
        "speaker",
        "entity",
        "evidence_text",
    ]
    payload = {key: observation.get(key) for key in keys if key in observation}
    if allowed_metadata:
        payload["metadata"] = allowed_metadata
    return payload


def _compact_slot_payload(slot: dict[str, Any]) -> dict[str, Any]:
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


def _compact_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_slot_payload(slot) for slot in slots]


def _task_row_dataset(row: dict[str, Any]) -> str:
    return str((row.get("_meta", {}) or {}).get("dataset", "unknown"))


def _task_row_sample_id(row: dict[str, Any]) -> str:
    return str((row.get("_meta", {}) or {}).get("sample_id", "unknown"))


def _task_row_relation(kind: str, row: dict[str, Any]) -> str:
    if kind == "observation":
        return str((row.get("input_observation") or {}).get("relation", "unknown"))
    if kind == "slot_assignment":
        return str((row.get("new_observation") or {}).get("relation", "unknown"))
    belief_items = ((row.get("target_belief_json") or {}).get("belief_items") or [])
    if belief_items:
        return str((belief_items[0] or {}).get("relation", "unknown"))
    memory_slots = row.get("memory_slots") or []
    return str((memory_slots[0] or {}).get("relation", "unknown")) if memory_slots else "unknown"


def _round_robin_sample(rows: list[dict[str, Any]], *, kind: str, max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_task_row_dataset(row), _task_row_relation(kind, row))].append(row)
    for key in grouped:
        grouped[key].sort(key=_task_row_sample_id)
    selected: list[dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < max_rows:
        progressed = False
        for key in keys:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= max_rows:
                break
        if not progressed:
            break
    return selected


def _batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _teacher_payload(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sample_id": _task_row_sample_id(row),
        "dataset": _task_row_dataset(row),
    }
    if kind == "observation":
        payload["raw_observation"] = _observation_teacher_payload(row["input_observation"])
        return payload
    if kind == "slot_assignment":
        payload["memory_context"] = _compact_slots(row["memory_context"])
        payload["new_observation"] = _compact_observation_payload(row["new_observation"])
        return payload
    payload["query"] = row["query"]
    payload["memory_slots"] = _compact_slots(row["memory_slots"])
    return payload


def _teacher_prompt(kind: str, rows: list[dict[str, Any]]) -> tuple[str, str]:
    examples = [_teacher_payload(kind, row) for row in rows]
    if kind == "observation":
        system_prompt = "You are a strict data-labeling teacher. Return JSON only."
        user_prompt = (
            "teacher_task: observation\n"
            "Infer semantic observation fields directly from each raw observation.\n"
            "Return a JSON object with a top-level `labels` array.\n"
            "Each label must contain: sample_id, relation, value, time_scope, status_hint, polarity.\n"
            "Use evidence_text and lightweight metadata as grounding.\n"
            "Do not rely on pre-filled relation/value/time labels; infer them from the raw observation content.\n"
            "Keep values concise and do not invent unsupported facts.\n"
            f"examples={json.dumps(examples, ensure_ascii=False, sort_keys=True)}"
        )
        return system_prompt, user_prompt
    if kind == "slot_assignment":
        system_prompt = "You are a strict data-labeling teacher. Return JSON only."
        user_prompt = (
            "teacher_task: slot_assignment\n"
            "Infer slot-assignment supervision from memory_context and new_observation.\n"
            "Return a JSON object with a top-level `labels` array.\n"
            "Each label must contain: sample_id, target_action, matched_slot_id, promote, stale_old.\n"
            "target_action must be one of merge, overwrite, new, ignore.\n"
            "matched_slot_id must be null or one slot_id from memory_context.\n"
            f"examples={json.dumps(examples, ensure_ascii=False, sort_keys=True)}"
        )
        return system_prompt, user_prompt
    system_prompt = "You are a strict data-labeling teacher. Return JSON only."
    user_prompt = (
        "teacher_task: belief_fields\n"
        "Recover belief-field supervision from query and memory_slots.\n"
        "Return a JSON object with a top-level `labels` array.\n"
        "Each label must contain sample_id and belief_items.\n"
        "belief_items must be a list of objects with relation, value, status, time_scope, support_slot_ids.\n"
        "support_slot_ids must only use slot_id values from the provided memory_slots.\n"
        f"examples={json.dumps(examples, ensure_ascii=False, sort_keys=True)}"
    )
    return system_prompt, user_prompt


def _coerce_label(kind: str, row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id", "")).strip()
    if sample_id != _task_row_sample_id(row):
        raise ValueError(f"Teacher label sample mismatch: {sample_id} != {_task_row_sample_id(row)}")
    if kind == "observation":
        label = {
            "sample_id": sample_id,
            "relation": str(payload.get("relation", "")).strip(),
            "value": str(payload.get("value", "")).strip(),
            "time_scope": str(payload.get("time_scope", "")).strip(),
            "status_hint": str(payload.get("status_hint", "")).strip(),
            "polarity": str(payload.get("polarity", "")).strip(),
        }
        if not all(label[field] for field in ("relation", "value", "time_scope", "status_hint", "polarity")):
            raise ValueError(f"Incomplete observation teacher label for {sample_id}")
        return label
    if kind == "slot_assignment":
        action = str(payload.get("target_action", "")).strip()
        if action not in {"merge", "overwrite", "new", "ignore"}:
            raise ValueError(f"Invalid slot-assignment action for {sample_id}: {action}")
        slot_ids = {str(slot.get("slot_id")) for slot in row.get("memory_context", []) if slot.get("slot_id")}
        matched_slot_id = payload.get("matched_slot_id")
        normalized_slot = None if matched_slot_id in {None, "", "null"} else str(matched_slot_id)
        if normalized_slot is not None and normalized_slot not in slot_ids:
            raise ValueError(f"Teacher matched_slot_id {normalized_slot} is outside memory_context for {sample_id}")
        return {
            "sample_id": sample_id,
            "target_action": action,
            "matched_slot_id": normalized_slot,
            "promote": bool(payload.get("promote", False)),
            "stale_old": bool(payload.get("stale_old", False)),
        }
    allowed_slot_ids = {str(slot.get("slot_id")) for slot in row.get("memory_slots", []) if slot.get("slot_id")}
    belief_items = []
    for item in payload.get("belief_items", []) or []:
        if not isinstance(item, dict):
            continue
        support_ids = [
            str(slot_id)
            for slot_id in (item.get("support_slot_ids") or [])
            if str(slot_id) in allowed_slot_ids
        ]
        relation = str(item.get("relation", "")).strip()
        value = str(item.get("value", "")).strip()
        if not relation or not value:
            continue
        belief_items.append(
            {
                "relation": relation,
                "value": value,
                "status": _normalize_teacher_belief_status(item.get("status", "unknown")),
                "time_scope": _normalize_teacher_belief_time_scope(item.get("time_scope", "unknown")),
                "support_slot_ids": support_ids,
            }
        )
    return {
        "sample_id": sample_id,
        "belief_items": belief_items,
    }


def _request_teacher_labels(
    kind: str,
    rows: list[dict[str, Any]],
    *,
    provider: Any,
) -> tuple[dict[str, dict[str, Any]], int, list[dict[str, Any]]]:
    if not rows:
        return {}, 0, []
    system_prompt, user_prompt = _teacher_prompt(kind, rows)
    response = provider.chat(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=2048,
    )
    try:
        payload = _extract_json_payload(response.content)
    except ValueError:
        if len(rows) > 1:
            label_map: dict[str, dict[str, Any]] = {}
            request_count = 1
            failures: list[dict[str, Any]] = []
            for row in rows:
                single_map, single_requests, single_failures = _request_teacher_labels(kind, [row], provider=provider)
                label_map.update(single_map)
                request_count += single_requests
                failures.extend(single_failures)
            return label_map, request_count, failures
        return (
            {},
            1,
            [
                {
                    "sample_id": _task_row_sample_id(rows[0]),
                    "error": "invalid_json",
                    "response_preview": response.content[:800],
                }
            ],
        )
    labels = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(labels, list):
        if len(rows) > 1:
            label_map = {}
            request_count = 1
            failures = []
            for row in rows:
                single_map, single_requests, single_failures = _request_teacher_labels(kind, [row], provider=provider)
                label_map.update(single_map)
                request_count += single_requests
                failures.extend(single_failures)
            return label_map, request_count, failures
        return (
            {},
            1,
            [
                {
                    "sample_id": _task_row_sample_id(rows[0]),
                    "error": "missing_labels_list",
                    "response_preview": response.content[:800],
                }
            ],
        )
    label_map = {
        str(item.get("sample_id")): item
        for item in labels
        if isinstance(item, dict) and item.get("sample_id")
    }
    missing_rows = [row for row in rows if _task_row_sample_id(row) not in label_map]
    request_count = 1
    failures: list[dict[str, Any]] = []
    if missing_rows and len(rows) > 1:
        for missing_row in missing_rows:
            single_map, single_requests, single_failures = _request_teacher_labels(kind, [missing_row], provider=provider)
            label_map.update(single_map)
            request_count += single_requests
            failures.extend(single_failures)
    if len(rows) == 1 and _task_row_sample_id(rows[0]) not in label_map:
        failures.append(
            {
                "sample_id": _task_row_sample_id(rows[0]),
                "error": "missing_sample_id",
                "response_preview": response.content[:800],
            }
        )
    return label_map, request_count, failures


def _source_label(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    sample_id = _task_row_sample_id(row)
    if kind == "observation":
        observation = row["input_observation"]
        return {
            "sample_id": sample_id,
            "relation": observation.get("relation"),
            "value": observation.get("value"),
            "time_scope": observation.get("time_scope"),
            "status_hint": observation.get("status_hint"),
            "polarity": observation.get("polarity"),
        }
    if kind == "slot_assignment":
        affected_ids = row.get("affected_slot_ids") or []
        return {
            "sample_id": sample_id,
            "target_action": row.get("target_action"),
            "matched_slot_id": str(affected_ids[0]) if affected_ids else None,
            "promote": bool((row.get("target_flags") or {}).get("promote", False)),
            "stale_old": bool((row.get("target_flags") or {}).get("stale_old", False)),
        }
    belief_items = ((row.get("target_belief_json") or {}).get("belief_items") or [])
    return {
        "sample_id": sample_id,
        "belief_items": belief_items,
    }


def _agreement_metrics(kind: str, source: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    if kind == "observation":
        return {
            "relation_match": teacher.get("relation") == source.get("relation"),
            "value_match": teacher.get("value") == source.get("value"),
            "time_scope_match": teacher.get("time_scope") == source.get("time_scope"),
            "status_hint_match": teacher.get("status_hint") == source.get("status_hint"),
            "polarity_match": teacher.get("polarity") == source.get("polarity"),
        }
    if kind == "slot_assignment":
        matched_slot_id = teacher.get("matched_slot_id")
        return {
            "target_action_match": teacher.get("target_action") == source.get("target_action"),
            "promote_match": bool(teacher.get("promote")) == bool(source.get("promote")),
            "stale_old_match": bool(teacher.get("stale_old")) == bool(source.get("stale_old")),
            "matched_slot_match": matched_slot_id == source.get("matched_slot_id"),
            "matched_slot_present": matched_slot_id is not None,
        }
    source_items = source.get("belief_items") or []
    teacher_items = teacher.get("belief_items") or []
    source_first = source_items[0] if source_items else {}
    teacher_first = teacher_items[0] if teacher_items else {}
    source_support = {str(slot_id) for slot_id in source_first.get("support_slot_ids", [])}
    teacher_support = {str(slot_id) for slot_id in teacher_first.get("support_slot_ids", [])}
    return {
        "belief_non_empty": bool(teacher_items),
        "first_relation_match": teacher_first.get("relation") == source_first.get("relation"),
        "first_value_match": teacher_first.get("value") == source_first.get("value"),
        "support_overlap": bool(source_support & teacher_support),
    }


def _summarize_teacher_rows(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"labeled_examples": 0}
    dataset_counter = Counter(str(row.get("dataset", "unknown")) for row in rows)
    relation_counter = Counter(str(row.get("relation", "unknown")) for row in rows)
    summary: dict[str, Any] = {
        "labeled_examples": len(rows),
        "dataset_counts": [{"value": value, "count": count} for value, count in dataset_counter.most_common(12)],
        "relation_counts": [{"value": value, "count": count} for value, count in relation_counter.most_common(12)],
    }
    agreement_totals: Counter[str] = Counter()
    for row in rows:
        for key, passed in (row.get("agreement") or {}).items():
            if passed:
                agreement_totals[key] += 1
    for key in sorted(agreement_totals):
        summary[f"{key}_rate"] = agreement_totals[key] / len(rows)
    if kind == "belief":
        non_empty = sum(1 for row in rows if row.get("teacher_label", {}).get("belief_items"))
        summary["non_empty_belief_rate"] = non_empty / len(rows)
    return summary


def _coerce_teacher_label_with_retry(
    kind: str,
    row: dict[str, Any],
    payload: dict[str, Any] | None,
    *,
    provider: Any,
) -> tuple[dict[str, Any] | None, int, list[dict[str, Any]]]:
    sample_id = _task_row_sample_id(row)
    if payload is None:
        return None, 0, []
    try:
        return _coerce_label(kind, row, payload), 0, []
    except ValueError as exc:
        retry_map, retry_requests, retry_failures = _request_teacher_labels(kind, [row], provider=provider)
        failures = [
            {
                **failure,
                "sample_id": failure.get("sample_id") or sample_id,
                "error": failure.get("error", "retry_failed"),
                "retry_after_error": str(exc),
            }
            for failure in retry_failures
        ]
        retry_payload = retry_map.get(sample_id)
        if retry_payload is None:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": "retry_missing_or_invalid_label",
                    "retry_after_error": str(exc),
                    "raw_label": payload,
                }
            )
            return None, retry_requests, failures
        try:
            return _coerce_label(kind, row, retry_payload), retry_requests, failures
        except ValueError as retry_exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": "coercion_failed_after_retry",
                    "retry_after_error": str(exc),
                    "final_error": str(retry_exc),
                    "raw_label": retry_payload,
                }
            )
            return None, retry_requests, failures


def _teacher_task_paths(manifest_payload: dict[str, Any]) -> dict[str, Path]:
    return {
        split_name: Path((split_payload or {}).get("prepared_manifest", ""))
        for split_name, split_payload in (manifest_payload.get("splits") or {}).items()
    }


def _load_teacher_label_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_jsonl(path)
    label_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        teacher_label = dict(row.get("teacher_label") or {})
        if "belief_items" in teacher_label:
            teacher_label["belief_items"] = [
                {
                    **dict(item),
                    "status": _normalize_teacher_belief_status(item.get("status", "unknown")),
                    "time_scope": _normalize_teacher_belief_time_scope(item.get("time_scope", "unknown")),
                }
                for item in (teacher_label.get("belief_items") or [])
                if isinstance(item, dict)
            ]
        label_map[sample_id] = teacher_label
    return label_map


def _teacher_label_changed(row: dict[str, Any]) -> bool:
    source_label = row.get("source_label")
    teacher_label = row.get("teacher_label")
    if not isinstance(teacher_label, dict):
        return False
    if not isinstance(source_label, dict):
        return True
    agreement = row.get("agreement")
    if isinstance(agreement, dict):
        return not all(bool(value) for value in agreement.values())
    return json.dumps(source_label, ensure_ascii=False, sort_keys=True) != json.dumps(
        teacher_label,
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_teacher_changed_label_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    changed_map: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or not _teacher_label_changed(row):
            continue
        teacher_label = dict(row.get("teacher_label") or {})
        if "belief_items" in teacher_label:
            teacher_label["belief_items"] = [
                {
                    **dict(item),
                    "status": _normalize_teacher_belief_status(item.get("status", "unknown")),
                    "time_scope": _normalize_teacher_belief_time_scope(item.get("time_scope", "unknown")),
                }
                for item in (teacher_label.get("belief_items") or [])
                if isinstance(item, dict)
            ]
        changed_map[sample_id] = teacher_label
    return changed_map


def _status_from_hint(status_hint: str) -> str:
    normalized = str(status_hint or "unknown").strip()
    return normalized or "unknown"


def _normalize_teacher_belief_status(status: Any) -> str:
    normalized = str(status or "unknown").strip().lower()
    if normalized in {"active", "affirmed", "asserted", "believed", "belief", "confirmed", "current", "factual", "known", "positive", "relevant", "stable", "verified"}:
        return "active"
    if normalized in {"stale", "historical", "inactive", "irrelevant", "negated", "negative", "previous", "retracted", "superseded"}:
        return "stale"
    if normalized == "conflicted":
        return "conflicted"
    return "unknown"


def _normalize_teacher_belief_time_scope(time_scope: Any) -> str:
    normalized = str(time_scope or "unknown").strip().lower()
    if normalized in {"current", "general", "global", "long_term", "preference", "present", "stable", "static"}:
        return "current"
    if normalized in {"past", "historical", "previous"}:
        return "past"
    if normalized == "future":
        return "future"
    if normalized in {"ephemeral", "instantaneous", "latest", "recent", "recent_change", "temporal"}:
        return "recent_change"
    return "unknown"


def _apply_teacher_label(kind: str, row: dict[str, Any], teacher_label: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(row, ensure_ascii=False))
    if kind == "observation":
        updated["target_record"] = {
            **dict(updated.get("target_record") or {}),
            "relation": teacher_label.get("relation"),
            "value": teacher_label.get("value"),
            "time_scope": teacher_label.get("time_scope"),
            "status": _status_from_hint(str(teacher_label.get("status_hint", "unknown"))),
        }
        return updated
    if kind == "slot_assignment":
        updated["target_action"] = teacher_label.get("target_action")
        updated["target_flags"] = {
            "promote": bool(teacher_label.get("promote", False)),
            "stale_old": bool(teacher_label.get("stale_old", False)),
        }
        matched_slot_id = teacher_label.get("matched_slot_id")
        updated["affected_slot_ids"] = [matched_slot_id] if matched_slot_id else []
        return updated
    updated["target_belief_json"] = {
        "belief_items": list(teacher_label.get("belief_items") or []),
    }
    return updated


def _build_teacher_enhanced_split_manifests(
    output_root: Path,
    *,
    v27_manifest_artifact: Path,
    teacher_root: Path,
) -> dict[str, dict[str, Any]]:
    manifest_payload = json.loads(v27_manifest_artifact.read_text(encoding="utf-8"))
    task_by_kind = {
        "observation": "slot_autoencoding",
        "slot_assignment": "lifecycle_prediction",
        "belief": "composition_to_belief",
    }
    task_kind = {task: kind for kind, task in task_by_kind.items()}
    enhanced_root = output_root / "artifacts" / "stage2_v28_teacher_manifests"
    manifests: dict[str, dict[str, Any]] = {}

    for split_name, manifest_path in _teacher_task_paths(manifest_payload).items():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing prepared manifest for split {split_name}: {manifest_path}")
        base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_root = enhanced_root / split_name
        task_files: dict[str, str] = {}
        teacher_replaced_counts: dict[str, int] = {}

        label_maps = {
            kind: _load_teacher_label_map(teacher_root / kind / f"{split_name}_labels.jsonl")
            for kind in task_by_kind
        }
        for task_name, task_file in (base_manifest.get("task_files") or {}).items():
            rows = _load_jsonl(Path(task_file))
            kind = task_kind.get(task_name)
            replaced = 0
            if kind is not None:
                label_map = label_maps[kind]
                rewritten_rows = []
                for row in rows:
                    sample_id = _task_row_sample_id(row)
                    if sample_id in label_map:
                        rewritten_rows.append(_apply_teacher_label(kind, row, label_map[sample_id]))
                        replaced += 1
                    else:
                        rewritten_rows.append(row)
                rows = rewritten_rows
            teacher_replaced_counts[task_name] = replaced
            target_path = split_root / Path(task_file).name
            _write_jsonl(target_path, rows)
            task_files[task_name] = str(target_path)

        enhanced_manifest = {
            **base_manifest,
            "prepared_by": "stage2_v28_teacher_suite",
            "base_prepared_manifest": str(manifest_path),
            "task_files": task_files,
            "teacher_replaced_counts": teacher_replaced_counts,
        }
        manifest_out = split_root / "stage2_prepared_samples_manifest.json"
        _write_json(manifest_out, enhanced_manifest)
        manifests[split_name] = {
            "prepared_manifest": str(manifest_out),
            "teacher_replaced_counts": teacher_replaced_counts,
            "base_prepared_manifest": str(manifest_path),
        }
    return manifests


def _build_matched_teacher_vs_silver_split_manifests(
    output_root: Path,
    *,
    v27_manifest_artifact: Path,
    teacher_root: Path,
    matched_kinds: tuple[str, ...] = ("observation", "slot_assignment", "belief"),
    teacher_apply_kinds: tuple[str, ...] | None = None,
    manifest_namespace: str = "stage2_v28_matched_manifests",
) -> dict[str, dict[str, dict[str, Any]]]:
    manifest_payload = json.loads(v27_manifest_artifact.read_text(encoding="utf-8"))
    task_by_kind = {
        "observation": "slot_autoencoding",
        "slot_assignment": "lifecycle_prediction",
        "belief": "composition_to_belief",
    }
    task_kind = {task: kind for kind, task in task_by_kind.items()}
    teacher_apply_kinds = teacher_apply_kinds or matched_kinds
    matched_root = output_root / "artifacts" / manifest_namespace
    manifests: dict[str, dict[str, dict[str, Any]]] = {}

    for split_name, manifest_path in _teacher_task_paths(manifest_payload).items():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing prepared manifest for split {split_name}: {manifest_path}")
        base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_root = matched_root / split_name
        silver_root = split_root / "silver"
        teacher_split_root = split_root / "teacher"

        label_maps = {
            kind: _load_teacher_label_map(teacher_root / kind / f"{split_name}_labels.jsonl")
            for kind in matched_kinds
        }
        changed_label_maps = {
            kind: _load_teacher_changed_label_map(teacher_root / kind / f"{split_name}_labels.jsonl")
            for kind in matched_kinds
        }
        matched_sample_ids = set()
        for changed_map in changed_label_maps.values():
            matched_sample_ids.update(changed_map.keys())

        split_payload: dict[str, dict[str, Any]] = {}
        for variant_name, variant_root in (("silver", silver_root), ("teacher", teacher_split_root)):
            task_files: dict[str, str] = {}
            task_counts: dict[str, int] = {}
            teacher_replaced_counts: dict[str, int] = {}
            matched_sample_counts: dict[str, int] = {}
            changed_sample_counts: dict[str, int] = {}

            for task_name, task_file in (base_manifest.get("task_files") or {}).items():
                rows = _load_jsonl(Path(task_file))
                rows = [
                    row
                    for row in rows
                    if _task_row_sample_id(row) in matched_sample_ids
                ]
                replaced = 0
                kind = task_kind.get(task_name)
                if variant_name == "teacher" and kind in teacher_apply_kinds:
                    label_map = label_maps[kind]
                    changed_label_map = changed_label_maps[kind]
                    rewritten_rows = []
                    for row in rows:
                        sample_id = _task_row_sample_id(row)
                        if sample_id in label_map:
                            teacher_label = label_map[sample_id]
                            rewritten_rows.append(_apply_teacher_label(kind, row, teacher_label))
                            if sample_id in changed_label_map:
                                replaced += 1
                        else:
                            rewritten_rows.append(row)
                    rows = rewritten_rows
                elif variant_name == "teacher":
                    rows = list(rows)
                if variant_name == "teacher" and kind in teacher_apply_kinds and not rows:
                    rows = []
                target_path = variant_root / Path(task_file).name
                _write_jsonl(target_path, rows)
                task_files[task_name] = str(target_path)
                task_counts[task_name] = len(rows)
                teacher_replaced_counts[task_name] = replaced
                matched_sample_counts[task_name] = len(rows)
                if kind in matched_kinds:
                    changed_sample_counts[task_name] = sum(
                        1 for row in rows if _task_row_sample_id(row) in changed_label_maps[kind]
                    )
                else:
                    changed_sample_counts[task_name] = 0

            manifest_out = variant_root / "stage2_prepared_samples_manifest.json"
            enhanced_manifest = {
                **base_manifest,
                "prepared_by": "stage2_v28_matched_teacher_vs_silver",
                "base_prepared_manifest": str(manifest_path),
                "variant": variant_name,
                "manifest_namespace": manifest_namespace,
                "matched_kinds": list(matched_kinds),
                "teacher_apply_kinds": list(teacher_apply_kinds),
                "matched_sample_id_count": len(matched_sample_ids),
                "task_files": task_files,
                "task_counts": task_counts,
                "teacher_replaced_counts": teacher_replaced_counts,
                "matched_sample_counts": matched_sample_counts,
                "changed_sample_counts": changed_sample_counts,
            }
            _write_json(manifest_out, enhanced_manifest)
            split_payload[variant_name] = {
                "prepared_manifest": str(manifest_out),
                "task_counts": task_counts,
                "teacher_replaced_counts": teacher_replaced_counts,
                "matched_sample_counts": matched_sample_counts,
                "changed_sample_counts": changed_sample_counts,
            }
        manifests[split_name] = split_payload

    return manifests


def publish_v27_teacher_artifacts(
    output_root: Path,
    *,
    v27_manifest_artifact: Path,
    teacher_config_path: Path,
    train_max_rows: int = 256,
    val_max_rows: int = 64,
    test_max_rows: int = 64,
    batch_size: int = 8,
    provider: Any | None = None,
) -> dict[str, Any]:
    manifest_payload = json.loads(v27_manifest_artifact.read_text(encoding="utf-8"))
    splits = manifest_payload.get("splits") or {}
    split_caps = {
        "train": train_max_rows,
        "val": val_max_rows,
        "test": test_max_rows,
    }
    teacher_meta, live_provider = _provider_from_config(teacher_config_path)
    provider = provider or live_provider
    if not provider.is_configured():
        raise RuntimeError(f"Teacher provider is not configured via {teacher_meta['api_key_env']}.")

    artifact_root = output_root / "artifacts"
    teacher_root = artifact_root / "stage2_v27_teacher"
    commit_hash = _current_commit_hash()
    task_mapping = {
        "observation": "slot_autoencoding",
        "slot_assignment": "lifecycle_prediction",
        "belief": "composition_to_belief",
    }
    published: dict[str, Any] = {}
    for kind, task_name in task_mapping.items():
        label_files: dict[str, str] = {}
        split_metrics: dict[str, Any] = {}
        total_requests = 0
        total_rows: list[dict[str, Any]] = []
        total_selected_examples = 0
        for split_name in ("train", "val", "test"):
            prepared_manifest = Path((splits.get(split_name) or {}).get("prepared_manifest", ""))
            if not prepared_manifest.exists():
                raise FileNotFoundError(f"Missing prepared manifest for split {split_name}: {prepared_manifest}")
            manifest = json.loads(prepared_manifest.read_text(encoding="utf-8"))
            task_file = Path(manifest["task_files"][task_name])
            available_rows = _load_jsonl(task_file)
            selected_rows = _round_robin_sample(
                available_rows,
                kind=kind,
                max_rows=split_caps.get(split_name),
            )
            selected_rows = sorted(selected_rows, key=_task_row_sample_id)
            total_selected_examples += len(selected_rows)
            label_path = teacher_root / kind / f"{split_name}_labels.jsonl"
            failure_path = teacher_root / kind / f"{split_name}_failures.jsonl"
            selected_sample_ids = {_task_row_sample_id(row) for row in selected_rows}
            teacher_row_map = _rows_by_sample_id(label_path, allowed_sample_ids=selected_sample_ids)
            failure_map = {
                sample_id: row
                for sample_id, row in _rows_by_sample_id(
                    failure_path,
                    allowed_sample_ids=selected_sample_ids,
                ).items()
                if sample_id not in teacher_row_map
            }
            reused_count = len(teacher_row_map)
            split_request_count = 0
            pending_rows = [
                row
                for row in selected_rows
                if _task_row_sample_id(row) not in teacher_row_map
            ]
            for batch in _batched(pending_rows, max(batch_size, 1)):
                if not batch:
                    continue
                label_map, request_count, failures = _request_teacher_labels(kind, batch, provider=provider)
                for failure in failures:
                    sample_id = str(failure.get("sample_id", "")).strip()
                    if not sample_id:
                        continue
                    failure_map[sample_id] = {**failure, "split": split_name, "task_name": task_name}
                for row in batch:
                    sample_id = _task_row_sample_id(row)
                    if sample_id not in label_map:
                        continue
                    teacher_label, retry_requests, retry_failures = _coerce_teacher_label_with_retry(
                        kind,
                        row,
                        label_map[sample_id],
                        provider=provider,
                    )
                    total_requests += retry_requests
                    split_request_count += retry_requests
                    for failure in retry_failures:
                        failure_sample_id = str(failure.get("sample_id") or sample_id).strip()
                        if not failure_sample_id:
                            continue
                        failure_map[failure_sample_id] = {
                            **failure,
                            "sample_id": failure_sample_id,
                            "split": split_name,
                            "task_name": task_name,
                        }
                    if teacher_label is None:
                        continue
                    source_label = _source_label(kind, row)
                    agreement = _agreement_metrics(kind, source_label, teacher_label)
                    relation = _task_row_relation(kind, row)
                    teacher_row_map[sample_id] = {
                        "split": split_name,
                        "task_name": task_name,
                        "dataset": _task_row_dataset(row),
                        "relation": relation,
                        "sample_id": sample_id,
                        "source_label": source_label,
                        "teacher_label": teacher_label,
                        "agreement": agreement,
                    }
                    failure_map.pop(sample_id, None)
                current_teacher_rows = [
                    teacher_row_map[sample_id]
                    for sample_id in sorted(teacher_row_map)
                ]
                current_failures = [
                    failure_map[sample_id]
                    for sample_id in sorted(failure_map)
                ]
                _write_jsonl(label_path, current_teacher_rows)
                _write_jsonl(failure_path, current_failures)
                total_requests += request_count
                split_request_count += request_count
            teacher_rows = [
                teacher_row_map[_task_row_sample_id(row)]
                for row in selected_rows
                if _task_row_sample_id(row) in teacher_row_map
            ]
            teacher_failures = [
                failure_map[_task_row_sample_id(row)]
                for row in selected_rows
                if _task_row_sample_id(row) in failure_map
            ]
            _write_jsonl(label_path, teacher_rows)
            _write_jsonl(failure_path, teacher_failures)
            label_files[split_name] = str(label_path)
            split_metrics[split_name] = {
                "requested_examples": min(split_caps.get(split_name) or len(available_rows), len(available_rows)),
                "available_examples": len(available_rows),
                "selected_examples": len(selected_rows),
                "request_count": split_request_count,
                "failed_examples": len(teacher_failures),
                "reused_labeled_examples": reused_count,
                "failure_path": str(failure_path),
                **_summarize_teacher_rows(kind, teacher_rows),
            }
            total_rows.extend(teacher_rows)
        artifact_payload = {
            "artifact_type": f"stage2_v27_teacher_{kind}",
            "status": "completed" if all((split_metrics[name]["failed_examples"] == 0) for name in split_metrics) else "completed_with_failures",
            "commit_hash": commit_hash,
            "v27_manifest_artifact": str(v27_manifest_artifact),
            "teacher": teacher_meta,
            "prompt_version": f"v27_teacher_{kind}_v1",
            "selection_strategy": "dataset_relation_round_robin",
            "batch_size": batch_size,
            "sample_caps": split_caps,
            "label_files": label_files,
            "total_request_count": total_requests,
            "total_requested_examples": total_selected_examples,
            "total_labeled_examples": len(total_rows),
            "total_failed_examples": sum(int(split_metrics[name]["failed_examples"]) for name in split_metrics),
            "success_rate": (len(total_rows) / total_selected_examples) if total_selected_examples else 0.0,
            "split_metrics": split_metrics,
        }
        latest_path = artifact_root / f"latest_stage2_v27_teacher_{kind}.json"
        stamped_path = artifact_root / f"{_timestamp()}_latest_stage2_v27_teacher_{kind}.json"
        _write_json(latest_path, artifact_payload)
        _write_json(stamped_path, artifact_payload)
        published[kind] = {
            "artifact_path": str(latest_path),
            "stamped_artifact_path": str(stamped_path),
            **artifact_payload,
        }
    return published


def publish_v28_teacher_suite(
    output_root: Path,
    *,
    v27_manifest_artifact: Path,
    teacher_config_path: Path,
    train_max_rows: int = 512,
    val_max_rows: int = 128,
    test_max_rows: int = 128,
    batch_size: int = 8,
    provider: Any | None = None,
) -> dict[str, Any]:
    commit_hash = _current_commit_hash()
    working_root = output_root / "_stage2_v28_teacher_tmp"
    published_v27 = publish_v27_teacher_artifacts(
        working_root,
        v27_manifest_artifact=v27_manifest_artifact,
        teacher_config_path=teacher_config_path,
        train_max_rows=train_max_rows,
        val_max_rows=val_max_rows,
        test_max_rows=test_max_rows,
        batch_size=batch_size,
        provider=provider,
    )
    artifact_root = output_root / "artifacts"
    teacher_root = working_root / "artifacts" / "stage2_v27_teacher"
    teacher_manifests = _build_teacher_enhanced_split_manifests(
        output_root,
        v27_manifest_artifact=v27_manifest_artifact,
        teacher_root=teacher_root,
    )
    matched_manifests = _build_matched_teacher_vs_silver_split_manifests(
        output_root,
        v27_manifest_artifact=v27_manifest_artifact,
        teacher_root=teacher_root,
    )

    published: dict[str, Any] = {}
    for kind, payload in published_v27.items():
        v28_payload = {
            **payload,
            "artifact_type": payload["artifact_type"].replace("stage2_v27", "stage2_v28"),
            "prompt_version": str(payload.get("prompt_version", "")).replace("v27_", "v28_", 1),
            "v27_manifest_artifact": str(v27_manifest_artifact),
            "teacher_enhanced_manifests": teacher_manifests,
            "matched_teacher_vs_silver_manifests": matched_manifests,
        }
        latest_path = artifact_root / f"latest_stage2_v28_teacher_{kind}.json"
        stamped_path = artifact_root / f"{_timestamp()}_latest_stage2_v28_teacher_{kind}.json"
        _write_json(latest_path, v28_payload)
        _write_json(stamped_path, v28_payload)
        published[kind] = {
            **v28_payload,
            "artifact_path": str(latest_path),
            "stamped_artifact_path": str(stamped_path),
        }

    quality_audit = {
        "artifact_type": "stage2_v28_teacher_quality_audit",
        "commit_hash": commit_hash,
        "tasks_covered": ["observation", "slot_assignment", "belief"],
        "sample_caps": {"train": train_max_rows, "val": val_max_rows, "test": test_max_rows},
        "teacher_artifacts": {
            kind: payload["artifact_path"]
            for kind, payload in published.items()
        },
        "teacher_enhanced_manifests": teacher_manifests,
        "matched_teacher_vs_silver_manifests": matched_manifests,
        "observation_success_rate": float(published["observation"].get("success_rate", 0.0)),
        "slot_assignment_success_rate": float(published["slot_assignment"].get("success_rate", 0.0)),
        "belief_success_rate": float(published["belief"].get("success_rate", 0.0)),
    }
    latest_audit = artifact_root / "latest_stage2_v28_teacher_quality_audit.json"
    stamped_audit = artifact_root / f"{_timestamp()}_latest_stage2_v28_teacher_quality_audit.json"
    _write_json(latest_audit, quality_audit)
    _write_json(stamped_audit, quality_audit)

    return {
        **published,
        "quality_audit": {
            **quality_audit,
            "artifact_path": str(latest_audit),
            "stamped_artifact_path": str(stamped_audit),
        },
        "teacher_enhanced_manifests": teacher_manifests,
    }


def prepare_v27_source_split(
    output_root: Path,
    *,
    source_config: Path,
    train_source_rows: int = 24000,
    val_source_rows: int = 4000,
    test_source_rows: int = 4000,
    split_seed: int = 27,
) -> dict[str, object]:
    source_summary = summarize_stage2_sources(source_config)
    split_payload = build_source_record_split_from_sources(
        source_config,
        split_counts={
            "train": train_source_rows,
            "val": val_source_rows,
            "test": test_source_rows,
        },
        split_seed=split_seed,
    )

    split_root = output_root / "artifacts" / "stage2_v27_32k"
    split_root.mkdir(parents=True, exist_ok=True)
    split_manifests: dict[str, dict[str, object]] = {}
    audit_splits: dict[str, dict[str, object]] = {}
    for split_name, rows in split_payload["split_rows"].items():
        selection = _selection_map(rows)
        prepared_payload = build_prepared_payload_from_sources(
            source_config,
            selected_sample_ids=selection,
        )
        source_rows_path = split_root / f"{split_name}_source_rows.jsonl"
        _write_jsonl(source_rows_path, rows)
        manifest_payload = _materialize_prepared_payload(
            output_root,
            prepared_payload=prepared_payload,
            source_config=source_config,
            source_summary=source_summary,
            max_rows_per_dataset=None,
            mode="source_level_split",
            namespace=f"stage2_v27_32k/{split_name}",
            extra_manifest={
                "source_split": split_name,
                "source_row_count": len(rows),
                "source_rows_path": str(source_rows_path),
                "split_seed": split_seed,
            },
        )
        split_manifests[split_name] = manifest_payload
        audit_splits[split_name] = {
            "source_row_count": len(rows),
            "task_counts": manifest_payload["task_counts"],
            "dataset_counts": _top_counter(rows, "dataset_key", limit=16),
            "relation_counts": _top_counter(rows, "relation", limit=16),
            "parser_bucket_counts": _top_counter(rows, "parser_bucket", limit=16),
            "lifecycle_source_rows": sum(1 for row in rows if bool(row.get("has_lifecycle"))),
        }

    latest_split_path = output_root / "artifacts" / "latest_stage2_v27_32k_split.json"
    latest_manifest_path = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    latest_audit_path = output_root / "artifacts" / "latest_stage2_v27_32k_audit.json"

    _write_json(
        latest_split_path,
        {
            "artifact_type": "stage2_v27_32k_source_split",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "available_source_rows": split_payload["available_source_rows"],
            "available_by_dataset": split_payload["available_by_dataset"],
            "split_counts": {name: len(rows) for name, rows in split_payload["split_rows"].items()},
            "split_rows_paths": {
                split_name: str(split_root / f"{split_name}_source_rows.jsonl")
                for split_name in split_payload["split_rows"]
            },
        },
    )
    _write_json(
        latest_manifest_path,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "splits": split_manifests,
        },
    )
    _write_json(
        latest_audit_path,
        {
            "artifact_type": "stage2_v27_32k_audit",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "available_source_rows": split_payload["available_source_rows"],
            "available_by_dataset": split_payload["available_by_dataset"],
            "splits": audit_splits,
        },
    )
    return {
        "v27_split_artifact": str(latest_split_path),
        "v27_manifest_artifact": str(latest_manifest_path),
        "v27_audit_artifact": str(latest_audit_path),
        "splits": {
            split_name: {
                "source_rows_path": str(split_root / f"{split_name}_source_rows.jsonl"),
                **manifest_payload,
            }
            for split_name, manifest_payload in split_manifests.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--source-config")
    parser.add_argument("--strict-sources", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--prepare-v27-32k", action="store_true")
    parser.add_argument("--v27-train-source-rows", type=int, default=24000)
    parser.add_argument("--v27-val-source-rows", type=int, default=4000)
    parser.add_argument("--v27-test-source-rows", type=int, default=4000)
    parser.add_argument("--v27-split-seed", type=int, default=27)
    parser.add_argument("--publish-v27-teacher-artifacts", action="store_true")
    parser.add_argument("--v27-manifest-artifact", default="outputs_v2/artifacts/latest_stage2_v27_32k_manifest.json")
    parser.add_argument("--teacher-config", default="configs/minimax_m27.yaml")
    parser.add_argument("--teacher-train-max-rows", type=int, default=256)
    parser.add_argument("--teacher-val-max-rows", type=int, default=64)
    parser.add_argument("--teacher-test-max-rows", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--publish-v28-teacher-suite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.publish_v28_teacher_suite:
        payload = publish_v28_teacher_suite(
            Path(args.output_root),
            v27_manifest_artifact=Path(args.v27_manifest_artifact),
            teacher_config_path=Path(args.teacher_config),
            train_max_rows=args.teacher_train_max_rows,
            val_max_rows=args.teacher_val_max_rows,
            test_max_rows=args.teacher_test_max_rows,
            batch_size=args.teacher_batch_size,
        )
    elif args.publish_v27_teacher_artifacts:
        payload = publish_v27_teacher_artifacts(
            Path(args.output_root),
            v27_manifest_artifact=Path(args.v27_manifest_artifact),
            teacher_config_path=Path(args.teacher_config),
            train_max_rows=args.teacher_train_max_rows,
            val_max_rows=args.teacher_val_max_rows,
            test_max_rows=args.teacher_test_max_rows,
            batch_size=args.teacher_batch_size,
        )
    elif args.prepare_v27_32k:
        if not args.source_config:
            raise ValueError("--prepare-v27-32k requires --source-config")
        payload = prepare_v27_source_split(
            Path(args.output_root),
            source_config=Path(args.source_config),
            train_source_rows=args.v27_train_source_rows,
            val_source_rows=args.v27_val_source_rows,
            test_source_rows=args.v27_test_source_rows,
            split_seed=args.v27_split_seed,
        )
    else:
        payload = prepare_stage2_data(
            Path(args.output_root),
            source_config=Path(args.source_config) if args.source_config else None,
            strict_sources=args.strict_sources,
            max_rows_per_dataset=args.max_rows_per_dataset,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if args.publish_v28_teacher_suite:
            print(f"teacher_observation_artifact={payload['observation']['artifact_path']}")
            print(f"teacher_slot_assignment_artifact={payload['slot_assignment']['artifact_path']}")
            print(f"teacher_belief_artifact={payload['belief']['artifact_path']}")
            print(f"teacher_quality_audit_artifact={payload['quality_audit']['artifact_path']}")
        elif args.publish_v27_teacher_artifacts:
            print(f"teacher_observation_artifact={payload['observation']['artifact_path']}")
            print(f"teacher_slot_assignment_artifact={payload['slot_assignment']['artifact_path']}")
            print(f"teacher_belief_artifact={payload['belief']['artifact_path']}")
        elif args.prepare_v27_32k:
            print(f"v27_split_artifact={payload['v27_split_artifact']}")
            print(f"v27_manifest_artifact={payload['v27_manifest_artifact']}")
            print(f"v27_audit_artifact={payload['v27_audit_artifact']}")
        else:
            print(f"prepared_manifest={payload['prepared_manifest']}")
            print(f"task_registry={payload['task_registry']}")
            print(f"dataset_registry={payload['dataset_registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
