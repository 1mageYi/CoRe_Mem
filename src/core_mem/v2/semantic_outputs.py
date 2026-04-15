"""Semantic-first structured output coercion for stage-2 learned decoding."""

from __future__ import annotations

import json
import re
from typing import Any


_SLOT_ID_RE = re.compile(r"slot_[A-Za-z0-9_-]+")
_QUOTED_FIELD_RE = re.compile(r'"(?P<key>[A-Za-z0-9_]+)"\s*:\s*"(?P<value>[^"]*)"')
_BOOL_FIELD_RE = re.compile(r'"(?P<key>[A-Za-z0-9_]+)"\s*:\s*(?P<value>true|false)', re.IGNORECASE)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _json_candidate(payload: str) -> Any | None:
    stripped = payload.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if ":" in stripped and not stripped.startswith(("{", "[")):
        candidates.append("{" + stripped.strip().strip(",") + "}")
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        candidates.append(stripped[start : end + 1])
    if "[" in stripped and "]" in stripped:
        start = stripped.find("[")
        end = stripped.rfind("]")
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def json_like_payload(payload: Any) -> Any | None:
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, str):
        return _json_candidate(payload)
    return None


def _coerce_retrieval_alignment(payload: Any) -> dict[str, Any] | None:
    candidate = json_like_payload(payload)
    if isinstance(candidate, dict):
        raw_ids = candidate.get("gold_support_slot_ids")
        if isinstance(raw_ids, list):
            support_ids = [str(item) for item in raw_ids if str(item)]
            if support_ids:
                return {"gold_support_slot_ids": _dedupe_preserve_order(support_ids)}
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
    support_ids = _dedupe_preserve_order(_SLOT_ID_RE.findall(text))
    if not support_ids:
        return None
    return {"gold_support_slot_ids": support_ids}


def _coerce_lifecycle_prediction(payload: Any) -> dict[str, Any] | None:
    candidate = json_like_payload(payload)
    action = None
    promote = None
    stale_old = None
    if isinstance(candidate, dict):
        action = candidate.get("target_action")
        flags = candidate.get("target_flags")
        if isinstance(flags, dict):
            promote = flags.get("promote")
            stale_old = flags.get("stale_old")

    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for match in _QUOTED_FIELD_RE.finditer(text):
        key = match.group("key")
        value = match.group("value")
        if key == "target_action" and action is None:
            action = value
    for match in _BOOL_FIELD_RE.finditer(text):
        key = match.group("key")
        value = match.group("value").lower() == "true"
        if key == "promote" and promote is None:
            promote = value
        if key == "stale_old" and stale_old is None:
            stale_old = value

    if action is None and promote is None and stale_old is None:
        return None
    return {
        "target_action": str(action or "unknown"),
        "target_flags": {
            "promote": bool(promote),
            "stale_old": bool(stale_old),
        },
    }


def _belief_items_from_dict(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = candidate.get("belief_items")
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation", "")).strip()
        value = str(item.get("value", "")).strip()
        support_ids = _dedupe_preserve_order([str(slot_id) for slot_id in item.get("support_slot_ids", []) if str(slot_id)])
        if relation or value or support_ids:
            items.append(
                {
                    "relation": relation,
                    "value": value,
                    "support_slot_ids": support_ids,
                }
            )
    return items


def _coerce_composition_to_belief(payload: Any) -> dict[str, Any] | None:
    candidate = json_like_payload(payload)
    if isinstance(candidate, dict):
        items = _belief_items_from_dict(candidate)
        if items:
            response = {"belief_items": items}
            for key in ("query_id", "entity", "query_type", "global_consistency"):
                if key in candidate:
                    response[key] = candidate[key]
            return response

    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
    quoted_fields: dict[str, list[str]] = {}
    for match in _QUOTED_FIELD_RE.finditer(text):
        quoted_fields.setdefault(match.group("key"), []).append(match.group("value"))

    relations = quoted_fields.get("relation", [])
    values = quoted_fields.get("value", [])
    support_ids = _dedupe_preserve_order(_SLOT_ID_RE.findall(text))
    if not relations and not values and not support_ids:
        return None

    item_count = max(len(relations), len(values), 1)
    belief_items: list[dict[str, Any]] = []
    for index in range(item_count):
        relation = relations[index] if index < len(relations) else (relations[-1] if relations else "")
        value = values[index] if index < len(values) else (values[-1] if values else "")
        if not relation and not value and not support_ids:
            continue
        belief_items.append(
            {
                "relation": relation,
                "value": value,
                "support_slot_ids": list(support_ids),
            }
        )
    if not belief_items:
        return None

    response: dict[str, Any] = {"belief_items": belief_items}
    for key in ("query_id", "entity", "query_type", "global_consistency"):
        values_for_key = quoted_fields.get(key)
        if values_for_key:
            response[key] = values_for_key[0]
    return response


def coerce_task_payload(task_name: str, payload: Any) -> dict[str, Any] | None:
    if task_name == "retrieval_alignment":
        return _coerce_retrieval_alignment(payload)
    if task_name == "lifecycle_prediction":
        return _coerce_lifecycle_prediction(payload)
    if task_name == "composition_to_belief":
        return _coerce_composition_to_belief(payload)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        candidate = json_like_payload(payload)
        if isinstance(candidate, dict):
            return candidate
    return None


def render_task_payload(task_name: str, payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    if task_name == "retrieval_alignment":
        canonical = {"gold_support_slot_ids": _dedupe_preserve_order([str(slot_id) for slot_id in payload.get("gold_support_slot_ids", [])])}
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    if task_name == "lifecycle_prediction":
        canonical = {
            "target_action": str(payload.get("target_action", "")),
            "target_flags": {
                "promote": bool((payload.get("target_flags") or {}).get("promote", False)),
                "stale_old": bool((payload.get("target_flags") or {}).get("stale_old", False)),
            },
        }
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    if task_name == "composition_to_belief":
        items = []
        for item in payload.get("belief_items", []):
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "relation": str(item.get("relation", "")),
                    "support_slot_ids": _dedupe_preserve_order([str(slot_id) for slot_id in item.get("support_slot_ids", [])]),
                    "value": str(item.get("value", "")),
                }
            )
        return json.dumps({"belief_items": items}, ensure_ascii=False, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _set_f1(predicted: set[str], target: set[str]) -> float:
    if not predicted and not target:
        return 1.0
    overlap = len(predicted & target)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(target) if target else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def semantic_task_scores(
    task_name: str,
    predicted: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> dict[str, float]:
    if predicted is None or target is None:
        return {"field_accuracy": 0.0, "field_f1": 0.0}

    if task_name == "retrieval_alignment":
        predicted_ids = set(str(slot_id) for slot_id in predicted.get("gold_support_slot_ids", []))
        target_ids = set(str(slot_id) for slot_id in target.get("gold_support_slot_ids", []))
        return {
            "field_accuracy": float(predicted_ids == target_ids),
            "field_f1": _set_f1(predicted_ids, target_ids),
        }

    if task_name == "lifecycle_prediction":
        predicted_flags = predicted.get("target_flags") or {}
        target_flags = target.get("target_flags") or {}
        matches = [
            float(str(predicted.get("target_action", "")) == str(target.get("target_action", ""))),
            float(bool(predicted_flags.get("promote", False)) == bool(target_flags.get("promote", False))),
            float(bool(predicted_flags.get("stale_old", False)) == bool(target_flags.get("stale_old", False))),
        ]
        accuracy = sum(matches) / len(matches)
        return {"field_accuracy": accuracy, "field_f1": accuracy}

    if task_name == "composition_to_belief":
        predicted_items = predicted.get("belief_items", [])
        target_items = target.get("belief_items", [])
        predicted_pairs = {
            f"{item.get('relation', '')}::{item.get('value', '')}"
            for item in predicted_items
            if isinstance(item, dict)
        }
        target_pairs = {
            f"{item.get('relation', '')}::{item.get('value', '')}"
            for item in target_items
            if isinstance(item, dict)
        }
        predicted_support = {
            str(slot_id)
            for item in predicted_items
            if isinstance(item, dict)
            for slot_id in item.get("support_slot_ids", [])
        }
        target_support = {
            str(slot_id)
            for item in target_items
            if isinstance(item, dict)
            for slot_id in item.get("support_slot_ids", [])
        }
        pair_f1 = _set_f1(predicted_pairs, target_pairs)
        support_f1 = _set_f1(predicted_support, target_support)
        return {
            "field_accuracy": 0.5 * float(predicted_pairs == target_pairs) + 0.5 * float(predicted_support == target_support),
            "field_f1": 0.5 * (pair_f1 + support_f1),
        }

    return {"field_accuracy": 0.0, "field_f1": 0.0}
