"""Normalization helpers for public stage-2 datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable
import zipfile

from core_mem.v2.lifecycle import LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.schemas import BeliefItem, BeliefState, Observation, SlotRecord, SoftRoleScores


DEFAULT_TIMESTAMP = "2026-04-13T00:00:00Z"
PERSONACHAT_TRAIN_FILE = "train_self_revised.txt"
LIFECYCLE = LifecycleManager()


def _hash_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _dense_vector(text: str, *, dim: int = 8, salt: str = "") -> list[float]:
    seed = f"{salt}|{text}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    raw = bytearray(digest)
    while len(raw) < dim:
        digest = hashlib.sha256(digest).digest()
        raw.extend(digest)
    return [round(byte / 255.0, 4) for byte in raw[:dim]]


def _latent_tokens(text: str, *, rows: int = 2, dim: int = 8) -> list[list[float]]:
    return [_dense_vector(text, dim=dim, salt=f"latent_{index}") for index in range(rows)]


def _slot_relation(slot_name: str) -> tuple[str, str]:
    lowered = slot_name.lower()
    if lowered in {"food", "cuisine"}:
        return "food_preference", "preference"
    if lowered in {"city", "area", "destination", "departure", "street_address"}:
        return "location", "location"
    if lowered in {"occupation", "job", "profession"}:
        return "occupation", "occupation"
    if lowered in {"intent", "goal"}:
        return "goal", "goal"
    if any(token in lowered for token in ("time", "day", "date", "leave", "arrive", "stay")):
        return "temporal_fact", "other"
    return "constraint", "constraint"


def _soft_roles(observation: Observation) -> SoftRoleScores:
    kwargs = {
        "stable": 0.75 if observation.time_scope == "current" else 0.3,
        "preference": 0.0,
        "constraint": 0.0,
        "goal": 0.0,
        "temporal": 0.7 if observation.time_scope in {"recent_change", "past", "future"} else 0.2,
        "social": 0.2 if observation.relation == "relationship" else 0.0,
    }
    if observation.value_type == "preference":
        kwargs["preference"] = 0.9
    elif observation.value_type == "constraint":
        kwargs["constraint"] = 0.9
    elif observation.value_type == "goal":
        kwargs["goal"] = 0.9
    elif observation.value_type in {"profile", "occupation", "location"}:
        kwargs["stable"] = max(kwargs["stable"], 0.85)
    return SoftRoleScores(**kwargs)


def build_observation(
    *,
    dataset: str,
    dialogue_id: str,
    turn_id: str,
    relation: str,
    value: str,
    evidence_text: str,
    speaker: str = "user",
    session_id: str | None = None,
    value_type: str = "other",
    time_scope: str = "current",
    status_hint: str = "active",
    polarity: str = "positive",
    confidence: float = 0.9,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    relation = relation or "other_fact"
    canonical_gloss = f"{relation}={value}"
    return Observation(
        obs_id=f"obs_{_hash_id(dataset, dialogue_id, turn_id, canonical_gloss)}",
        source_dataset=dataset,
        source_dialogue_id=dialogue_id,
        source_turn_id=turn_id,
        session_id=session_id or dialogue_id,
        speaker=speaker,
        entity="user",
        relation=relation,
        value=value,
        value_type=value_type,
        time_scope=time_scope,
        status_hint=status_hint,
        polarity=polarity,
        confidence=confidence,
        evidence_text=evidence_text,
        canonical_gloss=canonical_gloss,
        metadata=metadata or {},
    )


def build_slot(
    observation: Observation,
    *,
    slot_suffix: str,
    bank: str,
    active_flag: bool,
    revision_parent: str | None = None,
    revision_count: int = 0,
    timestamp: str = DEFAULT_TIMESTAMP,
) -> SlotRecord:
    slot_id = f"slot_{slot_suffix}_{_hash_id(observation.obs_id, bank, observation.value, revision_count)}"
    return SlotRecord(
        slot_id=slot_id,
        bank=bank,
        entity=observation.entity,
        relation=observation.relation,
        retrieval_key=_dense_vector(f"{observation.relation}|{observation.value}", salt=slot_id),
        latent_tokens=_latent_tokens(f"{observation.canonical_gloss}|{slot_id}"),
        soft_role_scores=_soft_roles(observation),
        confidence=observation.confidence,
        first_seen_ts=timestamp,
        last_update_ts=timestamp,
        revision_count=revision_count,
        active_flag=active_flag,
        revision_parent=revision_parent,
        canonical_gloss=observation.canonical_gloss,
    )


def build_negative_slot(
    observation: Observation,
    *,
    fallback_value: str = "unknown",
    timestamp: str = DEFAULT_TIMESTAMP,
) -> SlotRecord:
    negative_observation = build_observation(
        dataset=observation.source_dataset,
        dialogue_id=observation.source_dialogue_id,
        turn_id=f"{observation.source_turn_id}_neg",
        relation=observation.relation,
        value=fallback_value,
        evidence_text=f"Synthetic negative for {observation.canonical_gloss}",
        speaker=observation.speaker,
        session_id=observation.session_id,
        value_type=observation.value_type,
        time_scope="past" if observation.time_scope == "current" else observation.time_scope,
        status_hint="stale",
        polarity="neutral",
        confidence=min(observation.confidence, 0.5),
        metadata={"synthetic": True},
    )
    return build_slot(
        negative_observation,
        slot_suffix="negative",
        bank="core",
        active_flag=False,
        timestamp=timestamp,
    )


def _belief_for_slot(
    *,
    query_id: str,
    query_type: str,
    slot: SlotRecord,
    time_scope: str,
) -> dict[str, Any]:
    belief = BeliefState(
        query_id=query_id,
        entity="user",
        query_type=query_type,
        belief_items=[
            BeliefItem(
                relation=slot.relation,
                value=slot.canonical_gloss.split("=", 1)[-1],
                status="active" if slot.active_flag else "stale",
                time_scope=time_scope,
                confidence=slot.confidence,
                support_slot_ids=[slot.slot_id],
            )
        ],
        global_consistency="high",
    )
    return belief.to_dict()


def _default_query(observation: Observation, *, raw_slot_name: str | None = None) -> str:
    if observation.relation == "food_preference":
        return "What food does the user currently prefer?"
    if observation.relation == "drink_preference":
        return "What drink does the user currently prefer?"
    if observation.relation == "music_preference":
        return "What music does the user currently prefer?"
    if observation.relation == "location":
        return "Where is the relevant location for the user?"
    if observation.relation == "goal":
        return "What is the user's current goal?"
    if observation.relation == "occupation":
        return "What is the user's occupation?"
    if raw_slot_name:
        return f"What is the current value of {raw_slot_name}?"
    return "What is the latest fact about the user?"


def build_task_row(
    *,
    dataset: str,
    sample_id: str,
    observation: Observation,
    query: str,
    current_slot: SlotRecord,
    previous_slot: SlotRecord | None = None,
    negative_slot: SlotRecord | None = None,
    query_type: str = "single_fact",
) -> dict[str, Any]:
    negative = negative_slot or previous_slot or build_negative_slot(observation)
    lifecycle_payload = None
    if previous_slot is not None:
        pre_update_slot = SlotRecord(
            slot_id=previous_slot.slot_id,
            bank=previous_slot.bank,
            entity=previous_slot.entity,
            relation=previous_slot.relation,
            retrieval_key=list(previous_slot.retrieval_key),
            latent_tokens=[list(token) for token in previous_slot.latent_tokens],
            soft_role_scores=previous_slot.soft_role_scores,
            confidence=previous_slot.confidence,
            first_seen_ts=previous_slot.first_seen_ts,
            last_update_ts=previous_slot.last_update_ts,
            revision_count=previous_slot.revision_count,
            active_flag=True,
            revision_parent=previous_slot.revision_parent,
            canonical_gloss=previous_slot.canonical_gloss,
        )
        decision = LIFECYCLE.decide(observation, [pre_update_slot])
        lifecycle_payload = {
            "memory_context": [pre_update_slot.to_dict()],
            "new_observation": observation.to_dict(),
            "target_action": decision.action,
            "target_flags": {"promote": decision.promote, "stale_old": decision.stale_old},
            "affected_slot_ids": [pre_update_slot.slot_id],
        }
    row = {
        "dataset": dataset,
        "sample_id": sample_id,
        "tasks": {
            "slot_autoencoding": {
                "input_observation": observation.to_dict(),
                "target_record": {
                    "entity": observation.entity,
                    "relation": observation.relation,
                    "value": observation.value,
                    "time_scope": observation.time_scope,
                    "status": "active",
                },
            },
            "retrieval_alignment": {
                "query": query,
                "positive_slot": current_slot.to_dict(),
                "negative_slots": [negative.to_dict()],
                "gold_support_slot_ids": [current_slot.slot_id],
            },
            "composition_to_belief": {
                "query": query,
                "memory_slots": [slot.to_dict() for slot in [current_slot, negative] if slot is not None],
                "target_belief_json": _belief_for_slot(
                    query_id=f"belief_{sample_id}",
                    query_type=query_type,
                    slot=current_slot,
                    time_scope=observation.time_scope if observation.time_scope != "recent_change" else "current",
                ),
                "retrieved_slot_ids": [current_slot.slot_id],
            },
        },
    }
    if lifecycle_payload is not None:
        row["tasks"]["lifecycle_prediction"] = lifecycle_payload
    return row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def normalize_sgd_dataset(raw_root: Path, output_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    emitted = 0
    for dialogue_file in sorted((raw_root / "train").glob("dialogues_*.json")):
        dialogues = json.loads(dialogue_file.read_text(encoding="utf-8"))
        for dialogue in dialogues:
            state: dict[str, str] = {}
            dialogue_id = str(dialogue["dialogue_id"])
            for turn_index, turn in enumerate(dialogue.get("turns", [])):
                if turn.get("speaker") != "USER":
                    continue
                utterance = turn.get("utterance", "").strip()
                for frame_index, frame in enumerate(turn.get("frames", [])):
                    service = frame.get("service", "service")
                    slot_values = frame.get("state", {}).get("slot_values", {})
                    for slot_name, values in slot_values.items():
                        if not values:
                            continue
                        value = str(values[0]).strip()
                        key = f"{service}.{slot_name}"
                        previous_value = state.get(key)
                        if not value or previous_value == value:
                            continue
                        relation, value_type = _slot_relation(slot_name)
                        time_scope = "recent_change" if previous_value is not None else "current"
                        observation = build_observation(
                            dataset="sgd",
                            dialogue_id=dialogue_id,
                            turn_id=f"turn_{turn_index}_{frame_index}_{slot_name}",
                            relation=relation,
                            value=value,
                            evidence_text=utterance,
                            value_type=value_type,
                            time_scope=time_scope,
                            metadata={"service": service, "slot_name": slot_name},
                        )
                        current_slot = build_slot(
                            observation,
                            slot_suffix=f"{service}_{slot_name}_current",
                            bank="residual",
                            active_flag=True,
                            revision_count=1 if previous_value is not None else 0,
                        )
                        previous_slot = None
                        if previous_value is not None:
                            previous_observation = build_observation(
                                dataset="sgd",
                                dialogue_id=dialogue_id,
                                turn_id=f"turn_{turn_index}_{frame_index}_{slot_name}_prev",
                                relation=relation,
                                value=previous_value,
                                evidence_text=f"Previous value for {service}.{slot_name}",
                                value_type=value_type,
                                time_scope="past",
                                status_hint="stale",
                                polarity="neutral",
                                confidence=0.8,
                                metadata={"service": service, "slot_name": slot_name, "previous": True},
                            )
                            previous_slot = build_slot(
                                previous_observation,
                                slot_suffix=f"{service}_{slot_name}_previous",
                                bank="core",
                                active_flag=False,
                                revision_parent=current_slot.slot_id,
                                revision_count=0,
                            )
                        rows.append(
                            build_task_row(
                                dataset="sgd",
                                sample_id=f"sgd-{dialogue_id}-{service}-{slot_name}-{emitted}",
                                observation=observation,
                                query=_default_query(observation, raw_slot_name=f"{service}.{slot_name}"),
                                current_slot=current_slot,
                                previous_slot=previous_slot,
                                query_type="update_sensitive" if previous_slot is not None else "single_fact",
                            )
                        )
                        state[key] = value
                        emitted += 1
                        if max_rows is not None and emitted >= max_rows:
                            count = _write_jsonl(output_path, rows)
                            return {"dataset": "sgd", "rows": count, "output_path": str(output_path)}
    count = _write_jsonl(output_path, rows)
    return {"dataset": "sgd", "rows": count, "output_path": str(output_path)}


def _flatten_multiwoz_state(metadata: dict[str, Any]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for domain, domain_payload in metadata.items():
        if not isinstance(domain_payload, dict):
            continue
        for section_name, section in domain_payload.items():
            if not isinstance(section, dict):
                continue
            for slot_name, raw_value in section.items():
                if slot_name == "booked":
                    continue
                value = str(raw_value).strip()
                if not value or value in {"not mentioned", "none"}:
                    continue
                flattened[f"{domain}.{section_name}.{slot_name}"] = value
    return flattened


def normalize_multiwoz24_dataset(zip_path: Path, output_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as bundle:
        data = json.loads(bundle.read("MULTIWOZ2.4/data.json").decode("utf-8"))
        val_ids = {
            line.strip()
            for line in bundle.read("MULTIWOZ2.4/valListFile.json").decode("utf-8").splitlines()
            if line.strip()
        }
        test_ids = {
            line.strip()
            for line in bundle.read("MULTIWOZ2.4/testListFile.json").decode("utf-8").splitlines()
            if line.strip()
        }

    emitted = 0
    for dialogue_id, payload in data.items():
        if dialogue_id in val_ids or dialogue_id in test_ids:
            continue
        state: dict[str, str] = {}
        log = payload.get("log", [])
        for index in range(1, len(log), 2):
            user_turn = log[index - 1].get("text", "").strip()
            metadata = log[index].get("metadata", {})
            current_state = _flatten_multiwoz_state(metadata)
            for key, value in current_state.items():
                previous_value = state.get(key)
                if previous_value == value:
                    continue
                domain, _section, slot_name = key.split(".", 2)
                relation, value_type = _slot_relation(slot_name)
                observation = build_observation(
                    dataset="multiwoz24",
                    dialogue_id=dialogue_id,
                    turn_id=f"log_{index}_{slot_name}",
                    relation=relation,
                    value=value,
                    evidence_text=user_turn,
                    value_type=value_type,
                    time_scope="recent_change" if previous_value is not None else "current",
                    metadata={"domain": domain, "slot_name": slot_name},
                )
                current_slot = build_slot(
                    observation,
                    slot_suffix=f"{domain}_{slot_name}_current",
                    bank="residual",
                    active_flag=True,
                    revision_count=1 if previous_value is not None else 0,
                )
                previous_slot = None
                if previous_value is not None:
                    previous_observation = build_observation(
                        dataset="multiwoz24",
                        dialogue_id=dialogue_id,
                        turn_id=f"log_{index}_{slot_name}_prev",
                        relation=relation,
                        value=previous_value,
                        evidence_text=f"Previous value for {key}",
                        value_type=value_type,
                        time_scope="past",
                        status_hint="stale",
                        polarity="neutral",
                        confidence=0.8,
                        metadata={"domain": domain, "slot_name": slot_name, "previous": True},
                    )
                    previous_slot = build_slot(
                        previous_observation,
                        slot_suffix=f"{domain}_{slot_name}_previous",
                        bank="core",
                        active_flag=False,
                        revision_parent=current_slot.slot_id,
                    )
                rows.append(
                    build_task_row(
                        dataset="multiwoz24",
                        sample_id=f"multiwoz24-{dialogue_id}-{domain}-{slot_name}-{emitted}",
                        observation=observation,
                        query=_default_query(observation, raw_slot_name=f"{domain}.{slot_name}"),
                        current_slot=current_slot,
                        previous_slot=previous_slot,
                        query_type="update_sensitive" if previous_slot is not None else "single_fact",
                    )
                )
                state[key] = value
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    count = _write_jsonl(output_path, rows)
                    return {"dataset": "multiwoz24", "rows": count, "output_path": str(output_path)}
    count = _write_jsonl(output_path, rows)
    return {"dataset": "multiwoz24", "rows": count, "output_path": str(output_path)}


def _fallback_persona_observation(text: str, *, dialogue_id: str, turn_id: str) -> Observation:
    lowered = text.lower()
    relation = "profile_trait"
    value_type = "profile"
    if "like" in lowered or "love" in lowered or "enjoy" in lowered or "favorite" in lowered:
        relation = "hobby"
        value_type = "preference"
    return build_observation(
        dataset="personachat",
        dialogue_id=dialogue_id,
        turn_id=turn_id,
        relation=relation,
        value=text,
        evidence_text=text,
        value_type=value_type,
        time_scope="current",
        confidence=0.75,
        metadata={"parser": "fallback"},
    )


def _iter_personachat_personas(tar_path: Path) -> Iterable[tuple[int, list[str]]]:
    with tarfile.open(tar_path, "r:gz") as bundle:
        member = next(item for item in bundle.getmembers() if item.name.endswith(PERSONACHAT_TRAIN_FILE))
        handle = bundle.extractfile(member)
        if handle is None:
            return
        current: list[str] = []
        dialogue_index = 0
        for raw_line in handle:
            line = raw_line.decode("utf-8").rstrip("\n")
            if not line:
                continue
            prefix, rest = line.split(" ", 1)
            if prefix == "1" and rest.startswith("your persona:") and current:
                dialogue_index += 1
                yield dialogue_index, current
                current = []
            if rest.startswith("your persona:"):
                current.append(rest.replace("your persona:", "", 1).strip())
        if current:
            dialogue_index += 1
            yield dialogue_index, current


def normalize_personachat_dataset(tar_path: Path, output_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    parser = Stage2ObservationParser()
    rows: list[dict[str, Any]] = []
    emitted = 0
    for dialogue_index, persona_lines in _iter_personachat_personas(tar_path):
        previous_slot = None
        for persona_index, persona_text in enumerate(persona_lines):
            dialogue_id = f"personachat_{dialogue_index}"
            turn_id = f"persona_{persona_index}"
            parsed = parser.parse_turn(
                persona_text,
                source_dataset="personachat",
                source_dialogue_id=dialogue_id,
                source_turn_id=turn_id,
                session_id=dialogue_id,
            )
            observation = parsed[0] if parsed else _fallback_persona_observation(persona_text, dialogue_id=dialogue_id, turn_id=turn_id)
            current_slot = build_slot(
                observation,
                slot_suffix=f"persona_{persona_index}",
                bank="residual",
                active_flag=True,
                revision_count=0,
            )
            rows.append(
                build_task_row(
                    dataset="personachat",
                    sample_id=f"personachat-{dialogue_index}-{persona_index}",
                    observation=observation,
                    query=_default_query(observation),
                    current_slot=current_slot,
                    negative_slot=previous_slot or build_negative_slot(observation),
                    query_type="single_fact",
                )
            )
            previous_slot = current_slot
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                count = _write_jsonl(output_path, rows)
                return {"dataset": "personachat", "rows": count, "output_path": str(output_path)}
    count = _write_jsonl(output_path, rows)
    return {"dataset": "personachat", "rows": count, "output_path": str(output_path)}


def normalize_mquake_dataset(raw_root: Path, output_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    dataset_path = raw_root / "datasets" / "MQuAKE-CF-3k-v2.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(payload):
        case_id = str(case.get("case_id", index))
        old_answer = str(case.get("answer", "")).strip()
        new_answer = str(case.get("new_answer", "")).strip()
        if not old_answer or not new_answer:
            continue
        question = str((case.get("questions") or ["What changed?"])[0])
        observation = build_observation(
            dataset="mquake",
            dialogue_id=case_id,
            turn_id="question_0",
            relation="recent_change",
            value=new_answer,
            evidence_text=question,
            speaker="system",
            value_type="other",
            time_scope="recent_change",
            metadata={"source": "MQuAKE-CF-3k-v2"},
        )
        current_slot = build_slot(observation, slot_suffix=f"mquake_{case_id}_new", bank="residual", active_flag=True, revision_count=1)
        previous_observation = build_observation(
            dataset="mquake",
            dialogue_id=case_id,
            turn_id="question_0_prev",
            relation="recent_change",
            value=old_answer,
            evidence_text=f"Pre-edit answer for {question}",
            speaker="system",
            value_type="other",
            time_scope="past",
            status_hint="stale",
            polarity="neutral",
            confidence=0.82,
            metadata={"source": "MQuAKE-CF-3k-v2", "previous": True},
        )
        previous_slot = build_slot(
            previous_observation,
            slot_suffix=f"mquake_{case_id}_old",
            bank="core",
            active_flag=False,
            revision_parent=current_slot.slot_id,
        )
        rows.append(
            build_task_row(
                dataset="mquake",
                sample_id=f"mquake-{case_id}",
                observation=observation,
                query=question,
                current_slot=current_slot,
                previous_slot=previous_slot,
                query_type="update_sensitive",
            )
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    count = _write_jsonl(output_path, rows)
    return {"dataset": "mquake", "rows": count, "output_path": str(output_path)}


def _recoe_files(raw_root: Path) -> list[Path]:
    return sorted((raw_root / "data").glob("*/counterfactual_datapoints_verified_atomic.json"))


def normalize_recoe_dataset(raw_root: Path, output_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    emitted = 0
    for source_file in _recoe_files(raw_root):
        subset = source_file.parent.name
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        for index, item in enumerate(payload):
            question = str(item.get("question", "")).strip()
            old_answer = str(item.get("answer", "")).strip()
            new_answer = str(item.get("counterfactual_answer", "")).strip()
            if not question or not old_answer or not new_answer:
                continue
            dialogue_id = f"{subset}_{index}"
            observation = build_observation(
                dataset="recoe",
                dialogue_id=dialogue_id,
                turn_id="counterfactual_answer",
                relation="recent_change",
                value=new_answer,
                evidence_text=question,
                speaker="system",
                value_type="other",
                time_scope="recent_change",
                metadata={"subset": subset},
            )
            current_slot = build_slot(
                observation,
                slot_suffix=f"recoe_{subset}_{index}_new",
                bank="residual",
                active_flag=True,
                revision_count=1,
            )
            previous_observation = build_observation(
                dataset="recoe",
                dialogue_id=dialogue_id,
                turn_id="answer_prev",
                relation="recent_change",
                value=old_answer,
                evidence_text=f"Pre-edit answer for {question}",
                speaker="system",
                value_type="other",
                time_scope="past",
                status_hint="stale",
                polarity="neutral",
                confidence=0.82,
                metadata={"subset": subset, "previous": True},
            )
            previous_slot = build_slot(
                previous_observation,
                slot_suffix=f"recoe_{subset}_{index}_old",
                bank="core",
                active_flag=False,
                revision_parent=current_slot.slot_id,
            )
            rows.append(
                build_task_row(
                    dataset="recoe",
                    sample_id=f"recoe-{subset}-{index}",
                    observation=observation,
                    query=question,
                    current_slot=current_slot,
                    previous_slot=previous_slot,
                    query_type="update_sensitive",
                )
            )
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                count = _write_jsonl(output_path, rows)
                return {"dataset": "recoe", "rows": count, "output_path": str(output_path)}
    count = _write_jsonl(output_path, rows)
    return {"dataset": "recoe", "rows": count, "output_path": str(output_path)}
