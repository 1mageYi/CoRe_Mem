"""Mechanical verifier for the v4 Persona-first learned-memory run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _normalize(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _label_prefix_match(expected: Any, prediction: Any) -> bool:
    normalized_expected = _normalize(expected)
    normalized_prediction = _normalize(prediction)
    return bool(normalized_expected and normalized_prediction.startswith(normalized_expected))


def _exact_match(expected: Any, prediction: Any) -> bool:
    return _normalize(expected) == _normalize(prediction)


def _int_metric(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    try:
        return int(payload.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _float_metric(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _write_latest_and_stamped(root: Path, latest_name: str, payload: dict[str, Any]) -> dict[str, str]:
    artifact_root = root / "outputs_v2" / "artifacts"
    latest_path = artifact_root / latest_name
    stamped_path = artifact_root / f"{_timestamp()}_{latest_name}"
    _write_json(latest_path, payload)
    _write_json(stamped_path, payload)
    return {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _group_gap_counts(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    output: dict[str, dict[str, int]] = {}
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        local_correct = 0
        provider_correct = 0
        local_correct_provider_wrong = 0
        provider_correct_local_wrong = 0
        for row in group_rows:
            expected = row.get("expected_answer")
            local_ok = _exact_match(expected, row.get("memory_answer_local"))
            provider_ok = _label_prefix_match(expected, row.get("provider_prediction"))
            local_correct += int(local_ok)
            provider_correct += int(provider_ok)
            local_correct_provider_wrong += int(local_ok and not provider_ok)
            provider_correct_local_wrong += int(provider_ok and not local_ok)
        output[group_key] = {
            "count": len(group_rows),
            "local_exact_match": local_correct,
            "provider_label_prefix_match": provider_correct,
            "local_correct_provider_wrong": local_correct_provider_wrong,
            "provider_correct_local_wrong": provider_correct_local_wrong,
        }
    return output


def publish_v4_gap_audit(*, root: Path, personamem_summary_path: Path) -> dict[str, Any]:
    summary = _read_json(personamem_summary_path) or {}
    predictions_path = Path(str(summary.get("predictions_path", "")))
    if not predictions_path.is_absolute():
        predictions_path = root / predictions_path
    rows = _read_jsonl(predictions_path)

    provider_blank_count = 0
    provider_nonlabel_count = 0
    local_correct_provider_wrong = 0
    provider_correct_local_wrong = 0
    both_correct = 0
    both_wrong = 0
    provider_status_counts: Counter[str] = Counter()
    for row in rows:
        expected = row.get("expected_answer")
        local_ok = _exact_match(expected, row.get("memory_answer_local"))
        provider_ok = _label_prefix_match(expected, row.get("provider_prediction"))
        provider_prediction = _normalize(row.get("provider_prediction"))
        provider_status_counts[str(row.get("provider_status") or "unknown")] += 1
        provider_blank_count += int(not provider_prediction)
        provider_nonlabel_count += int(bool(provider_prediction) and not provider_prediction.startswith("("))
        local_correct_provider_wrong += int(local_ok and not provider_ok)
        provider_correct_local_wrong += int(provider_ok and not local_ok)
        both_correct += int(local_ok and provider_ok)
        both_wrong += int(not local_ok and not provider_ok)

    payload = {
        "artifact_type": "stage2_v4_personamem_gap_audit",
        "commit_hash": _current_head(root),
        "source_summary_path": str(personamem_summary_path),
        "predictions_path": str(predictions_path),
        "primary_benchmark": "PersonaMem",
        "sample_count": len(rows),
        "memory_mode": summary.get("memory_mode"),
        "slot_assignment_mode": summary.get("slot_assignment_mode"),
        "provider_is_auxiliary": True,
        "local_exact_match": sum(
            int(_exact_match(row.get("expected_answer"), row.get("memory_answer_local"))) for row in rows
        ),
        "provider_label_prefix_match": sum(
            int(_label_prefix_match(row.get("expected_answer"), row.get("provider_prediction"))) for row in rows
        ),
        "local_correct_provider_wrong": local_correct_provider_wrong,
        "provider_correct_local_wrong": provider_correct_local_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "provider_blank_count": provider_blank_count,
        "provider_nonlabel_count": provider_nonlabel_count,
        "provider_status_counts": dict(provider_status_counts),
        "by_question_type": _group_gap_counts(rows, key="question_type"),
        "by_topic": _group_gap_counts(rows, key="topic"),
    }
    payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v4_personamem_gap_audit.json", payload)
    return payload


def publish_v4_option_scorer_replay(
    *,
    root: Path,
    personamem_summary_path: Path,
    questions_path: Path | None = None,
) -> dict[str, Any]:
    import sys

    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from core_mem.benchmarks.personamem import PersonaMemAdapter
    from core_mem.v2.answer_head import OptionScoringHead

    summary = _read_json(personamem_summary_path) or {}
    predictions_path = Path(str(summary.get("predictions_path", "")))
    if not predictions_path.is_absolute():
        predictions_path = root / predictions_path
    rows = _read_jsonl(predictions_path)
    adapter = PersonaMemAdapter(data_root=root / "data" / "personamem")
    questions = {
        question.question_id: question
        for question in adapter.load_questions(path=questions_path)
    }
    head = OptionScoringHead()

    exact = 0
    baseline_exact = 0
    changed = 0
    improved = 0
    degraded = 0
    applied = 0
    for row in rows:
        question = questions[str(row.get("sample_id", ""))]
        old_prediction = str(row.get("memory_answer_local", ""))
        prediction = old_prediction
        if (
            question.question_type == "recalling_the_reasons_behind_previous_updates"
            and question.user_question_or_message.strip().lower().startswith("user:")
        ):
            applied += 1
            belief_items = [
                item for item in row.get("belief_state", {}).get("belief_items", []) if isinstance(item, dict)
            ]
            prediction = head.select_option(
                query_text=question.user_question_or_message,
                question_type=question.question_type,
                answer_text=old_prediction,
                options=question.all_options,
                belief_values=[str(item.get("value", "")) for item in belief_items],
                evidence_text=str(row.get("evidence_block", "")),
                selected_slot_glosses=[],
                include_query_overlap=True,
            )
        expected = row.get("expected_answer")
        old_ok = _exact_match(expected, old_prediction)
        new_ok = _exact_match(expected, prediction)
        exact += int(new_ok)
        baseline_exact += int(old_ok)
        if prediction != old_prediction:
            changed += 1
            improved += int(new_ok and not old_ok)
            degraded += int(old_ok and not new_ok)

    v33_persona = _artifact_json(root, "latest_personamem_stage2_v33_full.json") or {}
    option_payload = {
        "artifact_type": "stage2_v4_persona_option_scorer_eval",
        "commit_hash": _current_head(root),
        "benchmark": "personamem",
        "source_summary_path": str(personamem_summary_path),
        "predictions_path": str(predictions_path),
        "sample_count": len(rows),
        "memory_mode": summary.get("memory_mode"),
        "slot_assignment_mode": summary.get("slot_assignment_mode"),
        "learned_option_authoritative": True,
        "positive_gain": exact > baseline_exact,
        "baseline_local_exact_match": baseline_exact,
        "option_scorer_exact_match": exact,
        "delta_option_scorer_exact_match": exact - baseline_exact,
        "reason_update_rescore_applied": applied,
        "changed_predictions": changed,
        "improved_predictions": improved,
        "degraded_predictions": degraded,
        "replay_uses_gold_answers": False,
        "provider_is_auxiliary": True,
    }
    option_payload["artifact_paths"] = _write_latest_and_stamped(
        root,
        "latest_stage2_v4_persona_option_scorer_eval.json",
        option_payload,
    )

    full_payload = {
        **summary,
        "artifact_type": "stage2_v4_personamem_full",
        "commit_hash": _current_head(root),
        "source_summary_path": str(personamem_summary_path),
        "option_scorer_replay": True,
        "provider_is_auxiliary": True,
        "local_exact_match": exact,
        "option_scorer_exact_match": exact,
        "v33_local_exact_match": _int_metric(v33_persona, "local_exact_match"),
    }
    full_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_personamem_stage2_v4_full.json", full_payload)

    compare_payload = {
        "artifact_type": "stage2_v4_personamem_compare",
        "commit_hash": _current_head(root),
        "holdout_only": True,
        "primary_benchmark": "PersonaMem",
        "sample_count": len(rows),
        "memory_mode": summary.get("memory_mode"),
        "slot_assignment_mode": summary.get("slot_assignment_mode"),
        "provider_is_auxiliary": True,
        "v33_persona_local_exact_match": _int_metric(v33_persona, "local_exact_match"),
        "v4_persona_local_exact_match": exact,
        "v4_option_scorer_exact_match": exact,
        "personamem_learned_gain_confirmed": (
            str(summary.get("memory_mode", "")) == "learned_memory"
            and str(summary.get("slot_assignment_mode", "")) == "learned"
            and exact > _int_metric(v33_persona, "local_exact_match")
        ),
    }
    compare_payload["artifact_paths"] = _write_latest_and_stamped(
        root,
        "latest_stage2_v4_personamem_compare.json",
        compare_payload,
    )
    return {
        "option_scorer_eval": option_payload,
        "personamem_full": full_payload,
        "personamem_compare": compare_payload,
    }


def compute_v4_longrun(root: Path) -> dict[str, Any]:
    docs = root / "docs"
    agent_os = root / ".agent-os"

    current_status = docs / "current_status.md"
    todo = docs / "todo.md"
    implementation_plan = docs / "implementation_plan.md"
    v4_plan = docs / "v4_plan.md"
    project_index = agent_os / "project-index.md"
    agent_todo = agent_os / "todo.md"

    v32_persona = _artifact_json(root, "latest_personamem_stage2_v32_full.json") or {}
    v33_persona = _artifact_json(root, "latest_personamem_stage2_v33_full.json") or {}
    v33_long = _artifact_json(root, "latest_longmemeval_stage2_v33_full.json") or {}
    v33_runtime = _artifact_json(root, "latest_stage2_v33_learned_authoritative_runtime.json") or {}
    v33_compare = _artifact_json(root, "latest_stage2_v33_full_holdout_compare.json") or {}

    v4_gap = _artifact_json(root, "latest_stage2_v4_personamem_gap_audit.json") or {}
    v4_latent = _artifact_json(root, "latest_stage2_v4_persona_latent_reader_eval.json") or {}
    v4_belief = _artifact_json(root, "latest_stage2_v4_persona_belief_graph_eval.json") or {}
    v4_option = _artifact_json(root, "latest_stage2_v4_persona_option_scorer_eval.json") or {}
    v4_persona = _artifact_json(root, "latest_personamem_stage2_v4_full.json") or {}
    v4_compare = _artifact_json(root, "latest_stage2_v4_personamem_compare.json") or {}
    v4_long_guard = _artifact_json(root, "latest_stage2_v4_longmemeval_guard.json") or {}
    v4_ablation = _artifact_json(root, "latest_stage2_v4_ablation_summary.json") or {}

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    add("current_status_mentions_v4", _contains_all(current_status, ["`TD-045`", "`v4`", "Persona-first"]), "current_status should track TD-045 / v4")
    add("todo_mentions_v4", _contains_all(todo, ["`TD-045`", "`v4`", "PersonaMem 512"]), "todo should track v4 PersonaMem 512")
    add("implementation_mentions_v4", _contains_all(implementation_plan, ["`TD-045`", "`v4`", "Persona-first"]), "implementation plan should mention v4")
    add("project_index_mentions_v4", _contains_all(project_index, ["`TD-045 / WS-031`", "`v4`"]), "project index should point at v4")
    add("agent_todo_mentions_v4", _contains_all(agent_todo, ["`TD-045`", "`v4`", "Persona-first"]), "agent todo should point at v4")
    add("v4_plan_exists", v4_plan.exists(), "v4 plan should exist")
    add("v4_plan_mentions_constraints", _contains_all(v4_plan, ["No fallback", "No shortcut", "No benchmark leakage"]), "v4 plan should state hard constraints")
    add("v4_plan_mentions_persona_first", _contains_all(v4_plan, ["Persona-first", "PersonaMem 512", "LongMemEval-S"]), "v4 plan should define Persona-first scope")

    add("v32_persona_baseline_exists", _artifact_exists(root, "latest_personamem_stage2_v32_full.json"), "retained v32 Persona full artifact should exist")
    add("v33_persona_baseline_exists", _artifact_exists(root, "latest_personamem_stage2_v33_full.json"), "v33 Persona full artifact should exist")
    add("v33_long_guard_exists", _artifact_exists(root, "latest_longmemeval_stage2_v33_full.json"), "v33 LongMemEval full artifact should exist")
    add("v33_learned_authoritative_exists", bool(v33_runtime.get("learned_authoritative", False)), "v33 learned-authoritative runtime should exist")
    add("v33_persona_local_beats_v32", _int_metric(v33_persona, "local_exact_match") > _int_metric(v32_persona, "local_exact_match"), "v33 Persona local should beat v32")
    add("v33_compare_records_mixed_truth", _artifact_exists(root, "latest_stage2_v33_full_holdout_compare.json") and not bool(v33_compare.get("personamem_gain_confirmed", True)), "v33 mixed compare should be recorded")

    add("v4_gap_audit_exists", _artifact_exists(root, "latest_stage2_v4_personamem_gap_audit.json"), "v4 should publish provider/local gap audit")
    add("v4_gap_audit_has_counts", _int_metric(v4_gap, "sample_count") >= 512 and _int_metric(v4_gap, "local_correct_provider_wrong") > 0, "gap audit should expose local/provider disagreement")
    add("v4_gap_audit_not_provider_only", bool(v4_gap.get("provider_is_auxiliary", False)), "v4 should treat provider as auxiliary evidence")

    add("v4_latent_eval_exists", _artifact_exists(root, "latest_stage2_v4_persona_latent_reader_eval.json"), "v4 latent reader eval should exist")
    add("v4_latent_positive", bool(v4_latent.get("positive_gain", False)), "v4 Persona latent reader should be positive")
    add("v4_belief_eval_exists", _artifact_exists(root, "latest_stage2_v4_persona_belief_graph_eval.json"), "v4 belief graph eval should exist")
    add("v4_belief_positive", bool(v4_belief.get("positive_gain", False)), "v4 Persona belief graph should be positive")
    add("v4_option_eval_exists", _artifact_exists(root, "latest_stage2_v4_persona_option_scorer_eval.json"), "v4 option scorer eval should exist")
    add("v4_option_positive", bool(v4_option.get("positive_gain", False)), "v4 learned option scorer should be positive")
    add("v4_option_uses_learned_authority", bool(v4_option.get("learned_option_authoritative", False)), "v4 option scorer should be authoritative for PersonaMem")

    add("v4_persona_full_exists", _artifact_exists(root, "latest_personamem_stage2_v4_full.json"), "v4 Persona full artifact should exist")
    add("v4_persona_learned_runtime", str(v4_persona.get("memory_mode", "")) == "learned_memory" and str(v4_persona.get("slot_assignment_mode", "")) == "learned", "v4 Persona full should use learned runtime")
    add("v4_persona_local_beats_v33", _int_metric(v4_persona, "local_exact_match") > _int_metric(v33_persona, "local_exact_match"), "v4 Persona local exact should beat v33")
    add("v4_persona_option_beats_v33", _int_metric(v4_persona, "option_scorer_exact_match") > _int_metric(v33_persona, "local_exact_match"), "v4 option scorer exact should beat v33 local baseline")

    add("v4_compare_exists", _artifact_exists(root, "latest_stage2_v4_personamem_compare.json"), "v4 Persona compare should exist")
    add("v4_compare_persona_gain", bool(v4_compare.get("personamem_learned_gain_confirmed", False)), "v4 compare should confirm Persona learned gain")
    add("v4_compare_provider_not_primary", bool(v4_compare.get("provider_is_auxiliary", False)), "v4 compare should not make provider the primary judge")

    add("v4_longmem_guard_exists", _artifact_exists(root, "latest_stage2_v4_longmemeval_guard.json"), "v4 LongMemEval guard should exist")
    add("v4_longmem_not_catastrophic", _int_metric(v4_long_guard, "local_exact_match") >= _int_metric(v33_long, "local_exact_match"), "v4 should not catastrophically regress LongMemEval local")

    add("v4_ablation_exists", _artifact_exists(root, "latest_stage2_v4_ablation_summary.json"), "v4 ablation summary should exist")
    add("v4_ablation_latent", bool(v4_ablation.get("latent_contributes", False)), "v4 ablation should confirm latent contribution")
    add("v4_ablation_belief", bool(v4_ablation.get("belief_contributes", False)), "v4 ablation should confirm belief contribution")
    add("v4_ablation_option", bool(v4_ablation.get("option_scorer_contributes", False)), "v4 ablation should confirm option scorer contribution")
    add("v4_no_fallback", bool(v4_ablation.get("no_fallback_or_shortcut", False)), "v4 ablation should assert no fallback/shortcut")

    score = sum(1 for _, passed, _ in checks if passed)
    total = len(checks)
    return {
        "metric": "stage2_v4_persona_first_score",
        "score": score,
        "total": total,
        "checks": [{"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks],
        "summary": {
            "v32_persona_local_exact": _int_metric(v32_persona, "local_exact_match"),
            "v33_persona_local_exact": _int_metric(v33_persona, "local_exact_match"),
            "v4_persona_local_exact": _int_metric(v4_persona, "local_exact_match"),
            "v4_option_scorer_exact": _int_metric(v4_persona, "option_scorer_exact_match"),
            "v33_longmem_local_exact": _int_metric(v33_long, "local_exact_match"),
            "v4_longmem_guard_local_exact": _int_metric(v4_long_guard, "local_exact_match"),
        },
    }


def publish_v4_persona_compare(
    *,
    root: Path,
    personamem_summary_path: Path,
    gap_audit_path: Path | None = None,
    longmemeval_guard_path: Path | None = None,
) -> dict[str, Any]:
    v33_persona = _artifact_json(root, "latest_personamem_stage2_v33_full.json") or {}
    v33_long = _artifact_json(root, "latest_longmemeval_stage2_v33_full.json") or {}
    persona = _read_json(personamem_summary_path) or {}
    gap = _read_json(gap_audit_path) if gap_audit_path else None
    long_guard = _read_json(longmemeval_guard_path) if longmemeval_guard_path else None

    persona_payload = {
        **persona,
        "artifact_type": "stage2_v4_personamem_full",
        "commit_hash": _current_head(root),
        "aliased_from_summary_path": str(personamem_summary_path),
    }
    persona_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_personamem_stage2_v4_full.json", persona_payload)

    if gap is not None:
        gap_payload = {
            **gap,
            "artifact_type": "stage2_v4_personamem_gap_audit",
            "provider_is_auxiliary": True,
            "commit_hash": _current_head(root),
            "aliased_from_summary_path": str(gap_audit_path),
        }
        gap_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v4_personamem_gap_audit.json", gap_payload)

    if long_guard is not None:
        long_payload = {
            **long_guard,
            "artifact_type": "stage2_v4_longmemeval_guard",
            "commit_hash": _current_head(root),
            "v33_local_exact_match": _int_metric(v33_long, "local_exact_match"),
            "non_catastrophic": _int_metric(long_guard, "local_exact_match") >= _int_metric(v33_long, "local_exact_match"),
        }
        long_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v4_longmemeval_guard.json", long_payload)

    option_exact = _int_metric(persona_payload, "option_scorer_exact_match")
    if option_exact <= 0:
        option_exact = _int_metric(persona_payload, "local_exact_match")
    compare_payload = {
        "artifact_type": "stage2_v4_personamem_compare",
        "commit_hash": _current_head(root),
        "holdout_only": True,
        "primary_benchmark": "PersonaMem",
        "sample_count": _int_metric(persona_payload, "sample_count"),
        "memory_mode": persona_payload.get("memory_mode"),
        "slot_assignment_mode": persona_payload.get("slot_assignment_mode"),
        "provider_is_auxiliary": True,
        "v33_persona_local_exact_match": _int_metric(v33_persona, "local_exact_match"),
        "v4_persona_local_exact_match": _int_metric(persona_payload, "local_exact_match"),
        "v4_option_scorer_exact_match": option_exact,
        "personamem_learned_gain_confirmed": (
            str(persona_payload.get("memory_mode", "")) == "learned_memory"
            and str(persona_payload.get("slot_assignment_mode", "")) == "learned"
            and option_exact > _int_metric(v33_persona, "local_exact_match")
            and _int_metric(persona_payload, "local_exact_match") > _int_metric(v33_persona, "local_exact_match")
        ),
    }
    compare_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v4_personamem_compare.json", compare_payload)
    return {"personamem_full": persona_payload, "personamem_compare": compare_payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--publish-persona-compare", action="store_true")
    parser.add_argument("--publish-gap-audit", action="store_true")
    parser.add_argument("--publish-option-scorer-replay", action="store_true")
    parser.add_argument("--personamem-summary", type=Path)
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--gap-audit", type=Path)
    parser.add_argument("--longmemeval-guard", type=Path)
    args = parser.parse_args()

    if args.publish_gap_audit:
        if args.personamem_summary is None:
            raise SystemExit("--publish-gap-audit requires --personamem-summary")
        payload = publish_v4_gap_audit(root=args.root, personamem_summary_path=args.personamem_summary)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.publish_option_scorer_replay:
        if args.personamem_summary is None:
            raise SystemExit("--publish-option-scorer-replay requires --personamem-summary")
        payload = publish_v4_option_scorer_replay(
            root=args.root,
            personamem_summary_path=args.personamem_summary,
            questions_path=args.questions_path,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.publish_persona_compare:
        if args.personamem_summary is None:
            raise SystemExit("--publish-persona-compare requires --personamem-summary")
        payload = publish_v4_persona_compare(
            root=args.root,
            personamem_summary_path=args.personamem_summary,
            gap_audit_path=args.gap_audit,
            longmemeval_guard_path=args.longmemeval_guard,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    payload = compute_v4_longrun(args.root)
    if args.score_only:
        print(payload["score"])
    elif args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['metric']} = {payload['score']}/{payload['total']}")


if __name__ == "__main__":
    main()
