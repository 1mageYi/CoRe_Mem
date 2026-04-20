"""Mechanical verifier for the v3.3 / v33 learned-authoritative latent run."""

from __future__ import annotations

import argparse
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str:
    head = (root / ".git").exists()
    if not head:
        return "unknown"
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


def _float_metric(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _int_metric(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    try:
        return int(payload.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _rate(payload: dict[str, Any] | None, *, key: str) -> float:
    if not payload:
        return 0.0
    sample_count = _int_metric(payload, "sample_count")
    if sample_count <= 0:
        return 0.0
    return _int_metric(payload, key) / sample_count


def _publish_canary_alias(*, root: Path, summary_path: Path, latest_name: str, artifact_type: str) -> dict[str, Any]:
    payload = _read_json(summary_path)
    if payload is None:
        raise FileNotFoundError(summary_path)
    alias_payload = {
        **payload,
        "artifact_type": artifact_type,
        "commit_hash": _current_head(root),
        "aliased_from_summary_path": str(summary_path),
        "aliased_from_commit_hash": payload.get("commit_hash"),
        "aliased_at": datetime.now(timezone.utc).isoformat(),
        "provider_exact_rate": _rate(payload, key="provider_exact_match"),
        "local_exact_rate": _rate(payload, key="local_exact_match"),
    }
    alias_payload["artifact_paths"] = _write_latest_and_stamped(root, latest_name, alias_payload)
    return alias_payload


def publish_v33_full_holdout_artifacts(
    *,
    root: Path,
    longmemeval_summary_path: Path,
    personamem_summary_path: Path,
) -> dict[str, Any]:
    current_head = _current_head(root)
    retained_v32_long = _artifact_json(root, "latest_longmemeval_stage2_v32_full.json") or {}
    retained_v32_persona = _artifact_json(root, "latest_personamem_stage2_v32_full.json") or {}

    longmemeval_payload = _publish_canary_alias(
        root=root,
        summary_path=longmemeval_summary_path,
        latest_name="latest_longmemeval_stage2_v33_full.json",
        artifact_type="stage2_v33_longmemeval_full",
    )
    personamem_payload = _publish_canary_alias(
        root=root,
        summary_path=personamem_summary_path,
        latest_name="latest_personamem_stage2_v33_full.json",
        artifact_type="stage2_v33_personamem_full",
    )

    learned_authoritative = (
        str(longmemeval_payload.get("memory_mode", "")) == "learned_memory"
        and str(longmemeval_payload.get("slot_assignment_mode", "")) == "learned"
        and str(personamem_payload.get("memory_mode", "")) == "learned_memory"
        and str(personamem_payload.get("slot_assignment_mode", "")) == "learned"
    )

    runtime_payload = {
        "artifact_type": "stage2_v33_learned_authoritative_runtime",
        "commit_hash": current_head,
        "memory_mode": longmemeval_payload.get("memory_mode") or personamem_payload.get("memory_mode"),
        "slot_assignment_mode": longmemeval_payload.get("slot_assignment_mode") or personamem_payload.get("slot_assignment_mode"),
        "learned_authoritative": learned_authoritative,
        "longmemeval_status": longmemeval_payload.get("status"),
        "personamem_status": personamem_payload.get("status"),
    }
    runtime_payload["artifact_paths"] = _write_latest_and_stamped(
        root, "latest_stage2_v33_learned_authoritative_runtime.json", runtime_payload
    )

    holdout_payload = {
        "artifact_type": "stage2_v33_full_holdout_compare",
        "commit_hash": current_head,
        "holdout_only": True,
        "benchmark_runs_executed": 2,
        "memory_mode": runtime_payload["memory_mode"],
        "slot_assignment_mode": runtime_payload["slot_assignment_mode"],
        "learned_authoritative": learned_authoritative,
        "longmemeval_sample_count": _int_metric(longmemeval_payload, "sample_count"),
        "personamem_sample_count": _int_metric(personamem_payload, "sample_count"),
        "v32_longmemeval_provider_exact_match": _int_metric(retained_v32_long, "provider_exact_match"),
        "v32_longmemeval_local_exact_match": _int_metric(retained_v32_long, "local_exact_match"),
        "v32_personamem_provider_exact_rate": _float_metric(retained_v32_persona, "provider_exact_rate"),
        "v32_personamem_local_exact_rate": _float_metric(retained_v32_persona, "local_exact_rate"),
        "longmemeval_gain_confirmed": (
            _int_metric(longmemeval_payload, "provider_exact_match") > _int_metric(retained_v32_long, "provider_exact_match")
            and _int_metric(longmemeval_payload, "local_exact_match") >= _int_metric(retained_v32_long, "local_exact_match")
        ),
        "personamem_gain_confirmed": (
            _float_metric(personamem_payload, "provider_exact_rate") > _float_metric(retained_v32_persona, "provider_exact_rate")
            and _float_metric(personamem_payload, "local_exact_rate") >= _float_metric(retained_v32_persona, "local_exact_rate")
        ),
        "note": "V33 succeeds only when full holdout gains are achieved under learned authoritative runtime.",
    }
    holdout_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v33_full_holdout_compare.json", holdout_payload)
    return {
        "longmemeval_full": longmemeval_payload,
        "personamem_full": personamem_payload,
        "learned_authoritative_runtime": runtime_payload,
        "full_holdout_compare": holdout_payload,
    }


def publish_v33_ablation_summary(*, root: Path) -> dict[str, Any]:
    current_head = _current_head(root)
    write_eval = _artifact_json(root, "latest_stage2_v33_learned_write_eval.json") or {}
    latent_train = _artifact_json(root, "latest_stage2_v33_latent_reader_train.json") or {}
    temporal_slot = _artifact_json(root, "latest_stage2_v33_temporal_slot_eval.json") or {}
    latent_objective = _artifact_json(root, "latest_stage2_v33_latent_objective_eval.json") or {}
    belief_graph = _artifact_json(root, "latest_stage2_v33_belief_graph_eval.json") or {}
    answer_option = _artifact_json(root, "latest_stage2_v33_answer_option_eval.json") or {}
    runtime_payload = _artifact_json(root, "latest_stage2_v33_learned_authoritative_runtime.json") or {}

    payload = {
        "artifact_type": "stage2_v33_ablation_summary",
        "commit_hash": current_head,
        "latent_is_primary_driver": bool(latent_train.get("trainable_latent", False) or latent_train.get("positive_gain", False))
        and bool(latent_objective.get("positive_gain", False))
        and bool(temporal_slot.get("positive_gain", False)),
        "belief_contributes": bool(belief_graph.get("positive_gain", False)),
        "answer_head_contributes": bool(answer_option.get("positive_gain", False)),
        "write_contributes": bool(write_eval.get("positive_gain", False)),
        "learned_authoritative": bool(runtime_payload.get("learned_authoritative", False)),
        "note": "V33 ablation truth requires a learned authoritative runtime, latent as primary driver, and independent write/belief/answer contributions.",
    }
    payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v33_ablation_summary.json", payload)
    return payload


def compute_v33_longrun(root: Path) -> dict[str, Any]:
    docs = root / "docs"
    agent_os = root / ".agent-os"

    current_status = docs / "current_status.md"
    implementation_plan = docs / "implementation_plan.md"
    todo = docs / "todo.md"
    v33_plan = docs / "v33_plan.md"
    project_index = agent_os / "project-index.md"
    agent_todo = agent_os / "todo.md"

    retained_v32_long = _artifact_json(root, "latest_longmemeval_stage2_v32_full.json") or {}
    retained_v32_persona = _artifact_json(root, "latest_personamem_stage2_v32_full.json") or {}
    retained_v32_compare = _artifact_json(root, "latest_stage2_v32_full_holdout_compare.json") or {}
    retained_v32_ablation = _artifact_json(root, "latest_stage2_v32_ablation_summary.json") or {}
    retained_v32_latent = _artifact_json(root, "latest_stage2_v32_latent_holdout_compare.json") or {}
    retained_v32_belief = _artifact_json(root, "latest_stage2_v32_belief_holdout_compare.json") or {}
    retained_v32_answer = _artifact_json(root, "latest_stage2_v32_option_scoring_compare.json") or {}

    v33_modular = _artifact_json(root, "latest_stage2_v33_modular_authoritative_train.json") or {}
    v33_write = _artifact_json(root, "latest_stage2_v33_learned_write_eval.json") or {}
    v33_latent_train = _artifact_json(root, "latest_stage2_v33_latent_reader_train.json") or {}
    v33_temporal_slot = _artifact_json(root, "latest_stage2_v33_temporal_slot_eval.json") or {}
    v33_latent_objective = _artifact_json(root, "latest_stage2_v33_latent_objective_eval.json") or {}
    v33_belief_graph = _artifact_json(root, "latest_stage2_v33_belief_graph_eval.json") or {}
    v33_answer = _artifact_json(root, "latest_stage2_v33_answer_option_eval.json") or {}
    v33_runtime = _artifact_json(root, "latest_stage2_v33_learned_authoritative_runtime.json") or {}
    v33_ablation = _artifact_json(root, "latest_stage2_v33_ablation_summary.json") or {}
    v33_holdout = _artifact_json(root, "latest_stage2_v33_full_holdout_compare.json") or {}
    v33_long = _artifact_json(root, "latest_longmemeval_stage2_v33_full.json") or {}
    v33_persona = _artifact_json(root, "latest_personamem_stage2_v33_full.json") or {}

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    add("current_status_mentions_v33", _contains_all(current_status, ["`TD-044`", "`v33`", "learned-authoritative", "500", "512"]), "current_status should track TD-044 / v33")
    add("implementation_mentions_v33", _contains_all(implementation_plan, ["`TD-044`", "`v33`", "learned", "authoritative", "belief", "answer", "500", "512"]), "implementation_plan should mention v33 learned-authoritative path")
    add("todo_mentions_v33", _contains_all(todo, ["`TD-044`", "`v33`", "learned-authoritative", "LongMemEval-S 500", "PersonaMem 512"]), "todo should mention v33")
    add("agent_todo_mentions_v33", _contains_all(agent_todo, ["`TD-044`", "`v33`", "learned-authoritative"]), "agent todo should mention v33")
    add("project_index_mentions_v33", _contains_all(project_index, ["`TD-044 / WS-030`", "`v33`"]), "project index should point at v33")
    add("v33_plan_mentions_thesis", _contains_all(v33_plan, ["learned-authoritative", "Temporal-Semantic", "Belief Graph", "LongMemEval-S 500", "PersonaMem 512"]), "v33 plan should state the aggressive thesis")
    add("v33_plan_mentions_constraints", _contains_all(v33_plan, ["fallback", "shortcut", "benchmark leakage"]), "v33 plan should restate hard constraints")

    add("retained_v32_long_exists", _artifact_exists(root, "latest_longmemeval_stage2_v32_full.json"), "retained v32 LongMemEval full artifact should exist")
    add("retained_v32_persona_exists", _artifact_exists(root, "latest_personamem_stage2_v32_full.json"), "retained v32 PersonaMem full artifact should exist")
    add("retained_v32_compare_exists", _artifact_exists(root, "latest_stage2_v32_full_holdout_compare.json"), "retained v32 full-holdout compare should exist")
    add("retained_v32_ablation_exists", _artifact_exists(root, "latest_stage2_v32_ablation_summary.json"), "retained v32 ablation should exist")
    add("retained_v32_long_gain", bool(retained_v32_compare.get("longmemeval_gain_confirmed", False)), "retained v32 LongMemEval gain should stay true")
    add("retained_v32_persona_gain", bool(retained_v32_compare.get("personamem_gain_confirmed", False)), "retained v32 PersonaMem gain should stay true")
    add("retained_v32_latent_positive", bool(retained_v32_latent.get("positive_gain", False)), "retained v32 latent compare should stay positive")
    add("retained_v32_belief_positive", bool(retained_v32_belief.get("positive_gain", False)), "retained v32 belief compare should stay positive")
    add("retained_v32_answer_positive", bool(retained_v32_answer.get("positive_gain", False)), "retained v32 answer compare should stay positive")
    add("retained_v32_ablation_truth", bool(retained_v32_ablation.get("latent_is_primary_driver", False)), "retained v32 ablation should confirm latent as primary driver")

    add("v33_modular_train_exists", _artifact_exists(root, "latest_stage2_v33_modular_authoritative_train.json"), "v33 modular authoritative train artifact should exist")
    add("v33_write_eval_exists", _artifact_exists(root, "latest_stage2_v33_learned_write_eval.json"), "v33 learned write eval should exist")
    add("v33_latent_train_exists", _artifact_exists(root, "latest_stage2_v33_latent_reader_train.json"), "v33 latent reader train artifact should exist")
    add("v33_temporal_slot_exists", _artifact_exists(root, "latest_stage2_v33_temporal_slot_eval.json"), "v33 temporal slot eval should exist")
    add("v33_latent_objective_exists", _artifact_exists(root, "latest_stage2_v33_latent_objective_eval.json"), "v33 latent objective eval should exist")
    add("v33_belief_graph_exists", _artifact_exists(root, "latest_stage2_v33_belief_graph_eval.json"), "v33 belief graph eval should exist")
    add("v33_answer_exists", _artifact_exists(root, "latest_stage2_v33_answer_option_eval.json"), "v33 answer option eval should exist")
    add("v33_runtime_exists", _artifact_exists(root, "latest_stage2_v33_learned_authoritative_runtime.json"), "v33 learned runtime artifact should exist")
    add("v33_ablation_exists", _artifact_exists(root, "latest_stage2_v33_ablation_summary.json"), "v33 ablation summary should exist")
    add("v33_holdout_exists", _artifact_exists(root, "latest_stage2_v33_full_holdout_compare.json"), "v33 full holdout compare should exist")
    add("v33_long_exists", _artifact_exists(root, "latest_longmemeval_stage2_v33_full.json"), "v33 LongMemEval full artifact should exist")
    add("v33_persona_exists", _artifact_exists(root, "latest_personamem_stage2_v33_full.json"), "v33 PersonaMem full artifact should exist")

    add("v33_modular_positive", bool(v33_modular.get("positive_gain", False)), "v33 modular authoritative train should be positive")
    add("v33_write_positive", bool(v33_write.get("positive_gain", False)), "v33 learned write eval should be positive")
    add("v33_latent_trainable", bool(v33_latent_train.get("trainable_latent", False)) or bool(v33_latent_train.get("positive_gain", False)), "v33 latent reader should be trainable")
    add("v33_temporal_slot_positive", bool(v33_temporal_slot.get("positive_gain", False)), "v33 temporal slot eval should be positive")
    add("v33_latent_objective_positive", bool(v33_latent_objective.get("positive_gain", False)), "v33 latent objective should be positive")
    add("v33_belief_graph_positive", bool(v33_belief_graph.get("positive_gain", False)), "v33 belief graph eval should be positive")
    add("v33_answer_positive", bool(v33_answer.get("positive_gain", False)), "v33 answer option eval should be positive")
    add("v33_runtime_is_learned_authoritative", bool(v33_runtime.get("learned_authoritative", False)), "v33 runtime must be learned-authoritative")
    add("v33_runtime_memory_mode", str(v33_runtime.get("memory_mode", "")) == "learned_memory", "v33 runtime must use learned_memory")
    add("v33_runtime_slot_mode", str(v33_runtime.get("slot_assignment_mode", "")) == "learned", "v33 runtime must use learned slot assignment")
    add("v33_holdout_long_gain", bool(v33_holdout.get("longmemeval_gain_confirmed", False)), "v33 LongMemEval holdout gain must be confirmed")
    add("v33_holdout_persona_gain", bool(v33_holdout.get("personamem_gain_confirmed", False)), "v33 PersonaMem holdout gain must be confirmed")
    add("v33_long_beats_v32", _int_metric(v33_long, "provider_exact_match") > _int_metric(retained_v32_long, "provider_exact_match"), "v33 LongMemEval provider exact should beat v32")
    add("v33_persona_beats_v32", _float_metric(v33_persona, "provider_exact_rate") > _float_metric(retained_v32_persona, "provider_exact_rate"), "v33 PersonaMem provider rate should beat v32")
    add("v33_ablation_latent_primary", bool(v33_ablation.get("latent_is_primary_driver", False)), "v33 ablation should keep latent as primary driver")
    add("v33_ablation_belief_contributes", bool(v33_ablation.get("belief_contributes", False)), "v33 ablation should confirm belief contribution")
    add("v33_ablation_answer_contributes", bool(v33_ablation.get("answer_head_contributes", False)), "v33 ablation should confirm answer contribution")
    add("v33_ablation_write_contributes", bool(v33_ablation.get("write_contributes", False)), "v33 ablation should confirm write contribution")

    score = sum(1 for _, passed, _ in checks if passed)
    total = len(checks)
    return {
        "metric": "stage2_v33_longrun_score",
        "score": score,
        "total": total,
        "checks": [{"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks],
        "summary": {
            "retained_v32_long_provider_exact": _int_metric(retained_v32_long, "provider_exact_match"),
            "retained_v32_persona_provider_rate": _float_metric(retained_v32_persona, "provider_exact_rate"),
            "v33_long_provider_exact": _int_metric(v33_long, "provider_exact_match"),
            "v33_persona_provider_rate": _float_metric(v33_persona, "provider_exact_rate"),
            "learned_authoritative": bool(v33_runtime.get("learned_authoritative", False)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--publish-full-holdout-artifacts", action="store_true")
    parser.add_argument("--longmemeval-summary", type=Path)
    parser.add_argument("--personamem-summary", type=Path)
    parser.add_argument("--publish-ablation-summary", action="store_true")
    args = parser.parse_args()

    if args.publish_full_holdout_artifacts:
        if args.longmemeval_summary is None or args.personamem_summary is None:
            raise SystemExit("--publish-full-holdout-artifacts requires --longmemeval-summary and --personamem-summary")
        payload = publish_v33_full_holdout_artifacts(
            root=args.root,
            longmemeval_summary_path=args.longmemeval_summary,
            personamem_summary_path=args.personamem_summary,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.publish_ablation_summary:
        payload = publish_v33_ablation_summary(root=args.root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    payload = compute_v33_longrun(args.root)
    if args.score_only:
        print(payload["score"])
    elif args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['metric']} = {payload['score']}/{payload['total']}")


if __name__ == "__main__":
    main()
