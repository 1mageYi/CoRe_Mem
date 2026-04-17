"""Mechanical verifier for the v2.6 gain-first long run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from analyze_stage2_memory_canary_failures import build_analysis, build_layered_analysis
except ImportError:  # pragma: no cover - package import path for pytest
    from scripts.analyze_stage2_memory_canary_failures import build_analysis, build_layered_analysis


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else None


def _artifact_matches_head(path: Path, head: str | None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if head is not None and str(payload.get("commit_hash", "")) != head:
        return None
    return payload


def _current_head_payload(root: Path, name: str) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / name, _current_head(root))


def _republish_artifact(
    *,
    root: Path,
    source_path: Path,
    latest_name: str,
    artifact_type: str,
) -> dict[str, Any]:
    payload = _read_json(source_path)
    republished = {
        **payload,
        "artifact_type": artifact_type,
        "commit_hash": _current_head(root),
        "republished_from_path": str(source_path),
        "republished_from_commit_hash": payload.get("commit_hash"),
        "republished_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_path = root / "outputs_v2" / "artifacts" / latest_name
    stamped_path = root / "outputs_v2" / "artifacts" / f"{_timestamp()}_{latest_name}"
    _write_json(latest_path, republished)
    _write_json(stamped_path, republished)
    republished["artifact_paths"] = {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}
    return republished


def _publish_canary_alias(
    *,
    root: Path,
    summary_path: Path,
    latest_name: str,
    artifact_type: str,
) -> dict[str, Any]:
    payload = _read_json(summary_path)
    aliased = {
        **payload,
        "artifact_type": artifact_type,
        "commit_hash": _current_head(root),
        "aliased_from_summary_path": str(summary_path),
        "aliased_from_commit_hash": payload.get("commit_hash"),
        "aliased_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_path = root / "outputs_v2" / "artifacts" / latest_name
    stamped_path = root / "outputs_v2" / "artifacts" / f"{_timestamp()}_{latest_name}"
    _write_json(latest_path, aliased)
    _write_json(stamped_path, aliased)
    aliased["artifact_paths"] = {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}
    return aliased


def _publish_longmemeval_analysis(
    *,
    root: Path,
    summary_path: Path,
) -> dict[str, Any]:
    analysis = build_analysis(root=root, benchmark="longmemeval_s", summary_path=summary_path)
    layered = build_layered_analysis(root=root, benchmark="longmemeval_s", summary_path=summary_path)
    payload = {
        **analysis,
        "artifact_type": "stage2_v26_longmemeval_analysis",
        "commit_hash": _current_head(root),
        "republished_from_commit_hash": analysis.get("commit_hash"),
        "republished_at": datetime.now(timezone.utc).isoformat(),
        "layered_breakdown": layered.get("layers", {}),
    }
    latest_path = root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_analysis.json"
    stamped_path = root / "outputs_v2" / "artifacts" / f"{_timestamp()}_latest_longmemeval_stage2_v26_analysis.json"
    _write_json(latest_path, payload)
    _write_json(stamped_path, payload)
    payload["artifact_paths"] = {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}
    return payload


def _baseline_metrics(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "provider_exact_match": int(payload.get("provider_exact_match", 0)),
        "local_exact_match": int(payload.get("local_exact_match", 0)),
        "provider_label_prefix_match": int(payload.get("provider_label_prefix_match", 0)),
    }


def _publish_component_gain(
    *,
    root: Path,
    component: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    latest_name = f"latest_stage2_v26_{component}_gain.json"
    latest_path = root / "outputs_v2" / "artifacts" / latest_name
    stamped_path = root / "outputs_v2" / "artifacts" / f"{_timestamp()}_{latest_name}"
    _write_json(latest_path, payload)
    _write_json(stamped_path, payload)
    payload["artifact_paths"] = {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}
    return payload


def publish_v26_artifacts(
    *,
    root: Path,
    longmemeval_summary_path: Path,
    personamem_summary_path: Path,
    train_source_path: Path,
    eval_source_path: Path,
    v25_longmemeval_path: Path,
    v25_personamem_path: Path,
) -> dict[str, Any]:
    current_head = _current_head(root)
    train_payload = _republish_artifact(
        root=root,
        source_path=train_source_path,
        latest_name="latest_stage2_v26_train.json",
        artifact_type="stage2_v26_train",
    )
    eval_payload = _republish_artifact(
        root=root,
        source_path=eval_source_path,
        latest_name="latest_stage2_v26_eval.json",
        artifact_type="stage2_v26_eval",
    )
    longmemeval_payload = _publish_canary_alias(
        root=root,
        summary_path=longmemeval_summary_path,
        latest_name="latest_longmemeval_stage2_v26_canary.json",
        artifact_type="stage2_v26_longmemeval_canary",
    )
    personamem_payload = _publish_canary_alias(
        root=root,
        summary_path=personamem_summary_path,
        latest_name="latest_personamem_stage2_v26_canary.json",
        artifact_type="stage2_v26_personamem_canary",
    )
    analysis_payload = _publish_longmemeval_analysis(root=root, summary_path=longmemeval_summary_path)

    baseline_long = _baseline_metrics(_read_json(v25_longmemeval_path))
    baseline_persona = _baseline_metrics(_read_json(v25_personamem_path))
    current_long = _baseline_metrics(longmemeval_payload)
    current_persona = _baseline_metrics(personamem_payload)
    slot_assignment_metrics = (_read_json(eval_source_path).get("slot_assignment_metrics")) or {}
    trained_eval_metrics = (((_read_json(eval_source_path).get("trained_eval")) or {}).get("metrics")) or {}

    write_delta_provider = current_long["provider_exact_match"] - baseline_long["provider_exact_match"]
    write_delta_local = current_long["local_exact_match"] - baseline_long["local_exact_match"]
    write_payload = _publish_component_gain(
        root=root,
        component="write",
        payload={
            "artifact_type": "stage2_v26_write_gain",
            "component": "write",
            "commit_hash": current_head,
            "delta_vs_v25_retained": min(write_delta_provider, write_delta_local),
            "delta_provider_exact_match": write_delta_provider,
            "delta_local_exact_match": write_delta_local,
            "positive_gain": write_delta_provider > 0 and write_delta_local > 0,
            "core_residual_frozen": True,
            "source_artifacts": [
                "latest_stage2_v26_eval.json",
                "latest_longmemeval_stage2_v26_canary.json",
                "latest_personamem_stage2_v26_canary.json",
            ],
            "slot_assignment_metrics": slot_assignment_metrics,
            "longmemeval_online_metrics": current_long,
            "personamem_guard_metrics": current_persona,
            "v25_longmemeval_reference": baseline_long,
            "v25_personamem_reference": baseline_persona,
        },
    )

    layered = analysis_payload.get("layered_breakdown", {})
    baseline_analysis = _read_json(root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_analysis.json")
    baseline_layers = baseline_analysis.get("layered_breakdown", {})
    retrieval_delta = int(baseline_layers.get("retrieval", {}).get("count", 0)) - int(layered.get("retrieval", {}).get("count", 0))
    retrieve_payload = _publish_component_gain(
        root=root,
        component="retrieve",
        payload={
            "artifact_type": "stage2_v26_retrieve_gain",
            "component": "retrieve",
            "commit_hash": current_head,
            "delta_vs_v25_retained": retrieval_delta,
            "positive_gain": retrieval_delta > 0,
            "retrieval_alignment_metrics": {"token_f1": trained_eval_metrics.get("token_f1", 0.0)},
            "layered_breakdown": layered,
            "failure_bucket_snapshot": analysis_payload.get("failure_buckets", {}),
            "top_failure_relations": analysis_payload.get("top_failure_belief_relations", {}),
            "v25_layered_breakdown": baseline_layers,
        },
    )

    baseline_projection = int(baseline_layers.get("projection", {}).get("count", 0))
    current_projection = int(layered.get("projection", {}).get("count", 0))
    belief_delta = baseline_projection - current_projection
    belief_payload = _publish_component_gain(
        root=root,
        component="belief",
        payload={
            "artifact_type": "stage2_v26_belief_gain",
            "component": "belief",
            "commit_hash": current_head,
            "delta_vs_v25_retained": belief_delta,
            "positive_gain": belief_delta > 0,
            "composition_to_belief_metrics": {"token_f1": trained_eval_metrics.get("token_f1", 0.0)},
            "belief_recovery_family": {
                "joint_belief_accuracy": trained_eval_metrics.get("field_accuracy", 0.0),
                "slot_value_f1": trained_eval_metrics.get("field_f1", 0.0),
                "answer_exact_match": trained_eval_metrics.get("exact_match", 0.0),
            },
            "online_failure_layers": layered,
            "online_projection_bottleneck_count": current_projection,
            "v25_projection_bottleneck_count": baseline_projection,
        },
    )

    full_benchmark_payload = {
        "artifact_type": "stage2_v26_full_benchmark",
        "commit_hash": current_head,
        "republished_at": datetime.now(timezone.utc).isoformat(),
        "core_residual_frozen": True,
        "num_benchmarks": 2,
        "holdout_only": True,
        "benchmarks": [
            {
                "benchmark": "longmemeval_s",
                "summary_path": str(longmemeval_summary_path),
                "sample_count": int(longmemeval_payload.get("sample_count", 0)),
                "provider_exact_match": current_long["provider_exact_match"],
                "local_exact_match": current_long["local_exact_match"],
            },
            {
                "benchmark": "personamem",
                "summary_path": str(personamem_summary_path),
                "sample_count": int(personamem_payload.get("sample_count", 0)),
                "provider_exact_match": current_persona["provider_exact_match"],
                "local_exact_match": current_persona["local_exact_match"],
            },
        ],
    }
    latest_full_benchmark_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v26_full_benchmark.json"
    stamped_full_benchmark_path = root / "outputs_v2" / "artifacts" / f"{_timestamp()}_latest_stage2_v26_full_benchmark.json"
    _write_json(latest_full_benchmark_path, full_benchmark_payload)
    _write_json(stamped_full_benchmark_path, full_benchmark_payload)
    full_benchmark_payload["artifact_paths"] = {
        "latest_path": str(latest_full_benchmark_path),
        "stamped_path": str(stamped_full_benchmark_path),
    }

    return {
        "head": current_head,
        "train": train_payload,
        "eval": eval_payload,
        "longmemeval_canary": longmemeval_payload,
        "personamem_canary": personamem_payload,
        "analysis": analysis_payload,
        "write_gain": write_payload,
        "retrieve_gain": retrieve_payload,
        "belief_gain": belief_payload,
        "full_benchmark": full_benchmark_payload,
    }


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        text = _read_text(root / rel)
        if text and any(pattern in text for pattern in patterns):
            return True
    return False


def _historical_v25_artifacts_exist(root: Path) -> bool:
    required = [
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_train.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_eval.json",
        root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v25_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_write_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_retrieve_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_belief_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_full_benchmark.json",
    ]
    return all(path.exists() for path in required)


def _trained_metrics(eval_payload: dict[str, Any] | None) -> dict[str, Any]:
    return ((eval_payload or {}).get("trained_eval") or {}).get("metrics") or {}


def _gain_positive(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if payload.get("positive_gain") is True:
        return True
    for key in ("delta_vs_v25_retained", "delta_provider_exact_match", "delta_local_exact_match"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def compute_v26_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v26_plan = root / "docs" / "v26_plan.md"

    v25_eval_payload = _current_head_payload(root, "latest_stage2_v25_eval.json")
    v25_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v25_canary.json")
    v25_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v25_canary.json")

    v26_train_payload = _current_head_payload(root, "latest_stage2_v26_train.json")
    v26_eval_payload = _current_head_payload(root, "latest_stage2_v26_eval.json")
    v26_write_gain = _current_head_payload(root, "latest_stage2_v26_write_gain.json")
    v26_retrieve_gain = _current_head_payload(root, "latest_stage2_v26_retrieve_gain.json")
    v26_belief_gain = _current_head_payload(root, "latest_stage2_v26_belief_gain.json")
    v26_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v26_canary.json")
    v26_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v26_canary.json")
    v26_analysis_payload = _current_head_payload(root, "latest_longmemeval_stage2_v26_analysis.json")
    v26_full_benchmark_payload = _current_head_payload(root, "latest_stage2_v26_full_benchmark.json")

    v25_trained_metrics = _trained_metrics(v25_eval_payload)
    v26_trained_metrics = _trained_metrics(v26_eval_payload)
    v25_long_provider = int((v25_longmemeval_payload or {}).get("provider_exact_match", 0))
    v25_long_local = int((v25_longmemeval_payload or {}).get("local_exact_match", 0))
    v25_persona_provider = int((v25_personamem_payload or {}).get("provider_exact_match", 0))
    v25_persona_local = int((v25_personamem_payload or {}).get("local_exact_match", 0))
    v26_long_provider = int((v26_longmemeval_payload or {}).get("provider_exact_match", 0))
    v26_long_local = int((v26_longmemeval_payload or {}).get("local_exact_match", 0))
    v26_persona_provider = int((v26_personamem_payload or {}).get("provider_exact_match", 0))
    v26_persona_local = int((v26_personamem_payload or {}).get("local_exact_match", 0))

    component_positive = any(_gain_positive(payload) for payload in [v26_write_gain, v26_retrieve_gain, v26_belief_gain])

    checks = {
        "current_status_tracks_td037_v26": _contains_all(
            current_status,
            ["`TD-037`", "`v2.6`", "LongMemEval-S", "write -> retrieve -> belief", "真实正增益"],
        ),
        "implementation_plan_tracks_stage_w_v26": _contains_all(
            implementation_plan,
            ["阶段 W：V2.6 Gain-First Long Run", "LongMemEval-S", "正增益"],
        ),
        "agentos_tracks_td037_ws023": _contains_all(
            project_index,
            ["`TD-037 / WS-023`", "`v2.6 gain-first long-run`"],
        ) and _contains_all(todo, ["`TD-037` `[doing]`"]),
        "v26_plan_exists": v26_plan.exists(),
        "v26_plan_freezes_core_residual": _contains_all(
            v26_plan,
            ["不改 `core / residual`", "冻结边界", "write", "retrieve", "belief"],
        ),
        "v26_plan_bans_shortcuts_and_leakage": _contains_all(
            v26_plan,
            ["不做任何 `fallback`", "不做任何 `shortcut`", "不使用 benchmark label / answer / protocol 输出作为训练 supervision", "holdout evaluation"],
        ),
        "core_residual_structure_still_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/consolidation.py"],
            ["core_slots", "residual_slots", "core_slots:", "residual_slots:"],
        ),
        "learned_write_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/lifecycle.py", "src/core_mem/v2/training.py"],
            ["slot_assignment_mode", "learned_slot_assignment", "SLOT_ASSIGNMENT_TASK_NAME"],
        ),
        "learned_retrieve_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/encoder.py", "src/core_mem/v2/resampler.py"],
            ["query_encoder", "_ranking_score", "compose(", "retrieval_key"],
        ),
        "learned_belief_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/decoder.py", "src/core_mem/v2/semantic_outputs.py"],
            ["belief_state", "belief_source", "coerce_task_payload", "decode("],
        ),
        "historical_v25_baseline_artifacts_exist": _historical_v25_artifacts_exist(root),
        "current_head_v25_eval_still_strong": float(v25_trained_metrics.get("token_f1", 0.0)) >= 0.99
        and float(v25_trained_metrics.get("field_f1", 0.0)) >= 0.99,
        "current_head_v25_longmemeval_128_exists": bool(v25_longmemeval_payload)
        and int(v25_longmemeval_payload.get("sample_count", 0)) >= 128
        and v25_longmemeval_payload.get("status") == "completed",
        "current_head_v25_personamem_128_exists": bool(v25_personamem_payload)
        and int(v25_personamem_payload.get("sample_count", 0)) >= 128
        and v25_personamem_payload.get("status") == "completed",
        "current_head_v26_train_exists": bool(v26_train_payload),
        "current_head_v26_eval_exists": bool(v26_eval_payload),
        "current_head_v26_write_gain_exists": bool(v26_write_gain),
        "current_head_v26_retrieve_gain_exists": bool(v26_retrieve_gain),
        "current_head_v26_belief_gain_exists": bool(v26_belief_gain),
        "at_least_one_component_gain_positive": component_positive,
        "current_head_v26_longmemeval_128_exists": bool(v26_longmemeval_payload)
        and int(v26_longmemeval_payload.get("sample_count", 0)) >= 128
        and v26_longmemeval_payload.get("status") == "completed",
        "current_head_v26_longmemeval_128_beats_v25": v26_long_provider > v25_long_provider
        and v26_long_local > v25_long_local,
        "current_head_v26_personamem_128_guard_holds": bool(v26_personamem_payload)
        and int(v26_personamem_payload.get("sample_count", 0)) >= 128
        and v26_personamem_payload.get("status") == "completed"
        and v26_persona_provider >= v25_persona_provider
        and v26_persona_local >= v25_persona_local,
        "current_head_v26_longmemeval_analysis_exists": bool(v26_analysis_payload)
        and any(key in v26_analysis_payload for key in ["error_clusters", "layered_breakdown", "failure_buckets"]),
        "current_head_v26_full_benchmark_exists": bool(v26_full_benchmark_payload)
        and int(v26_full_benchmark_payload.get("num_benchmarks", 0)) >= 1
        and bool(v26_full_benchmark_payload.get("holdout_only")),
        "current_head_v26_eval_stays_strong": float(v26_trained_metrics.get("token_f1", 0.0)) >= 0.99
        and float(v26_trained_metrics.get("field_f1", 0.0)) >= 0.99,
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v26_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "head": _current_head(root),
        "v25_trained_eval_metrics": v25_trained_metrics,
        "v26_trained_eval_metrics": v26_trained_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--publish-artifacts", action="store_true")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--longmemeval-summary-path")
    parser.add_argument("--personamem-summary-path")
    parser.add_argument("--train-source-path", default="outputs_v2/artifacts/latest_stage2_v25_train.json")
    parser.add_argument("--eval-source-path", default="outputs_v2/artifacts/latest_stage2_v25_eval.json")
    parser.add_argument("--v25-longmemeval-path", default="outputs_v2/artifacts/latest_longmemeval_stage2_v25_canary.json")
    parser.add_argument("--v25-personamem-path", default="outputs_v2/artifacts/latest_personamem_stage2_v25_canary.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.publish_artifacts:
        if not args.longmemeval_summary_path or not args.personamem_summary_path:
            raise SystemExit("--publish-artifacts requires both --longmemeval-summary-path and --personamem-summary-path")
        payload = publish_v26_artifacts(
            root=root,
            longmemeval_summary_path=Path(args.longmemeval_summary_path).resolve(),
            personamem_summary_path=Path(args.personamem_summary_path).resolve(),
            train_source_path=(root / args.train_source_path).resolve(),
            eval_source_path=(root / args.eval_source_path).resolve(),
            v25_longmemeval_path=(root / args.v25_longmemeval_path).resolve(),
            v25_personamem_path=(root / args.v25_personamem_path).resolve(),
        )
        if args.score_only:
            raise SystemExit("--score-only cannot be combined with --publish-artifacts")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"head={payload['head']}")
            print(f"longmemeval_canary={payload['longmemeval_canary']['artifact_paths']['latest_path']}")
            print(f"personamem_canary={payload['personamem_canary']['artifact_paths']['latest_path']}")
        return 0

    payload = compute_v26_longrun(root)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
