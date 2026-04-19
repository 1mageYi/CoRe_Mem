"""Mechanical verifier for the v2.9 learned-core-path long run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else "unknown"


def _trained_eval_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    return (((payload or {}).get("trained_eval")) or {}).get("metrics") or {}


def _trained_eval_task_metrics(payload: dict[str, Any] | None, task_name: str) -> dict[str, Any]:
    per_task = (((payload or {}).get("trained_eval")) or {}).get("per_task") or {}
    return per_task.get(task_name) or {}


def _float_metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_metric(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_latest_and_stamped(root: Path, latest_name: str, payload: dict[str, Any]) -> dict[str, str]:
    artifact_root = root / "outputs_v2" / "artifacts"
    latest_path = artifact_root / latest_name
    stamped_path = artifact_root / f"{_timestamp()}_{latest_name}"
    _write_json(latest_path, payload)
    _write_json(stamped_path, payload)
    return {"latest_path": str(latest_path), "stamped_path": str(stamped_path)}


def _publish_canary_alias(
    *,
    root: Path,
    summary_path: Path,
    latest_name: str,
    artifact_type: str,
) -> dict[str, Any]:
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
    }
    alias_payload["artifact_paths"] = _write_latest_and_stamped(root, latest_name, alias_payload)
    return alias_payload


def publish_v29_artifacts(
    *,
    root: Path,
    train_summary_path: Path,
    val_eval_path: Path,
    test_eval_path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    test_manifest_path: Path,
    longmemeval_summary_path: Path,
    personamem_summary_path: Path,
) -> dict[str, Any]:
    current_head = _current_head(root)
    train_summary = _read_json(train_summary_path)
    val_eval = _read_json(val_eval_path)
    test_eval = _read_json(test_eval_path)
    train_manifest = _read_json(train_manifest_path)
    val_manifest = _read_json(val_manifest_path)
    test_manifest = _read_json(test_manifest_path)
    if not all([train_summary, val_eval, test_eval, train_manifest, val_manifest, test_manifest]):
        raise FileNotFoundError("Missing v29 publish inputs.")

    retained_v27_eval = _artifact_json(root, "latest_stage2_v27_eval.json") or {}
    retained_v26_long = _artifact_json(root, "latest_longmemeval_stage2_v26_canary.json") or {}
    retained_v26_persona = _artifact_json(root, "latest_personamem_stage2_v26_canary.json") or {}

    train_payload = {
        "artifact_type": "stage2_v29_training_timing",
        "commit_hash": current_head,
        "source_split": "train",
        "prepared_manifest": str(train_manifest_path),
        "task_counts": train_manifest.get("task_counts", {}),
        "device": train_summary.get("device"),
        "cuda_visible_devices": train_summary.get("cuda_visible_devices"),
        "gpu_name": train_summary.get("gpu_name"),
        "num_examples": train_summary.get("num_examples"),
        "num_steps": train_summary.get("num_steps"),
        "optimizer_steps": train_summary.get("optimizer_steps"),
        "wall_clock_seconds": train_summary.get("wall_clock_seconds"),
        "examples_per_second": train_summary.get("examples_per_second"),
        "steps_per_second": train_summary.get("steps_per_second"),
        "peak_gpu_memory_mb": train_summary.get("peak_gpu_memory_mb"),
        "run_dir": train_summary.get("run_dir"),
        "metrics_path": train_summary.get("metrics_path"),
        "checkpoint_dir": train_summary.get("checkpoint_dir"),
        "train_summary_path": str(train_summary_path),
        "train_config_path": train_summary.get("config_snapshot_path") or train_summary.get("config_path"),
    }
    train_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v29_training_timing.json", train_payload)

    longmemeval_payload = _publish_canary_alias(
        root=root,
        summary_path=longmemeval_summary_path,
        latest_name="latest_longmemeval_stage2_v29_canary.json",
        artifact_type="stage2_v29_longmemeval_canary",
    )
    personamem_payload = _publish_canary_alias(
        root=root,
        summary_path=personamem_summary_path,
        latest_name="latest_personamem_stage2_v29_canary.json",
        artifact_type="stage2_v29_personamem_canary",
    )

    holdout_payload = {
        "artifact_type": "stage2_v29_holdout_summary",
        "commit_hash": current_head,
        "holdout_only": True,
        "benchmark_runs_executed": 2,
        "training_manifests": [
            str(train_manifest_path),
            str(val_manifest_path),
            str(test_manifest_path),
        ],
        "train_config_path": train_payload["train_config_path"],
        "val_eval_path": str(val_eval_path),
        "test_eval_path": str(test_eval_path),
        "longmemeval_summary_path": str(longmemeval_summary_path),
        "personamem_summary_path": str(personamem_summary_path),
        "longmemeval_sample_count": _int_metric(longmemeval_payload, "sample_count"),
        "personamem_sample_count": _int_metric(personamem_payload, "sample_count"),
        "longmemeval_provider_configured": bool(longmemeval_payload.get("provider_configured", False)),
        "personamem_provider_configured": bool(personamem_payload.get("provider_configured", False)),
        "memory_mode": longmemeval_payload.get("memory_mode") or personamem_payload.get("memory_mode"),
        "slot_assignment_mode": longmemeval_payload.get("slot_assignment_mode") or personamem_payload.get("slot_assignment_mode"),
        "note": "Benchmarks remain holdout-only for v2.9. This summary may still be provider-blocked while local memory outputs continue to be auditable.",
    }
    holdout_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v29_holdout_summary.json", holdout_payload)

    baseline_lifecycle = _trained_eval_task_metrics(retained_v27_eval, "lifecycle_prediction")
    current_lifecycle = _trained_eval_task_metrics(test_eval, "lifecycle_prediction")
    baseline_retrieval = _trained_eval_task_metrics(retained_v27_eval, "retrieval_alignment")
    current_retrieval = _trained_eval_task_metrics(test_eval, "retrieval_alignment")
    baseline_belief = _trained_eval_task_metrics(retained_v27_eval, "composition_to_belief")
    current_belief = _trained_eval_task_metrics(test_eval, "composition_to_belief")
    val_trained = _trained_eval_metrics(val_eval)
    test_trained = _trained_eval_metrics(test_eval)

    long_local_delta = _int_metric(longmemeval_payload, "local_exact_match") - _int_metric(retained_v26_long, "local_exact_match")
    persona_local_delta = _int_metric(personamem_payload, "local_exact_match") - _int_metric(retained_v26_persona, "local_exact_match")

    write_payload = {
        "artifact_type": "stage2_v29_write_gain",
        "component": "write",
        "commit_hash": current_head,
        "train_summary_path": str(train_summary_path),
        "test_eval_path": str(test_eval_path),
        "baseline_eval_path": str(root / "outputs_v2" / "artifacts" / "latest_stage2_v27_eval.json"),
        "baseline_longmemeval_path": str(root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json"),
        "baseline_personamem_path": str(root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json"),
        "baseline_lifecycle_prediction": baseline_lifecycle,
        "current_lifecycle_prediction": current_lifecycle,
        "delta_token_f1": _float_metric(current_lifecycle, "token_f1") - _float_metric(baseline_lifecycle, "token_f1"),
        "delta_field_f1": _float_metric(current_lifecycle, "field_f1") - _float_metric(baseline_lifecycle, "field_f1"),
        "delta_exact_match": _float_metric(current_lifecycle, "exact_match") - _float_metric(baseline_lifecycle, "exact_match"),
        "delta_longmemeval_local_exact_match": long_local_delta,
        "delta_personamem_local_exact_match": persona_local_delta,
        "personamem_local_guard_retained": _int_metric(personamem_payload, "local_exact_match") >= _int_metric(retained_v26_persona, "local_exact_match"),
        "positive_gain": (
            (_float_metric(current_lifecycle, "token_f1") > _float_metric(baseline_lifecycle, "token_f1")
             or _float_metric(current_lifecycle, "field_f1") > _float_metric(baseline_lifecycle, "field_f1"))
            and _int_metric(personamem_payload, "local_exact_match") >= _int_metric(retained_v26_persona, "local_exact_match")
        ),
    }
    write_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v29_write_gain.json", write_payload)

    latent_payload = {
        "artifact_type": "stage2_v29_latent_gain",
        "component": "latent",
        "commit_hash": current_head,
        "baseline_retrieval_alignment": baseline_retrieval,
        "current_retrieval_alignment": current_retrieval,
        "val_retrieval_family": (val_eval.get("families") or {}).get("retrieval_family", {}),
        "test_retrieval_family": (test_eval.get("families") or {}).get("retrieval_family", {}),
        "test_resampler_metrics": (test_eval.get("modules") or {}).get("resampler", {}),
        "delta_token_f1": _float_metric(current_retrieval, "token_f1") - _float_metric(baseline_retrieval, "token_f1"),
        "delta_field_f1": _float_metric(current_retrieval, "field_f1") - _float_metric(baseline_retrieval, "field_f1"),
        "delta_exact_match": _float_metric(current_retrieval, "exact_match") - _float_metric(baseline_retrieval, "exact_match"),
        "trained_eval_token_f1": _float_metric(test_trained, "token_f1"),
        "trained_eval_field_f1": _float_metric(test_trained, "field_f1"),
        "positive_gain": (
            _float_metric(current_retrieval, "token_f1") > _float_metric(baseline_retrieval, "token_f1")
            or _float_metric(current_retrieval, "field_f1") > _float_metric(baseline_retrieval, "field_f1")
        ),
    }
    latent_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v29_latent_gain.json", latent_payload)

    belief_payload = {
        "artifact_type": "stage2_v29_belief_gain",
        "component": "belief",
        "commit_hash": current_head,
        "baseline_composition_to_belief": baseline_belief,
        "current_composition_to_belief": current_belief,
        "val_belief_recovery_family": (val_eval.get("families") or {}).get("belief_recovery_family", {}),
        "test_belief_recovery_family": (test_eval.get("families") or {}).get("belief_recovery_family", {}),
        "test_belief_decoder_metrics": (test_eval.get("modules") or {}).get("belief_decoder", {}),
        "delta_token_f1": _float_metric(current_belief, "token_f1") - _float_metric(baseline_belief, "token_f1"),
        "delta_field_f1": _float_metric(current_belief, "field_f1") - _float_metric(baseline_belief, "field_f1"),
        "delta_exact_match": _float_metric(current_belief, "exact_match") - _float_metric(baseline_belief, "exact_match"),
        "trained_eval_token_f1": _float_metric(test_trained, "token_f1"),
        "trained_eval_field_f1": _float_metric(test_trained, "field_f1"),
        "positive_gain": (
            _float_metric(current_belief, "token_f1") > _float_metric(baseline_belief, "token_f1")
            or _float_metric(current_belief, "field_f1") > _float_metric(baseline_belief, "field_f1")
        ),
    }
    belief_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v29_belief_gain.json", belief_payload)

    return {
        "training_timing": train_payload,
        "write_gain": write_payload,
        "latent_gain": latent_payload,
        "belief_gain": belief_payload,
        "holdout_summary": holdout_payload,
        "longmemeval_canary": longmemeval_payload,
        "personamem_canary": personamem_payload,
    }


def compute_v29_longrun(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(slug: str, passed: bool, detail: str) -> None:
        checks.append({"slug": slug, "passed": bool(passed), "detail": detail})

    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo_doc = root / ".agent-os" / "todo.md"
    v29_plan = root / "docs" / "v29_plan.md"

    add(
        "current_status_tracks_td040",
        _contains_all(current_status, ["`TD-040`", "`v2.9`", "write", "latent", "belief", "32k"]),
        "current_status should track TD-040 / v2.9 learned-core-path on 32k",
    )
    add(
        "implementation_plan_mentions_v29_axes",
        _contains_all(implementation_plan, ["`TD-040`", "`v2.9`", "write", "latent", "belief", "500", "512"]),
        "implementation_plan should mention v2.9 write/latent/belief and expanded 500/512 holdout",
    )
    add(
        "project_index_tracks_ws026",
        _contains_all(project_index, ["`TD-040 / WS-026`", "`v2.9`"]),
        "project-index should track TD-040 / WS-026",
    )
    add(
        "todo_tracks_td040_doing",
        _contains_all(todo_doc, ["`TD-040`", "`[doing]`", "write", "latent", "belief"]),
        "todo should track TD-040 as doing",
    )
    add("v29_plan_exists", v29_plan.exists(), "docs/v29_plan.md should exist")
    add(
        "v29_plan_mentions_no_fallback",
        _contains_all(v29_plan, ["不做任何 `fallback`", "不做任何 `shortcut`"]),
        "v29 plan should keep no-fallback / no-shortcut constraints",
    )
    add(
        "v29_plan_mentions_core_residual_frozen",
        _contains_all(v29_plan, ["不改 `core / residual`", "双银行结构"]),
        "v29 plan should explicitly freeze core / residual",
    )
    add(
        "v29_plan_mentions_32k_and_expanded_holdout",
        _contains_all(v29_plan, ["24k train", "4k val", "4k test", "LongMemEval-S 128 -> 500", "PersonaMem 128 -> 512"]),
        "v29 plan should anchor on 32k and expanded 512 holdout",
    )

    for artifact_name in [
        "latest_stage2_v26_write_gain.json",
        "latest_stage2_v26_belief_gain.json",
        "latest_longmemeval_stage2_v26_canary.json",
        "latest_personamem_stage2_v26_canary.json",
        "latest_stage2_v27_32k_split.json",
        "latest_stage2_v27_32k_manifest.json",
        "latest_stage2_v27_32k_audit.json",
        "latest_stage2_v27_train.json",
        "latest_stage2_v27_eval.json",
        "latest_stage2_v27_training_timing.json",
        "latest_stage2_v28_teacher_compare.json",
        "latest_stage2_v28_internal_test.json",
    ]:
        add(
            artifact_name.replace(".", "_"),
            _artifact_exists(root, artifact_name),
            f"{artifact_name} should exist as retained baseline evidence",
        )

    v26_write = _artifact_json(root, "latest_stage2_v26_write_gain.json") or {}
    v26_belief = _artifact_json(root, "latest_stage2_v26_belief_gain.json") or {}
    v26_long = _artifact_json(root, "latest_longmemeval_stage2_v26_canary.json") or {}
    v26_persona = _artifact_json(root, "latest_personamem_stage2_v26_canary.json") or {}
    v28_compare = _artifact_json(root, "latest_stage2_v28_teacher_compare.json") or {}
    v28_internal = _artifact_json(root, "latest_stage2_v28_internal_test.json") or {}

    add(
        "v26_write_positive_gain_retained",
        bool(v26_write.get("positive_gain", False)),
        "v26 retained write gain should remain positive",
    )
    add(
        "v26_belief_positive_gain_retained",
        bool(v26_belief.get("positive_gain", False)),
        "v26 retained belief gain should remain positive",
    )
    add(
        "v26_longmemeval_retained_breakout",
        int(v26_long.get("provider_exact_match", 0)) >= 11 and int(v26_long.get("local_exact_match", 0)) >= 11,
        "v26 retained LongMemEval-S baseline should remain >= 11/11",
    )
    add(
        "v26_personamem_guard_retained",
        int(v26_persona.get("provider_exact_match", 0)) >= 44 and int(v26_persona.get("local_exact_match", 0)) >= 33,
        "v26 retained PersonaMem guard should remain >= 44/33",
    )
    add(
        "v28_blocked_truth_preserved",
        float(v28_compare.get("delta_internal_token_f1", 0.0)) <= 0.0 and not bool(v28_internal.get("gate_passed", True)),
        "v28 compare should preserve the current blocked truth (no positive internal teacher delta)",
    )

    add("v29_write_gain_exists", _artifact_exists(root, "latest_stage2_v29_write_gain.json"), "v29 write gain artifact should exist")
    add("v29_latent_gain_exists", _artifact_exists(root, "latest_stage2_v29_latent_gain.json"), "v29 latent gain artifact should exist")
    add("v29_belief_gain_exists", _artifact_exists(root, "latest_stage2_v29_belief_gain.json"), "v29 belief gain artifact should exist")
    add("v29_training_timing_exists", _artifact_exists(root, "latest_stage2_v29_training_timing.json"), "v29 training timing artifact should exist")
    add("v29_holdout_summary_exists", _artifact_exists(root, "latest_stage2_v29_holdout_summary.json"), "v29 expanded holdout summary should exist")
    add("v29_longmemeval_canary_exists", _artifact_exists(root, "latest_longmemeval_stage2_v29_canary.json"), "v29 LongMemEval canary should exist")
    add("v29_personamem_canary_exists", _artifact_exists(root, "latest_personamem_stage2_v29_canary.json"), "v29 PersonaMem canary should exist")

    v29_write = _artifact_json(root, "latest_stage2_v29_write_gain.json") or {}
    v29_latent = _artifact_json(root, "latest_stage2_v29_latent_gain.json") or {}
    v29_belief = _artifact_json(root, "latest_stage2_v29_belief_gain.json") or {}
    v29_timing = _artifact_json(root, "latest_stage2_v29_training_timing.json") or {}
    v29_holdout = _artifact_json(root, "latest_stage2_v29_holdout_summary.json") or {}
    v29_long = _artifact_json(root, "latest_longmemeval_stage2_v29_canary.json") or {}
    v29_persona = _artifact_json(root, "latest_personamem_stage2_v29_canary.json") or {}

    add("v29_write_positive_gain", bool(v29_write.get("positive_gain", False)), "v29 write gain should be positive")
    add("v29_latent_positive_gain", bool(v29_latent.get("positive_gain", False)), "v29 latent gain should be positive")
    add("v29_belief_positive_gain", bool(v29_belief.get("positive_gain", False)), "v29 belief gain should be positive")
    add(
        "v29_training_timing_gpu2_recorded",
        str(v29_timing.get("device", "")).lower() == "cuda" and str(v29_timing.get("cuda_visible_devices", "")) == "2",
        "v29 training timing should record gpu2 usage",
    )
    add(
        "v29_holdout_summary_expanded",
        int(v29_holdout.get("longmemeval_sample_count", 0)) >= 500 and int(v29_holdout.get("personamem_sample_count", 0)) >= 512,
        "v29 holdout summary should include expanded 500-sample LongMemEval-S and 512-sample PersonaMem runs",
    )
    add(
        "v29_longmemeval_breaks_v26_baseline",
        int(v29_long.get("provider_exact_match", 0)) >= 12 and int(v29_long.get("local_exact_match", 0)) >= 12,
        "v29 LongMemEval-S should exceed the retained 11/11 v26 baseline",
    )
    add(
        "v29_personamem_guard_held",
        int(v29_persona.get("provider_exact_match", 0)) >= 44 and int(v29_persona.get("local_exact_match", 0)) >= 33,
        "v29 PersonaMem guard should remain at least 44/33",
    )

    passed = sum(1 for check in checks if check["passed"])
    total = len(checks)
    return {"score": passed, "total": total, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify stage-2 v2.9 long-run status")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publish-artifacts", action="store_true")
    parser.add_argument("--train-summary", type=Path)
    parser.add_argument("--val-eval", type=Path)
    parser.add_argument("--test-eval", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--longmemeval-summary", type=Path)
    parser.add_argument("--personamem-summary", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.publish_artifacts:
        required = [
            args.train_summary,
            args.val_eval,
            args.test_eval,
            args.train_manifest,
            args.val_manifest,
            args.test_manifest,
            args.longmemeval_summary,
            args.personamem_summary,
        ]
        if not all(required):
            raise SystemExit("--publish-artifacts requires train/eval/manifest inputs plus both benchmark summaries.")
        if args.score_only:
            raise SystemExit("--score-only cannot be combined with --publish-artifacts")
        payload = publish_v29_artifacts(
            root=args.root,
            train_summary_path=args.train_summary,
            val_eval_path=args.val_eval,
            test_eval_path=args.test_eval,
            train_manifest_path=args.train_manifest,
            val_manifest_path=args.val_manifest,
            test_manifest_path=args.test_manifest,
            longmemeval_summary_path=args.longmemeval_summary,
            personamem_summary_path=args.personamem_summary,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("published_v29_artifacts")
        return 0

    payload = compute_v29_longrun(args.root)
    if args.score_only:
        print(payload["score"])
    elif args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"stage2_v29_longrun_score={payload['score']}/{payload['total']}")
        for check in payload["checks"]:
            marker = "PASS" if check["passed"] else "FAIL"
            print(f"[{marker}] {check['slug']}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
