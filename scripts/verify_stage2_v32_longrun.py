"""Mechanical verifier for the v3.2 / v32 latent-first modular redesign run."""

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


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


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
    value = payload.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_metric(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    value = payload.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rate(payload: dict[str, Any] | None, *, key: str) -> float:
    if not payload:
        return 0.0
    sample_count = _int_metric(payload, "sample_count")
    if sample_count <= 0:
        return 0.0
    return _int_metric(payload, key) / sample_count


def _subset_exact_metrics(summary_payload: dict[str, Any], sample_ids: set[str]) -> dict[str, float]:
    predictions_path = summary_payload.get("predictions_path")
    if not predictions_path:
        return {"sample_count": 0.0, "provider_exact_rate": 0.0, "local_exact_rate": 0.0}
    rows = _load_jsonl(Path(str(predictions_path)))
    scoped = [row for row in rows if str(row.get("sample_id", "")) in sample_ids]
    if not scoped:
        return {"sample_count": 0.0, "provider_exact_rate": 0.0, "local_exact_rate": 0.0}
    provider_exact = sum(
        1
        for row in scoped
        if " ".join(str(row.get("provider_prediction", "")).strip().lower().split())
        == " ".join(str(row.get("expected_answer", "")).strip().lower().split())
    )
    local_exact = sum(
        1
        for row in scoped
        if " ".join(str(row.get("memory_answer_local", "")).strip().lower().split())
        == " ".join(str(row.get("expected_answer", "")).strip().lower().split())
    )
    total = len(scoped)
    return {
        "sample_count": float(total),
        "provider_exact_rate": provider_exact / total,
        "local_exact_rate": local_exact / total,
    }


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
        "provider_exact_rate": _rate(payload, key="provider_exact_match"),
        "local_exact_rate": _rate(payload, key="local_exact_match"),
    }
    alias_payload["artifact_paths"] = _write_latest_and_stamped(root, latest_name, alias_payload)
    return alias_payload


def publish_v32_full_holdout_artifacts(
    *,
    root: Path,
    longmemeval_summary_path: Path,
    personamem_summary_path: Path,
) -> dict[str, Any]:
    current_head = _current_head(root)
    retained_v30_holdout = _artifact_json(root, "latest_stage2_v30_full_holdout_baseline.json") or {}
    retained_v30_long = _artifact_json(root, "latest_longmemeval_stage2_v30_full.json") or {}
    retained_v30_persona = _artifact_json(root, "latest_personamem_stage2_v30_full.json") or {}

    longmemeval_payload = _publish_canary_alias(
        root=root,
        summary_path=longmemeval_summary_path,
        latest_name="latest_longmemeval_stage2_v32_full.json",
        artifact_type="stage2_v32_longmemeval_full",
    )
    personamem_payload = _publish_canary_alias(
        root=root,
        summary_path=personamem_summary_path,
        latest_name="latest_personamem_stage2_v32_full.json",
        artifact_type="stage2_v32_personamem_full",
    )

    longmemeval_gain_confirmed = (
        _int_metric(longmemeval_payload, "provider_exact_match") > _int_metric(retained_v30_long, "provider_exact_match")
        or _int_metric(longmemeval_payload, "local_exact_match") > _int_metric(retained_v30_long, "local_exact_match")
    )
    retained_overlap_manifest = retained_v30_holdout.get("canary_manifest")
    overlap_sample_ids: set[str] = set()
    if retained_overlap_manifest:
        retained_manifest_payload = _read_json(Path(str(retained_overlap_manifest))) or {}
        overlap_sample_ids = {str(sample_id) for sample_id in retained_manifest_payload.get("sample_ids", [])}
    personamem_overlap_metrics = _subset_exact_metrics(personamem_payload, overlap_sample_ids)
    if personamem_overlap_metrics["sample_count"] > 0:
        personamem_gain_confirmed = (
            personamem_overlap_metrics["provider_exact_rate"]
            > _float_metric(retained_v30_holdout, "personamem_overlap_provider_exact_rate")
            and personamem_overlap_metrics["local_exact_rate"]
            >= _float_metric(retained_v30_holdout, "personamem_overlap_local_exact_rate")
        )
    else:
        personamem_gain_confirmed = (
            _float_metric(personamem_payload, "provider_exact_rate")
            > _float_metric(retained_v30_holdout, "personamem_overlap_provider_exact_rate")
            and _float_metric(personamem_payload, "local_exact_rate")
            >= _float_metric(retained_v30_holdout, "personamem_overlap_local_exact_rate")
        )

    holdout_payload = {
        "artifact_type": "stage2_v32_full_holdout_compare",
        "commit_hash": current_head,
        "holdout_only": True,
        "benchmark_runs_executed": 2,
        "longmemeval_summary_path": str(longmemeval_summary_path),
        "personamem_summary_path": str(personamem_summary_path),
        "longmemeval_sample_count": _int_metric(longmemeval_payload, "sample_count"),
        "personamem_sample_count": _int_metric(personamem_payload, "sample_count"),
        "memory_mode": longmemeval_payload.get("memory_mode") or personamem_payload.get("memory_mode"),
        "slot_assignment_mode": longmemeval_payload.get("slot_assignment_mode") or personamem_payload.get("slot_assignment_mode"),
        "longmemeval_gain_confirmed": longmemeval_gain_confirmed,
        "personamem_gain_confirmed": personamem_gain_confirmed,
        "personamem_overlap_sample_count": int(personamem_overlap_metrics["sample_count"]),
        "personamem_overlap_provider_exact_rate": personamem_overlap_metrics["provider_exact_rate"],
        "personamem_overlap_local_exact_rate": personamem_overlap_metrics["local_exact_rate"],
        "v30_longmemeval_provider_exact_match": _int_metric(retained_v30_long, "provider_exact_match"),
        "v30_longmemeval_local_exact_match": _int_metric(retained_v30_long, "local_exact_match"),
        "v30_personamem_provider_exact_rate": _float_metric(retained_v30_persona, "provider_exact_rate"),
        "v30_personamem_local_exact_rate": _float_metric(retained_v30_persona, "local_exact_rate"),
        "note": "V32 full benchmark remains holdout-only and requires clear gains over retained v30 baselines.",
    }
    holdout_payload["artifact_paths"] = _write_latest_and_stamped(root, "latest_stage2_v32_full_holdout_compare.json", holdout_payload)
    return {
        "longmemeval_full": longmemeval_payload,
        "personamem_full": personamem_payload,
        "full_holdout_compare": holdout_payload,
    }


def compute_v32_longrun(root: Path) -> dict[str, Any]:
    docs = root / "docs"
    agent_os = root / ".agent-os"
    current_status = docs / "current_status.md"
    implementation_plan = docs / "implementation_plan.md"
    todo = docs / "todo.md"
    project_index = agent_os / "project-index.md"
    agent_todo = agent_os / "todo.md"
    v32_plan = docs / "v32_plan.md"

    retained_v30_adapter = _artifact_json(root, "latest_stage2_v30_task_adapter_compare.json") or {}
    retained_v30_latent = _artifact_json(root, "latest_stage2_v30_latent_gain.json") or {}
    retained_v30_belief = _artifact_json(root, "latest_stage2_v30_belief_gain.json") or {}
    retained_v30_write = _artifact_json(root, "latest_stage2_v30_write_gain.json") or {}
    retained_v30_holdout = _artifact_json(root, "latest_stage2_v30_full_holdout_baseline.json") or {}
    retained_v30_long = _artifact_json(root, "latest_longmemeval_stage2_v30_full.json") or {}
    retained_v30_persona = _artifact_json(root, "latest_personamem_stage2_v30_full.json") or {}
    blocked_v31_latent = _artifact_json(root, "latest_stage2_v31_latent_holdout_compare.json") or {}
    blocked_v31_belief = _artifact_json(root, "latest_stage2_v31_belief_holdout_compare.json") or {}
    blocked_v31_write = _artifact_json(root, "latest_stage2_v31_write_holdout_compare.json") or {}

    v32_modular_train = _artifact_json(root, "latest_stage2_v32_modular_backbone_train.json") or {}
    v32_write_eval = _artifact_json(root, "latest_stage2_v32_write_head_eval.json") or {}
    v32_latent_train = _artifact_json(root, "latest_stage2_v32_latent_module_train.json") or {}
    v32_latent_objective = _artifact_json(root, "latest_stage2_v32_latent_objective_eval.json") or {}
    v32_latent_compare = _artifact_json(root, "latest_stage2_v32_latent_holdout_compare.json") or {}
    v32_belief_eval = _artifact_json(root, "latest_stage2_v32_belief_decoder_eval.json") or {}
    v32_belief_compare = _artifact_json(root, "latest_stage2_v32_belief_holdout_compare.json") or {}
    v32_answer_eval = _artifact_json(root, "latest_stage2_v32_answer_head_eval.json") or {}
    v32_option_compare = _artifact_json(root, "latest_stage2_v32_option_scoring_compare.json") or {}
    v32_ablation = _artifact_json(root, "latest_stage2_v32_ablation_summary.json") or {}
    v32_holdout = _artifact_json(root, "latest_stage2_v32_full_holdout_compare.json") or {}
    v32_long = _artifact_json(root, "latest_longmemeval_stage2_v32_full.json") or {}
    v32_persona = _artifact_json(root, "latest_personamem_stage2_v32_full.json") or {}

    checks: list[dict[str, Any]] = []

    def add(slug: str, passed: bool, detail: str) -> None:
        checks.append({"slug": slug, "passed": bool(passed), "detail": detail})

    add(
        "current_status_tracks_td043",
        _contains_all(current_status, ["`TD-043`", "`v32`", "latent", "belief", "answer", "500", "512"]),
        "current_status should track TD-043 / v32 latent-first modular redesign",
    )
    add(
        "implementation_plan_mentions_v32_axes",
        _contains_all(implementation_plan, ["`TD-043`", "`v32`", "latent", "belief", "answer", "option-scoring", "500", "512"]),
        "implementation_plan should mention latent-first modular redesign, answer head, and full holdout",
    )
    add(
        "project_index_tracks_ws029",
        _contains_all(project_index, ["`TD-043 / WS-029`", "`v32`"]),
        "project-index should track TD-043 / WS-029 / v32",
    )
    add(
        "todo_tracks_td043_doing",
        _contains_all(agent_todo, ["`TD-043`", "[doing]"]) or _contains_all(todo, ["`TD-043`"]),
        "todo should track TD-043 as active work",
    )
    add("v32_plan_exists", v32_plan.exists(), "docs/v32_plan.md should exist")
    add(
        "v32_plan_mentions_no_fallback",
        _contains_all(v32_plan, ["fallback", "shortcut", "benchmark leakage"]),
        "v32 plan should preserve no-fallback / no-shortcut / no-leakage constraints",
    )
    add(
        "v32_plan_mentions_core_residual_frozen",
        _contains_all(v32_plan, ["core / residual", "不改 `core / residual`"]),
        "v32 plan should explicitly freeze core / residual",
    )
    add(
        "v32_plan_mentions_latent_first_redesign",
        _contains_all(v32_plan, ["latent-first", "modular", "trainable latent reader"]),
        "v32 plan should explicitly define the latent-first redesign",
    )
    add(
        "v32_plan_mentions_modular_heads",
        _contains_all(v32_plan, ["write head", "belief head", "answer / option-scoring head"]),
        "v32 plan should require modular task heads",
    )
    add(
        "v32_plan_mentions_option_scoring",
        _contains_all(v32_plan, ["option scoring", "candidate scoring", "有限候选答案"]),
        "v32 plan should require a general option-scoring / answer head",
    )
    add(
        "v32_plan_mentions_full_holdout",
        _contains_all(v32_plan, ["LongMemEval-S = 500", "PersonaMem = 512", "holdout"]),
        "v32 plan should keep full benchmark as holdout baseline",
    )

    add("retained_v30_adapter_positive", bool(retained_v30_adapter.get("task_specific_positive_gain", False)), "retained v30 task adapters should stay positive")
    add("retained_v30_latent_positive", bool(retained_v30_latent.get("positive_gain", False)), "retained v30 latent gain should stay positive")
    add("retained_v30_belief_positive", bool(retained_v30_belief.get("positive_gain", False)), "retained v30 belief gain should stay positive")
    add("retained_v30_write_positive", bool(retained_v30_write.get("positive_gain", False)), "retained v30 write gain should stay positive")
    add(
        "retained_v30_holdout_full",
        bool(retained_v30_holdout.get("holdout_only", False))
        and _int_metric(retained_v30_holdout, "longmemeval_sample_count") >= 500
        and _int_metric(retained_v30_holdout, "personamem_overlap_sample_count") >= 512,
        "retained v30 full holdout should stay available",
    )
    add("blocked_v31_latent_positive", bool(blocked_v31_latent.get("positive_gain", False)), "blocked v31 latent internal compare should stay positive")
    add("blocked_v31_belief_positive", bool(blocked_v31_belief.get("positive_gain", False)), "blocked v31 belief internal compare should stay positive")
    add("blocked_v31_write_positive", bool(blocked_v31_write.get("positive_gain", False)), "blocked v31 write internal compare should stay positive")

    add("v32_modular_train_exists", _artifact_exists(root, "latest_stage2_v32_modular_backbone_train.json"), "v32 modular backbone train artifact should exist")
    add("v32_write_head_eval_exists", _artifact_exists(root, "latest_stage2_v32_write_head_eval.json"), "v32 write head eval should exist")
    add("v32_latent_module_train_exists", _artifact_exists(root, "latest_stage2_v32_latent_module_train.json"), "v32 latent module train artifact should exist")
    add("v32_latent_objective_eval_exists", _artifact_exists(root, "latest_stage2_v32_latent_objective_eval.json"), "v32 latent objective eval should exist")
    add("v32_latent_holdout_compare_exists", _artifact_exists(root, "latest_stage2_v32_latent_holdout_compare.json"), "v32 latent holdout compare should exist")
    add("v32_belief_decoder_eval_exists", _artifact_exists(root, "latest_stage2_v32_belief_decoder_eval.json"), "v32 belief decoder eval should exist")
    add("v32_belief_holdout_compare_exists", _artifact_exists(root, "latest_stage2_v32_belief_holdout_compare.json"), "v32 belief holdout compare should exist")
    add("v32_answer_head_eval_exists", _artifact_exists(root, "latest_stage2_v32_answer_head_eval.json"), "v32 answer head eval should exist")
    add("v32_option_scoring_compare_exists", _artifact_exists(root, "latest_stage2_v32_option_scoring_compare.json"), "v32 option-scoring compare should exist")
    add("v32_ablation_summary_exists", _artifact_exists(root, "latest_stage2_v32_ablation_summary.json"), "v32 ablation summary should exist")
    add("v32_full_holdout_compare_exists", _artifact_exists(root, "latest_stage2_v32_full_holdout_compare.json"), "v32 full holdout compare should exist")
    add(
        "v32_long_full_exists",
        _artifact_exists(root, "latest_longmemeval_stage2_v32_full.json")
        and _int_metric(v32_long, "sample_count") >= 500,
        "v32 LongMemEval full artifact should exist with 500-sample gate coverage",
    )
    add(
        "v32_persona_full_exists",
        _artifact_exists(root, "latest_personamem_stage2_v32_full.json")
        and _int_metric(v32_persona, "sample_count") >= 512,
        "v32 PersonaMem full artifact should exist with 512-sample stable gate coverage",
    )

    add("v32_modular_architecture_positive", bool(v32_modular_train.get("positive_gain", False)), "v32 modular architecture compare must be positive")
    add("v32_write_positive", bool(v32_write_eval.get("positive_gain", False)), "v32 write head must be positive")
    add("v32_latent_module_positive", bool(v32_latent_train.get("trainable_latent", False)) or bool(v32_latent_train.get("positive_gain", False)), "v32 latent module must be trainable and landed")
    add("v32_latent_objective_positive", bool(v32_latent_objective.get("positive_gain", False)), "v32 latent objective must be positive")
    add("v32_latent_holdout_positive", bool(v32_latent_compare.get("positive_gain", False)), "v32 latent holdout compare must be positive")
    add("v32_belief_positive", bool(v32_belief_compare.get("positive_gain", False)) and bool(v32_belief_eval.get("positive_gain", False)), "v32 belief head must be positive")
    add("v32_answer_head_positive", bool(v32_answer_eval.get("positive_gain", False)), "v32 answer head must be positive")
    add("v32_option_scoring_positive", bool(v32_option_compare.get("positive_gain", False)), "v32 option-scoring compare must be positive")
    add(
        "v32_longmemeval_beats_v30",
        bool(v32_holdout.get("longmemeval_gain_confirmed", False))
        or _int_metric(v32_long, "provider_exact_match") > _int_metric(retained_v30_long, "provider_exact_match")
        or _int_metric(v32_long, "local_exact_match") > _int_metric(retained_v30_long, "local_exact_match"),
        "v32 LongMemEval should beat retained v30 on at least one exact metric",
    )
    add(
        "v32_personamem_beats_v30",
        bool(v32_holdout.get("personamem_gain_confirmed", False))
        or (
            _float_metric(v32_holdout, "personamem_overlap_provider_exact_rate")
            > _float_metric(retained_v30_holdout, "personamem_overlap_provider_exact_rate")
            and _float_metric(v32_holdout, "personamem_overlap_local_exact_rate")
            >= _float_metric(retained_v30_holdout, "personamem_overlap_local_exact_rate")
        ),
        "v32 PersonaMem 512 should beat retained v30 overlap guard, not merely match it",
    )
    add(
        "v32_ablation_confirms_latent_mainline",
        bool(v32_ablation.get("latent_is_primary_driver", False))
        and bool(v32_ablation.get("belief_contributes", False))
        and bool(v32_ablation.get("answer_head_contributes", False)),
        "v32 ablation should confirm latent as primary driver, with belief and answer head contributions",
    )
    add(
        "v32_holdout_guard",
        bool(v32_holdout.get("holdout_only", False))
        and bool(v32_holdout.get("longmemeval_gain_confirmed", False))
        and bool(v32_holdout.get("personamem_gain_confirmed", False)),
        "v32 holdout compare should confirm holdout-only benchmark use and gains on both benchmarks",
    )

    score = sum(1 for item in checks if item["passed"])
    total = len(checks)
    return {
        "score": score,
        "total": total,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-full-holdout-artifacts", action="store_true")
    parser.add_argument("--longmemeval-summary", type=Path)
    parser.add_argument("--personamem-summary", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    if args.publish_full_holdout_artifacts:
        if not args.longmemeval_summary or not args.personamem_summary:
            raise SystemExit("--publish-full-holdout-artifacts requires both --longmemeval-summary and --personamem-summary")
        if args.score_only:
            raise SystemExit("--score-only cannot be combined with --publish-full-holdout-artifacts")
        payload = publish_v32_full_holdout_artifacts(
            root=Path(args.repo_root),
            longmemeval_summary_path=args.longmemeval_summary,
            personamem_summary_path=args.personamem_summary,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("published_v32_full_holdout_artifacts")
        return 0

    payload = compute_v32_longrun(Path(args.repo_root))
    if args.score_only:
        print(payload["score"])
        return 0
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"stage2_v32_longrun_score={payload['score']}/{payload['total']}")
    for check in payload["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['slug']}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
