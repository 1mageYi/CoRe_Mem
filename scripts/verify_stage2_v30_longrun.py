"""Mechanical verifier for the v3.0 / v30 architecture-focused long run."""

from __future__ import annotations

import argparse
import json
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


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


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


def compute_v30_longrun(root: Path) -> dict[str, Any]:
    docs = root / "docs"
    agent_os = root / ".agent-os"
    current_status = docs / "current_status.md"
    implementation_plan = docs / "implementation_plan.md"
    todo = docs / "todo.md"
    project_index = agent_os / "project-index.md"
    agent_todo = agent_os / "todo.md"
    v30_plan = docs / "v30_plan.md"

    retained_v29_write = _artifact_json(root, "latest_stage2_v29_write_gain.json") or {}
    retained_v29_latent = _artifact_json(root, "latest_stage2_v29_latent_gain.json") or {}
    retained_v29_belief = _artifact_json(root, "latest_stage2_v29_belief_gain.json") or {}
    retained_v29_holdout = _artifact_json(root, "latest_stage2_v29_holdout_summary.json") or {}
    retained_v29_long = _artifact_json(root, "latest_longmemeval_stage2_v29_canary.json") or {}
    retained_v29_persona = _artifact_json(root, "latest_personamem_stage2_v29_canary.json") or {}

    v30_full_holdout = _artifact_json(root, "latest_stage2_v30_full_holdout_baseline.json") or {}
    v30_shared_train = _artifact_json(root, "latest_stage2_v30_shared_backbone_train.json") or {}
    v30_adapter_compare = _artifact_json(root, "latest_stage2_v30_task_adapter_compare.json") or {}
    v30_latent_train = _artifact_json(root, "latest_stage2_v30_latent_module_train.json") or {}
    v30_latent_objective = _artifact_json(root, "latest_stage2_v30_latent_objective_eval.json") or {}
    v30_belief_decoder = _artifact_json(root, "latest_stage2_v30_belief_decoder_eval.json") or {}
    v30_write_gain = _artifact_json(root, "latest_stage2_v30_write_gain.json") or {}
    v30_latent_gain = _artifact_json(root, "latest_stage2_v30_latent_gain.json") or {}
    v30_belief_gain = _artifact_json(root, "latest_stage2_v30_belief_gain.json") or {}
    v30_long_full = _artifact_json(root, "latest_longmemeval_stage2_v30_full.json") or {}
    v30_persona_full = _artifact_json(root, "latest_personamem_stage2_v30_full.json") or {}

    checks: list[dict[str, Any]] = []

    def add(slug: str, passed: bool, detail: str) -> None:
        checks.append({"slug": slug, "passed": bool(passed), "detail": detail})

    add(
        "current_status_tracks_td041",
        _contains_all(current_status, ["`TD-041`", "`v3.0`", "`v30`", "shared backbone", "latent", "belief", "589", "500"]),
        "current_status should track TD-041 / v30 architecture-first long run and full benchmark holdout",
    )
    add(
        "implementation_plan_mentions_v30_axes",
        _contains_all(
            implementation_plan,
            ["`TD-041`", "`v30`", "shared backbone", "task-specific", "trainable latent", "belief", "589", "500"],
        ),
        "implementation_plan should mention shared backbone + task adapters, trainable latent, learned belief, and full benchmark holdout",
    )
    add(
        "project_index_tracks_ws027",
        _contains_all(project_index, ["`TD-041 / WS-027`", "`v30`"]),
        "project-index should track TD-041 / WS-027 / v30",
    )
    add(
        "todo_tracks_td041_doing",
        _contains_all(agent_todo, ["`TD-041`", "[doing]"]) or _contains_all(todo, ["`TD-041`"]),
        "todo should track TD-041 as active work",
    )
    add("v30_plan_exists", v30_plan.exists(), "docs/v30_plan.md should exist")
    add(
        "v30_plan_mentions_no_fallback",
        _contains_all(v30_plan, ["fallback", "shortcut", "benchmark leakage"]),
        "v30 plan should keep no-fallback / no-shortcut / no-leakage constraints",
    )
    add(
        "v30_plan_mentions_core_residual_frozen",
        _contains_all(v30_plan, ["core / residual", "不改 `core / residual`"]),
        "v30 plan should explicitly freeze core / residual",
    )
    add(
        "v30_plan_mentions_shared_backbone_task_adapters",
        _contains_all(v30_plan, ["shared backbone", "task-specific adapters"]),
        "v30 plan should explicitly target shared backbone + task-specific adapters",
    )
    add(
        "v30_plan_mentions_trainable_latent",
        _contains_all(v30_plan, ["trainable latent", "encoder / resampler"]),
        "v30 plan should explicitly target trainable latent modules",
    )
    add(
        "v30_plan_mentions_direct_latent_objectives",
        _contains_all(v30_plan, ["direct latent", "retrieval ranking", "support-slot"]),
        "v30 plan should explicitly target direct latent objectives",
    )
    add(
        "v30_plan_mentions_learned_belief_decoder",
        _contains_all(v30_plan, ["learned belief decoder", "belief"]),
        "v30 plan should explicitly target learned belief decoder",
    )
    add(
        "v30_plan_mentions_full_benchmark_holdout",
        _contains_all(v30_plan, ["LongMemEval-S = 500", "PersonaMem = 589", "holdout"]),
        "v30 plan should explicitly lock full benchmark holdout sizes",
    )

    add(
        "v29_write_positive_gain_retained",
        bool(retained_v29_write.get("positive_gain", False)),
        "v29 retained write gain should remain positive",
    )
    add(
        "v29_latent_positive_gain_retained",
        bool(retained_v29_latent.get("positive_gain", False)),
        "v29 retained latent gain should remain positive",
    )
    add(
        "v29_belief_positive_gain_retained",
        bool(retained_v29_belief.get("positive_gain", False)),
        "v29 retained belief gain should remain positive",
    )
    add(
        "v29_holdout_retained_fullish",
        _int_metric(retained_v29_holdout, "longmemeval_sample_count") >= 500
        and _int_metric(retained_v29_holdout, "personamem_sample_count") >= 512,
        "v29 retained expanded holdout should still exist",
    )
    add(
        "v29_longmemeval_retained_breakout",
        _int_metric(retained_v29_long, "provider_exact_match") >= 19 and _int_metric(retained_v29_long, "local_exact_match") >= 14,
        "v29 retained LongMemEval-S 500 evidence should remain available",
    )
    add(
        "v29_personamem_retained_breakout",
        _int_metric(retained_v29_persona, "provider_exact_match") >= 171 and _int_metric(retained_v29_persona, "local_exact_match") >= 108,
        "v29 retained PersonaMem 512 evidence should remain available",
    )

    add("v30_full_holdout_baseline_exists", _artifact_exists(root, "latest_stage2_v30_full_holdout_baseline.json"), "v30 full holdout baseline should exist")
    add("v30_shared_backbone_train_exists", _artifact_exists(root, "latest_stage2_v30_shared_backbone_train.json"), "v30 shared-backbone train artifact should exist")
    add("v30_task_adapter_compare_exists", _artifact_exists(root, "latest_stage2_v30_task_adapter_compare.json"), "v30 task-adapter compare artifact should exist")
    add("v30_latent_module_train_exists", _artifact_exists(root, "latest_stage2_v30_latent_module_train.json"), "v30 latent-module train artifact should exist")
    add("v30_latent_objective_eval_exists", _artifact_exists(root, "latest_stage2_v30_latent_objective_eval.json"), "v30 latent objective eval should exist")
    add("v30_belief_decoder_eval_exists", _artifact_exists(root, "latest_stage2_v30_belief_decoder_eval.json"), "v30 belief decoder eval should exist")
    add("v30_write_gain_exists", _artifact_exists(root, "latest_stage2_v30_write_gain.json"), "v30 write gain artifact should exist")
    add("v30_latent_gain_exists", _artifact_exists(root, "latest_stage2_v30_latent_gain.json"), "v30 latent gain artifact should exist")
    add("v30_belief_gain_exists", _artifact_exists(root, "latest_stage2_v30_belief_gain.json"), "v30 belief gain artifact should exist")
    add("v30_longmemeval_full_exists", _artifact_exists(root, "latest_longmemeval_stage2_v30_full.json"), "v30 LongMemEval-S full benchmark artifact should exist")
    add("v30_personamem_full_exists", _artifact_exists(root, "latest_personamem_stage2_v30_full.json"), "v30 PersonaMem full benchmark artifact should exist")

    add(
        "v30_full_holdout_is_holdout_only",
        bool(v30_full_holdout.get("holdout_only", False)),
        "v30 full benchmark baseline must explicitly remain holdout-only",
    )
    add(
        "v30_full_holdout_counts_full",
        _int_metric(v30_full_holdout, "longmemeval_sample_count") >= 500
        and _int_metric(v30_full_holdout, "personamem_sample_count") >= 589,
        "v30 full holdout baseline should cover LongMemEval-S 500 and PersonaMem 589",
    )
    add(
        "v30_shared_backbone_train_gpu2_recorded",
        str(v30_shared_train.get("cuda_visible_devices", "")) == "2",
        "v30 shared-backbone train should record gpu2 usage",
    )
    add(
        "v30_task_adapter_positive_gain",
        bool(v30_adapter_compare.get("task_specific_positive_gain", False)),
        "v30 task-specific adapter compare should be positive",
    )
    add(
        "v30_latent_module_is_trainable",
        bool(v30_latent_train.get("trainable_encoder_resampler", False)),
        "v30 latent-module artifact should confirm encoder/resampler are trainable",
    )
    add(
        "v30_direct_latent_objective_positive",
        bool(v30_latent_objective.get("positive_gain", False)),
        "v30 direct latent objective should be positive",
    )
    add(
        "v30_learned_belief_decoder_positive",
        bool(v30_belief_decoder.get("positive_gain", False)),
        "v30 learned belief decoder eval should be positive",
    )
    add(
        "v30_write_positive_gain",
        bool(v30_write_gain.get("positive_gain", False)),
        "v30 write gain should be positive",
    )
    add(
        "v30_latent_positive_gain",
        bool(v30_latent_gain.get("positive_gain", False)),
        "v30 latent gain should be positive",
    )
    add(
        "v30_belief_positive_gain",
        bool(v30_belief_gain.get("positive_gain", False)),
        "v30 belief gain should be positive",
    )
    add(
        "v30_longmemeval_full_nonregression",
        bool(v30_full_holdout.get("longmemeval_nonregression_guard", False))
        or (
            _int_metric(v30_long_full, "provider_exact_match") >= _int_metric(retained_v29_long, "provider_exact_match")
            and _int_metric(v30_long_full, "local_exact_match") >= _int_metric(retained_v29_long, "local_exact_match")
        ),
        "v30 LongMemEval full benchmark should not regress against v29 expanded holdout",
    )
    add(
        "v30_personamem_full_nonregression",
        bool(v30_full_holdout.get("personamem_nonregression_guard", False))
        or (
            _float_metric(v30_persona_full, "provider_exact_rate") >= _float_metric(v30_full_holdout, "v29_personamem_provider_exact_rate")
            and _float_metric(v30_persona_full, "local_exact_rate") >= _float_metric(v30_full_holdout, "v29_personamem_local_exact_rate")
        ),
        "v30 PersonaMem full benchmark should not regress against v29 expanded holdout rate",
    )

    score = sum(1 for check in checks if check["passed"])
    return {"score": score, "total": len(checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the v30 architecture-focused long run state.")
    parser.add_argument("--score-only", action="store_true", help="Only print the score.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to inspect.")
    args = parser.parse_args()

    payload = compute_v30_longrun(Path(args.repo_root))
    if args.score_only:
        print(payload["score"])
        return
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"stage2_v30_longrun_score={payload['score']}/{payload['total']}")
    for check in payload["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['slug']}: {check['detail']}")


if __name__ == "__main__":
    main()
