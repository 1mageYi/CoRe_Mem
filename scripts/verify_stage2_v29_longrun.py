"""Mechanical verifier for the v2.9 learned-core-path long run."""

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


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


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
        _contains_all(implementation_plan, ["`TD-040`", "`v2.9`", "write", "latent", "belief", "512"]),
        "implementation_plan should mention v2.9 write/latent/belief and expanded 512 holdout",
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
        _contains_all(v29_plan, ["24k train", "4k val", "4k test", "LongMemEval-S 128 -> 512", "PersonaMem 128 -> 512"]),
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
        int(v29_holdout.get("longmemeval_sample_count", 0)) >= 512 and int(v29_holdout.get("personamem_sample_count", 0)) >= 512,
        "v29 holdout summary should include expanded 512-sample LongMemEval-S and PersonaMem runs",
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
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

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
