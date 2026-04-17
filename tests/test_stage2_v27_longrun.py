from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v27_longrun import compute_v27_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v27_longrun_verifier_passes_when_teacher_first_32k_state_exists(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-038`\n`v2.7`\n32k\nteacher-first\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`TD-038 / WS-024`\n`v2.7`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text(
        "`TD-038` `[doing]`\n32k\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "v27_plan.md").write_text(
        "冻结边界\n不改 `core / residual`\n不做任何 `fallback`\n不做任何 `shortcut`\nbenchmark-specific heuristic\n32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n只有在 32k internal test work well 后\n暂不进入 full-data 训练\n",
        encoding="utf-8",
    )

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json", {"positive_gain": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_belief_gain.json", {"positive_gain": True})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json",
        {"provider_exact_match": 11, "local_exact_match": 11},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json",
        {"provider_exact_match": 44, "local_exact_match": 33},
    )

    for name in [
        "latest_stage2_v27_32k_split.json",
        "latest_stage2_v27_32k_manifest.json",
        "latest_stage2_v27_32k_audit.json",
        "latest_stage2_v27_teacher_observation.json",
        "latest_stage2_v27_teacher_slot_assignment.json",
        "latest_stage2_v27_teacher_belief.json",
        "latest_stage2_v27_train.json",
        "latest_stage2_v27_eval.json",
        "latest_stage2_v27_training_timing.json",
        "latest_stage2_v27_internal_test.json",
        "latest_stage2_v27_holdout_summary.json",
    ]:
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, {"ok": True})

    payload = compute_v27_longrun(repo_root)
    assert payload["score"] == payload["total"] == 26
