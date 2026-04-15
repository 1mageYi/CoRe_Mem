from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v21_semantic_model import compute_semantic_model_score


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_semantic_model_verifier_tracks_semantic_first_state(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-031`\n语义优先\n格式外部约束\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 P：Semantic-First Learned Decoder\n语义字段\n格式约束外置\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-031`\nActive workstreams: `WS-017`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text(
        "`TD-031` `[doing]`\nsemantic-first\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "v21_longrun_plan.md").write_text(
        "语义正确优先于 raw JSON 表面格式完全匹配\n阶段 F：语义优先训练重构\n格式外部约束处理\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "def main():\n    return 'clean runner'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "memory_mode='learned_memory'\nuse_learned_memory=True\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "train_stage2.py").write_text(
        "online_aligned=True\n",
        encoding="utf-8",
    )
    (repo_root / "configs" / "stage2_train.yaml").write_text(
        "online_aligned: true\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "\n".join(
            [
                "online_aligned = True",
                "composition_to_belief = True",
                "retrieval_alignment = True",
                "lifecycle_prediction = True",
                "json_validity_rate = 0.0",
                "field_f1 = 0.0",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        repo_root / "outputs_v2" / "evals_local" / "20260415T000000Z_stage2_local_eval.json",
        {
            "trained_eval": {
                "metrics": {"token_f1": 0.5},
                "per_task": {
                    "lifecycle_prediction": {"token_f1": 0.56},
                    "composition_to_belief": {"token_f1": 0.61},
                    "retrieval_alignment": {"token_f1": 0.06},
                },
            }
        },
    )
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json", {})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json", {})

    payload = compute_semantic_model_score(repo_root)
    assert payload["score"] == payload["total"] == 17
    assert payload["best_trained_eval_token_f1"] == 0.5
