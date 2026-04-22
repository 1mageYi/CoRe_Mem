from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v5_longrun import (
    compute_v5_longrun,
    publish_v5_context_selfsupervised,
    publish_v5_personamem_isolation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_docs(repo_root: Path) -> None:
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "docs" / "v5_plan.md").write_text(
        "Core-Residual Latent Substrate\npretrained encoder\ncore latent\nresidual latent\n"
        "fallback\nshortcut\nbenchmark leakage\nPersonaMem\ngold\nBGE\nE5\nContriever\n"
        "latent-only\ntext-only\nshuffled-latent\ncore-only\nresidual-only\nno-controller\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "current_status.md").write_text("`TD-046 / WS-032 / v5`\nCore\nresidual\nlatent\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-046 / WS-032 / v5`\nBGE\nE5\nContriever\nlatent-only\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "todo.md").write_text("`TD-046`\nCore-Residual\nlatent-only\nPersonaMem gold\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-046 / WS-032`\nv5\nCore-Residual\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-046`\n`WS-032`\nBGE\nContriever\n", encoding="utf-8")
    (repo_root / ".agent-os" / "change-decisions.md").write_text(
        "`CD-013`\nPersonaMem gold\nBGE\nContriever\nMiniMax-M2.7\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "architecture-milestones.md").write_text("`MS-013`\nCore-Residual\nanti-shortcut\n", encoding="utf-8")
    (repo_root / ".agent-os" / "acceptance-report.md").write_text("`EV-018`\n`WS-032 / TD-046`\ncontract lock\n", encoding="utf-8")


def test_v5_longrun_scores_contract_baseline(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_docs(repo_root)

    payload = compute_v5_longrun(repo_root)

    assert payload["metric"] == "stage2_v5_longrun_score"
    assert payload["score"] == 13
    assert payload["total"] == 52


def test_publish_v5_personamem_isolation_groups_by_persona_and_context(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data" / "personamem"
    data_root.mkdir(parents=True)
    questions = data_root / "questions_32k.csv"
    questions.write_text(
        "\n".join(
            [
                "persona_id,question_id,question_type,topic,user_question_or_message,correct_answer,all_options,shared_context_id,end_index_in_shared_context",
                'p1,q1,recall,t,m,(a),"[""(a) x"", ""(b) y""]",c1,10',
                'p1,q2,recall,t,m,(b),"[""(a) x"", ""(b) y""]",c2,20',
                'p2,q3,recall,t,m,(a),"[""(a) x"", ""(b) y""]",c3,30',
                'p3,q4,recall,t,m,(b),"[""(a) x"", ""(b) y""]",c4,40',
                'p4,q5,recall,t,m,(a),"[""(a) x"", ""(b) y""]",c5,50',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contexts = data_root / "shared_contexts_32k.jsonl"
    contexts.write_text(
        "\n".join(
            json.dumps({"shared_context_id": context_id, "shared_context": []})
            for context_id in ("c1", "c2", "c3", "c4", "c5")
        )
        + "\n",
        encoding="utf-8",
    )

    payload = publish_v5_personamem_isolation(root=repo_root, questions_path=questions, contexts_path=contexts)

    assert payload["split_by_shared_context_id"] is True
    assert payload["split_by_persona"] is True
    assert payload["train_eval_contexts_disjoint"] is True
    assert payload["train_eval_personas_disjoint"] is True
    assert payload["no_gold_leakage"] is True
    assert payload["gold_used_for_memory_substrate"] is False
    assert payload["gold_policy"]["thin_answer_head_calibration"] == "train_split_only"
    assert (repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_personamem_isolation.json").exists()


def test_v5_longrun_counts_isolation_checks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_docs(repo_root)
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_personamem_isolation.json",
        {
            "split_by_shared_context_id": True,
            "split_by_persona": True,
            "no_gold_leakage": True,
            "gold_used_for_memory_substrate": False,
            "train_eval_contexts_disjoint": True,
        },
    )

    payload = compute_v5_longrun(repo_root)

    assert payload["score"] == 17


def test_publish_v5_context_selfsupervised_uses_raw_context_without_gold(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data" / "personamem"
    data_root.mkdir(parents=True)
    questions = data_root / "questions_32k.csv"
    questions.write_text(
        "\n".join(
            [
                "persona_id,question_id,question_type,topic,user_question_or_message,correct_answer,all_options,shared_context_id,end_index_in_shared_context",
                'p1,q1,recall,t,m,(a),"[""(a) x"", ""(b) y""]",c1,10',
                'p2,q2,recall,t,m,(b),"[""(a) x"", ""(b) y""]",c2,20',
                'p3,q3,recall,t,m,(a),"[""(a) x"", ""(b) y""]",c3,30',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contexts = data_root / "shared_contexts_32k.jsonl"
    contexts.write_text(
        "\n".join(
            json.dumps(
                {
                    "shared_context_id": context_id,
                    "shared_context": [
                        {"role": "system", "content": "persona"},
                        {"role": "user", "content": f"user turn {context_id}"},
                        {"role": "assistant", "content": f"assistant turn {context_id}"},
                    ],
                }
            )
            for context_id in ("c1", "c2", "c3")
        )
        + "\n",
        encoding="utf-8",
    )
    publish_v5_personamem_isolation(root=repo_root, questions_path=questions, contexts_path=contexts)

    payload = publish_v5_context_selfsupervised(root=repo_root, contexts_path=contexts, max_samples_per_context=2)

    assert payload["no_gold_answers"] is True
    assert payload["train_samples"] > 0
    assert payload["eval_samples"] > 0
    assert payload["stage2_32k_used_as_warmup"] is True
    assert payload["claims_large_scale_pretraining"] is False
    train_rows = (repo_root / payload["task_files"]["train"]).read_text(encoding="utf-8")
    assert "correct_answer" not in train_rows
    assert "all_options" not in train_rows


def test_v5_longrun_counts_context_selfsupervision_checks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_docs(repo_root)
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_personamem_isolation.json",
        {
            "split_by_shared_context_id": True,
            "split_by_persona": True,
            "no_gold_leakage": True,
            "gold_used_for_memory_substrate": False,
            "train_eval_contexts_disjoint": True,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_context_selfsupervised.json",
        {
            "no_gold_answers": True,
            "train_samples": 10,
            "eval_samples": 4,
            "stage2_32k_used_as_warmup": True,
            "claims_large_scale_pretraining": False,
        },
    )

    payload = compute_v5_longrun(repo_root)

    assert payload["score"] == 21
