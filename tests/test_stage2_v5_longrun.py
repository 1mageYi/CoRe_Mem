from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v5_longrun import (
    compute_v5_longrun,
    publish_v5_context_selfsupervised,
    publish_v5_core_residual_train,
    publish_v5_encoder_compare,
    publish_v5_latent_reader_eval,
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


def test_publish_v5_encoder_compare_records_three_backbones_without_gold(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    sample_root = artifact_root / "stage2_v5_context_selfsupervised"
    sample_root.mkdir(parents=True)
    rows = [
        {
            "sample_id": "s1",
            "input_context": [{"role": "user", "content": "I love music and technology."}],
            "target_text": "I remix electronic music.",
        },
        {
            "sample_id": "s2",
            "input_context": [{"role": "user", "content": "Libraries feel relaxing."}],
            "target_text": "I volunteer at the local library.",
        },
        {
            "sample_id": "s3",
            "input_context": [{"role": "user", "content": "Food markets are too crowded."}],
            "target_text": "I prefer quiet grocery stores.",
        },
    ]
    (sample_root / "val.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (sample_root / "eval.jsonl").write_text("", encoding="utf-8")
    _write_json(
        artifact_root / "latest_stage2_v5_context_selfsupervised.json",
        {
            "task_files": {
                "val": str(sample_root / "val.jsonl"),
                "eval": str(sample_root / "eval.jsonl"),
            },
            "no_gold_answers": True,
        },
    )

    payload = publish_v5_encoder_compare(root=repo_root)

    assert len(payload["compared_backbones"]) == 3
    assert payload["selected_backbone"]
    assert payload["uses_ablation_metrics"] is True
    assert payload["not_selected_by_personamem_only"] is True
    assert payload["uses_personamem_gold"] is False
    assert payload["pretrained_weights_loaded"] is False
    assert (artifact_root / "latest_stage2_v5_encoder_compare.json").exists()


def test_v5_longrun_counts_encoder_compare_checks(tmp_path: Path) -> None:
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
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_encoder_compare.json",
        {
            "compared_backbones": ["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2", "facebook/contriever"],
            "selected_backbone": "intfloat/e5-base-v2",
            "uses_ablation_metrics": True,
            "not_selected_by_personamem_only": True,
        },
    )

    payload = compute_v5_longrun(repo_root)

    assert payload["score"] == 25


def test_publish_v5_core_residual_train_records_positive_controller_gain(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    sample_root = artifact_root / "stage2_v5_context_selfsupervised"
    sample_root.mkdir(parents=True)
    train_rows = []
    eval_rows = []
    source_rows = [
        ("system", "Current user persona: I usually love quiet libraries."),
        ("user", "I recently decided to stop attending crowded festivals."),
        ("user", "I love calm reading nights."),
        ("assistant", "That sounds like a new plan."),
        ("user", "Now I prefer smaller gatherings."),
        ("user", "I never liked deadline pressure."),
    ]
    for idx in range(80):
        role, text = source_rows[idx % len(source_rows)]
        train_rows.append({"sample_id": f"train-{idx}", "masked_role": role, "target_text": text, "task": "masked_turn_reconstruction"})
    for idx in range(36):
        role, text = source_rows[idx % len(source_rows)]
        eval_rows.append({"sample_id": f"eval-{idx}", "masked_role": role, "target_text": text, "task": "masked_turn_reconstruction"})
    for split, rows in {"train": train_rows, "val": eval_rows[:18], "eval": eval_rows[18:]}.items():
        (sample_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    _write_json(
        artifact_root / "latest_stage2_v5_context_selfsupervised.json",
        {
            "task_files": {
                "train": str(sample_root / "train.jsonl"),
                "val": str(sample_root / "val.jsonl"),
                "eval": str(sample_root / "eval.jsonl"),
            }
        },
    )

    payload = publish_v5_core_residual_train(root=repo_root, max_train_samples=80, max_eval_samples=36)

    train_payload = payload["core_residual_train"]
    controller_payload = payload["controller_ablation"]
    assert train_payload["trainable_core_residual"] is True
    assert train_payload["trainable_write_controller"] is True
    assert train_payload["positive_gain"] is True
    assert train_payload["train_seconds"] > 0
    assert controller_payload["learned_controller_beats_disabled"] is True
    assert len(controller_payload["covered_actions"]) >= 4
    assert (artifact_root / "latest_stage2_v5_core_residual_train.json").exists()
    assert (artifact_root / "latest_stage2_v5_controller_ablation.json").exists()


def test_v5_longrun_counts_core_residual_and_controller_checks(tmp_path: Path) -> None:
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
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_encoder_compare.json",
        {
            "compared_backbones": ["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2", "facebook/contriever"],
            "selected_backbone": "intfloat/e5-base-v2",
            "uses_ablation_metrics": True,
            "not_selected_by_personamem_only": True,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_core_residual_train.json",
        {
            "trainable_core_residual": True,
            "trainable_write_controller": True,
            "positive_gain": True,
            "train_seconds": 0.1,
            "device": "cpu",
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v5_controller_ablation.json",
        {
            "learned_controller_beats_disabled": True,
            "covered_actions": ["new", "merge", "overwrite", "stale"],
        },
    )

    payload = compute_v5_longrun(repo_root)

    assert payload["score"] == 32


def test_publish_v5_latent_reader_eval_records_anti_shortcut_metrics(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    sample_root = artifact_root / "stage2_v5_context_selfsupervised"
    sample_root.mkdir(parents=True)
    rows = [
        {
            "sample_id": "s1",
            "input_context": [{"role": "user", "content": "quiet library reading"}],
            "target_text": "quiet library reading",
        },
        {
            "sample_id": "s2",
            "input_context": [{"role": "user", "content": "crowded festivals chaotic"}],
            "target_text": "crowded festivals chaotic",
        },
        {
            "sample_id": "s3",
            "input_context": [{"role": "user", "content": "Pacific electronic music"}],
            "target_text": "Pacific electronic music",
        },
        {
            "sample_id": "s4",
            "input_context": [{"role": "user", "content": "deadline pressure reading"}],
            "target_text": "deadline pressure reading",
        },
    ]
    (sample_root / "val.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (sample_root / "eval.jsonl").write_text("", encoding="utf-8")
    _write_json(
        artifact_root / "latest_stage2_v5_context_selfsupervised.json",
        {"task_files": {"val": str(sample_root / "val.jsonl"), "eval": str(sample_root / "eval.jsonl")}},
    )
    _write_json(
        artifact_root / "latest_stage2_v5_encoder_compare.json",
        {"selected_backbone": "BAAI/bge-base-en-v1.5"},
    )

    payload = publish_v5_latent_reader_eval(root=repo_root, max_eval_samples=4)

    assert payload["latent_reader"]["query_conditioned_reader"] is True
    assert payload["latent_reader"]["latent_only_above_random"] is True
    assert payload["latent_reader"]["shuffled_latent_drops"] is True
    assert payload["text_ablation"]["text_dropout_enabled"] is True
    assert payload["text_ablation"]["full_beats_text_only"] is True
    assert (artifact_root / "latest_stage2_v5_latent_reader_eval.json").exists()
    assert (artifact_root / "latest_stage2_v5_text_ablation.json").exists()


def test_v5_longrun_counts_latent_reader_and_text_ablation_checks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_docs(repo_root)
    for name, payload in {
        "latest_stage2_v5_personamem_isolation.json": {
            "split_by_shared_context_id": True,
            "split_by_persona": True,
            "no_gold_leakage": True,
            "gold_used_for_memory_substrate": False,
            "train_eval_contexts_disjoint": True,
        },
        "latest_stage2_v5_context_selfsupervised.json": {
            "no_gold_answers": True,
            "train_samples": 10,
            "eval_samples": 4,
            "stage2_32k_used_as_warmup": True,
            "claims_large_scale_pretraining": False,
        },
        "latest_stage2_v5_encoder_compare.json": {
            "compared_backbones": ["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2", "facebook/contriever"],
            "selected_backbone": "intfloat/e5-base-v2",
            "uses_ablation_metrics": True,
            "not_selected_by_personamem_only": True,
        },
        "latest_stage2_v5_core_residual_train.json": {
            "trainable_core_residual": True,
            "trainable_write_controller": True,
            "positive_gain": True,
            "train_seconds": 0.1,
            "device": "cpu",
        },
        "latest_stage2_v5_controller_ablation.json": {
            "learned_controller_beats_disabled": True,
            "covered_actions": ["new", "merge", "overwrite", "stale"],
        },
        "latest_stage2_v5_latent_reader_eval.json": {
            "latent_only_above_random": True,
            "shuffled_latent_drops": True,
            "query_conditioned_reader": True,
        },
        "latest_stage2_v5_text_ablation.json": {
            "full_beats_text_only": True,
            "text_dropout_enabled": True,
        },
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v5_longrun(repo_root)

    assert payload["score"] == 39
