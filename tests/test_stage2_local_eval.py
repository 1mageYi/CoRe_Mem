from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core_mem.v2.eval_local import evaluate_local, module_inventory_payload
from eval_stage2_local import (
    _publish_v33_belief_graph_eval_artifact,
    _publish_v33_learned_write_eval_artifact,
    _publish_v33_temporal_slot_eval_artifact,
)
from core_mem.v2.training import _repair_belief_payload_from_input_context, _repair_lifecycle_payload_from_input_context


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_evaluate_local_returns_module_and_budget_sections(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    payload = evaluate_local(manifest, output_root, top_k=2, budgets=[1, 2])
    summary = payload["summary"]

    assert "retrieval_family" in summary["families"]
    assert "belief_decoder" in summary["modules"]
    assert "1" in summary["budget_sweep"]
    assert "2" in summary["budget_sweep"]
    assert 0.0 <= summary["metrics"]["joint_belief_accuracy"] <= 1.0
    assert 0.0 <= summary["metrics"]["answer_exact_match"] <= 1.0
    assert summary["metrics"]["recall_at_k"] >= 0.0


def test_eval_script_writes_json_and_csv_outputs(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--top-k",
        "2",
        "--budget",
        "1",
        "--budget",
        "2",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["result_path"]).exists()
    assert Path(payload["summary_table_path"]).exists()
    assert Path(payload["budget_table_path"]).exists()


def test_eval_script_can_include_checkpoint_metrics(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "trained_eval" in payload
    assert 0.0 <= payload["trained_eval"]["metrics"]["exact_match"] <= 1.0
    assert 0.0 <= payload["trained_eval"]["metrics"]["semantic_validity_rate"] <= 1.0
    assert 0.0 <= payload["trained_eval"]["metrics"]["field_f1"] <= 1.0


def test_train_script_can_publish_semantic_full_train_artifact(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--publish-semantic-full-train",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    payload = json.loads(train.stdout)
    artifact_path = Path(payload["semantic_full_train_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["artifact_type"] == "stage2_semantic_full_train"
    assert artifact_payload["num_examples"] == 4


def test_train_script_can_publish_v24_train_artifact(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--publish-v24-train",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    payload = json.loads(train.stdout)
    artifact_path = Path(payload["v24_train_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["artifact_type"] == "stage2_v24_train"
    assert artifact_payload["slot_assignment_task"] == "lifecycle_prediction"
    assert artifact_payload["num_examples"] == 4


def test_eval_script_can_publish_semantic_full_eval_artifact(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-semantic-full-eval",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    artifact_path = Path(payload["semantic_full_eval_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["artifact_type"] == "stage2_semantic_full_local_eval"
    assert "trained_eval" in artifact_payload


def test_eval_script_can_publish_slot_assignment_eval_artifact(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--publish-slot-assignment-train",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)
    assert Path(train_payload["slot_assignment_train_artifact"]).exists()

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-slot-assignment-eval",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    artifact_path = Path(payload["slot_assignment_eval_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["artifact_type"] == "stage2_slot_assignment_eval"
    assert artifact_payload["slot_assignment_task"] == "lifecycle_prediction"
    assert artifact_payload["slot_assignment_count"] >= 0


def test_eval_script_can_publish_v33_eval_artifacts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    manifest = output_root / "artifacts" / "stage2_prepared_samples_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"prepared": True}), encoding="utf-8")
    payload = {
        "result_path": str(output_root / "evals_local" / "fake_eval.json"),
        "modules": {
            "lifecycle": {
                "update_action_accuracy": 1.0,
                "update_flag_accuracy": 1.0,
                "update_success": 1.0,
            },
            "retrieval_key_head": {
                "recall_at_k": 1.0,
                "ndcg_at_k": 0.75,
                "support_slot_recall": 1.0,
            },
            "belief_decoder": {
                "joint_belief_accuracy": 1.0,
                "slot_value_f1": 1.0,
                "propagation_accuracy": 1.0,
            },
        },
        "families": {
            "retrieval_family": {
                "recall_at_k": 1.0,
                "ndcg_at_k": 0.75,
                "support_slot_recall": 1.0,
            }
        },
    }
    write_artifact = Path(
        _publish_v33_learned_write_eval_artifact(
            prepared_manifest_path=manifest,
            output_root=output_root,
            payload=payload,
        )
    )
    temporal_artifact = Path(
        _publish_v33_temporal_slot_eval_artifact(
            prepared_manifest_path=manifest,
            output_root=output_root,
            payload=payload,
        )
    )
    belief_artifact = Path(
        _publish_v33_belief_graph_eval_artifact(
            prepared_manifest_path=manifest,
            output_root=output_root,
            payload=payload,
        )
    )
    assert write_artifact.exists()
    assert temporal_artifact.exists()
    assert belief_artifact.exists()

    write_payload = json.loads(write_artifact.read_text(encoding="utf-8"))
    temporal_payload = json.loads(temporal_artifact.read_text(encoding="utf-8"))
    belief_payload = json.loads(belief_artifact.read_text(encoding="utf-8"))
    assert write_payload["positive_gain"] is True
    assert temporal_payload["positive_gain"] is True
    assert belief_payload["positive_gain"] is True
    assert temporal_payload["latent_reader_type"] == "temporal_semantic_latent_reader"
    assert belief_payload["belief_head_type"] == "belief_graph_head"


def test_eval_script_can_publish_v24_eval_artifact(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-v24-eval",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    artifact_path = Path(payload["v24_eval_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["artifact_type"] == "stage2_v24_eval"
    assert "trained_eval" in artifact_payload
    assert artifact_payload["slot_assignment_metrics"]["slot_assignment_task"] == "lifecycle_prediction"
    assert artifact_payload["slot_assignment_metrics"]["token_f1"] >= 0.0


def test_eval_script_can_publish_v30_belief_and_gain_artifacts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    baseline_eval = output_root / "artifacts" / "v30_shared_baseline_eval.json"
    baseline_eval.parent.mkdir(parents=True, exist_ok=True)
    baseline_eval.write_text(
        json.dumps(
            {
                "trained_eval": {
                    "per_task": {
                        "lifecycle_prediction": {"token_f1": -0.1, "field_f1": -0.1, "exact_match": -0.1},
                        "composition_to_belief": {"token_f1": -0.1, "field_f1": -0.1, "exact_match": -0.1},
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_v30_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_v30_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-v30-belief-decoder-eval",
        "--publish-v30-write-gain",
        "--publish-v30-belief-gain",
        "--v30-baseline-eval",
        str(baseline_eval),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    for key, expected_type in (
        ("v30_belief_decoder_eval_artifact", "stage2_v30_belief_decoder_eval"),
        ("v30_write_gain_artifact", "stage2_v30_write_gain"),
        ("v30_belief_gain_artifact", "stage2_v30_belief_gain"),
    ):
        artifact_path = Path(payload[key])
        assert artifact_path.exists()
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact_payload["artifact_type"] == expected_type
        assert artifact_payload["positive_gain"] is True


def test_repair_belief_payload_from_input_context_can_fill_relation_value_and_support() -> None:
    input_text = "\n".join(
        [
            "task: composition_to_belief",
            'query: "What food does the user currently prefer?"',
            'memory_slots: [{"active_flag": true, "bank": "core", "canonical_gloss": "food_preference=burger", "confidence": 0.9, "relation": "food_preference", "slot_id": "slot_food_current"}]',
        ]
    )
    repaired = _repair_belief_payload_from_input_context(
        input_text,
        '"food_preference" is the current preference of the user.',
        None,
    )
    assert repaired == {
        "belief_items": [
            {
                "relation": "food_preference",
                "value": "burger",
                "support_slot_ids": ["slot_food_current"],
            }
        ]
    }


def test_repair_lifecycle_payload_from_input_context_can_recover_overwrite() -> None:
    input_text = "\n".join(
        [
            "task: lifecycle_prediction",
            'memory_context: [{"active_flag": true, "bank": "core", "canonical_gloss": "temporal_fact=half past 7 in the evening", "confidence": 0.8, "relation": "temporal_fact", "slot_id": "slot_old_time"}]',
            'new_observation: {"canonical_gloss": "temporal_fact=7:30 pm", "relation": "temporal_fact", "status_hint": "active"}',
        ]
    )
    repaired = _repair_lifecycle_payload_from_input_context(
        input_text,
        'temporal_fact=half past 7 in the evening"',
        None,
    )
    assert repaired == {
        "target_action": "overwrite",
        "target_flags": {
            "promote": True,
            "stale_old": True,
        },
    }


def test_eval_script_can_publish_v31_belief_artifacts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    baseline_eval = output_root / "artifacts" / "v31_belief_baseline_eval.json"
    baseline_eval.parent.mkdir(parents=True, exist_ok=True)
    baseline_eval.write_text(
        json.dumps(
            {
                "current_composition_to_belief": {
                    "token_f1": -0.1,
                    "field_f1": -0.1,
                    "exact_match": -0.1,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_v30_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_v30_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-v31-belief-mainline-eval",
        "--publish-v31-belief-holdout-compare",
        "--v31-belief-baseline-eval",
        str(baseline_eval),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    for key, expected_type in (
        ("v31_belief_mainline_eval_artifact", "stage2_v31_belief_mainline_eval"),
        ("v31_belief_holdout_compare_artifact", "stage2_v31_belief_holdout_compare"),
    ):
        artifact_path = Path(payload[key])
        assert artifact_path.exists()
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact_payload["artifact_type"] == expected_type
    compare_payload = json.loads(Path(payload["v31_belief_holdout_compare_artifact"]).read_text(encoding="utf-8"))
    assert compare_payload["positive_gain"] is True


def test_eval_script_can_publish_v31_write_artifacts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    baseline_eval = output_root / "artifacts" / "v31_write_baseline_eval.json"
    baseline_eval.parent.mkdir(parents=True, exist_ok=True)
    baseline_eval.write_text(
        json.dumps(
            {
                "current_lifecycle_prediction": {
                    "token_f1": -0.1,
                    "field_f1": -0.1,
                    "exact_match": -0.1,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_v30_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_v30_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-v31-write-mainline-eval",
        "--publish-v31-write-holdout-compare",
        "--v31-write-baseline-eval",
        str(baseline_eval),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    for key, expected_type in (
        ("v31_write_mainline_eval_artifact", "stage2_v31_write_mainline_eval"),
        ("v31_write_holdout_compare_artifact", "stage2_v31_write_holdout_compare"),
    ):
        artifact_path = Path(payload[key])
        assert artifact_path.exists()
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact_payload["artifact_type"] == expected_type
    compare_payload = json.loads(Path(payload["v31_write_holdout_compare_artifact"]).read_text(encoding="utf-8"))
    assert compare_payload["positive_gain"] is True


def test_eval_script_can_publish_v32_write_and_belief_artifacts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    baseline_eval = output_root / "artifacts" / "v32_belief_baseline_eval.json"
    baseline_eval.parent.mkdir(parents=True, exist_ok=True)
    baseline_eval.write_text(
        json.dumps(
            {
                "current_composition_to_belief": {
                    "token_f1": -0.1,
                    "field_f1": -0.1,
                    "exact_match": -0.1,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_v32_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "1",
        "--max-train-examples",
        "4",
        "--json",
    )
    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--checkpoint-dir",
        train_payload["checkpoint_dir"],
        "--train-config",
        "configs/stage2_train_v32_tiny.yaml",
        "--max-eval-examples",
        "4",
        "--publish-v32-write-head-eval",
        "--publish-v32-belief-decoder-eval",
        "--publish-v32-belief-holdout-compare",
        "--v32-belief-baseline-eval",
        str(baseline_eval),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    for key, expected_type in (
        ("v32_write_head_eval_artifact", "stage2_v32_write_head_eval"),
        ("v32_belief_decoder_eval_artifact", "stage2_v32_belief_decoder_eval"),
        ("v32_belief_holdout_compare_artifact", "stage2_v32_belief_holdout_compare"),
    ):
        artifact_path = Path(payload[key])
        assert artifact_path.exists()
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact_payload["artifact_type"] == expected_type
    compare_payload = json.loads(Path(payload["v32_belief_holdout_compare_artifact"]).read_text(encoding="utf-8"))
    assert compare_payload["positive_gain"] is True


def test_eval_script_accepts_experiment_variant(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--experiment-id",
        "ablation_without_synthetic_pool",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["variant"]["disabled_pools"] == ["minimal_synthetic"]


def test_module_inventory_payload_is_stable():
    names = {item["name"] for item in module_inventory_payload()}
    assert names == {
        "parser",
        "slot_encoder",
        "query_encoder",
        "retrieval_key_head",
        "lifecycle",
        "consolidation",
        "core_residual_memory",
        "resampler",
        "belief_decoder",
        "answer_projection",
    }
