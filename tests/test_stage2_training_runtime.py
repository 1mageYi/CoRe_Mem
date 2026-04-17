from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.training import build_training_examples
from core_mem.v2.training import _balanced_cap_examples
from core_mem.v2.training import compact_slot_payload


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_training_examples_reads_all_stage2_tasks(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    examples = build_training_examples(manifest)
    assert len(examples) == 4
    assert {example.task_name for example in examples} == {
        "slot_autoencoding",
        "retrieval_alignment",
        "lifecycle_prediction",
        "composition_to_belief",
    }


def test_lifecycle_training_examples_use_explicit_slot_assignment_schema(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    examples = build_training_examples(manifest)
    lifecycle_example = next(example for example in examples if example.task_name == "lifecycle_prediction")
    assert "Choose exactly one target_action from [merge, overwrite, new, ignore]." in lifecycle_example.input_text
    assert 'Return JSON only with the schema {"target_action":"new","target_flags":{"promote":false,"stale_old":false}}.' in lifecycle_example.input_text


def test_train_stage2_execute_train_uses_tiny_runtime(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
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
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["num_steps"] == 1
    assert payload["optimizer_steps"] == 1
    assert payload["num_examples"] == 4
    assert payload["online_aligned"] is True
    assert Path(payload["metrics_path"]).exists()
    assert Path(payload["checkpoint_dir"]).exists()
    assert payload["wall_clock_seconds"] >= 0.0
    assert payload["examples_per_second"] is not None
    assert payload["steps_per_second"] is not None


def test_balanced_cap_examples_spreads_budget_across_tasks(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    examples = build_training_examples(manifest)
    capped = _balanced_cap_examples(examples, max_examples=4)
    assert len(capped) == 4
    assert {example.task_name for example in capped} == {
        "slot_autoencoding",
        "retrieval_alignment",
        "lifecycle_prediction",
        "composition_to_belief",
    }


def test_compact_slot_payload_removes_dense_latent_fields():
    payload = compact_slot_payload(
        {
            "slot_id": "slot-1",
            "bank": "residual",
            "entity": "user",
            "relation": "drink_preference",
            "canonical_gloss": "drink_preference=matcha",
            "confidence": 0.9,
            "active_flag": True,
            "revision_count": 0,
            "soft_role_scores": {"preference": 0.8, "constraint": 0.1},
            "latent_tokens": [[0.1, 0.2]],
            "retrieval_key": [0.1, 0.2],
        }
    )
    assert payload["dominant_role"] == "preference"
    assert "latent_tokens" not in payload
    assert "retrieval_key" not in payload


def test_train_stage2_respects_gradient_accumulation(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train_tiny.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--execute-train",
        "--max-steps",
        "3",
        "--max-train-examples",
        "6",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["num_steps"] == 3
    assert payload["optimizer_steps"] == 3
    assert payload["wall_clock_seconds"] >= 0.0


def test_stage2_train_plan_emits_direct_train_launcher(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train.yaml",
        "--prepared-manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    launch_script = Path(payload["run_dir"]) / "launch_stage2_training.sh"
    assert launch_script.exists()
    content = launch_script.read_text(encoding="utf-8")
    assert "--execute-train" in content


def test_train_stage2_can_register_experiment_outputs(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    manifest = Path(json.loads(prepare.stdout)["prepared_manifest"])

    result = _run(
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
        "--experiment-id",
        "mainline",
        "--register-experiment",
        "--max-eval-examples",
        "4",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["local_eval_path"]).exists()
    assert Path(payload["experiment_index_path"]).exists()
    index_payload = json.loads(Path(payload["experiment_index_path"]).read_text(encoding="utf-8"))
    assert index_payload["experiments"]["mainline"]["completed"] is True
