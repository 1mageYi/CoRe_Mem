from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.eval_local import evaluate_local, module_inventory_payload


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
