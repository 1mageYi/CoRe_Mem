from pathlib import Path
import subprocess
import sys
import json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_prepare_train_and_eval_scripts(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"

    prepare = _run("scripts/prepare_stage2_data.py", "--output-root", str(output_root), "--json")
    assert prepare.returncode == 0
    prepared_payload = json.loads(prepare.stdout)
    prepared_manifest = Path(prepared_payload["prepared_manifest"])
    assert prepared_manifest.exists()

    train = _run(
        "scripts/train_stage2.py",
        "--config",
        "configs/stage2_train.yaml",
        "--prepared-manifest",
        str(prepared_manifest),
        "--output-root",
        str(output_root),
        "--json",
    )
    assert train.returncode == 0
    train_payload = json.loads(train.stdout)
    training_plan = Path(train_payload["run_dir"]) / "training_plan.json"
    assert training_plan.exists()

    evaluate = _run(
        "scripts/eval_stage2_local.py",
        "--prepared-manifest",
        str(prepared_manifest),
        "--output-root",
        str(output_root),
        "--json",
    )
    assert evaluate.returncode == 0
    eval_payload = json.loads(evaluate.stdout)
    assert Path(eval_payload["result_path"]).exists()
    assert 0.0 <= eval_payload["metrics"]["joint_belief_accuracy"] <= 1.0
    assert Path(eval_payload["summary_table_path"]).exists()
    assert Path(eval_payload["budget_table_path"]).exists()


def test_stage2_canary_manifest_script(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    result = _run(
        "scripts/run_stage2_canary.py",
        "--output-root",
        str(output_root),
        "--personamem-size",
        "8",
        "--longmemeval-size",
        "8",
        "--json",
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert Path(payload["summary_path"]).exists()
    assert payload["personamem_count"] == 8
    assert payload["longmemeval_count"] == 8
