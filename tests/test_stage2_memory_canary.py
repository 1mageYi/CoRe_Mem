from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_memory_canary_writes_honest_blocked_artifact_without_provider(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    result = _run(
        "scripts/run_stage2_memory_canary.py",
        "--config",
        "configs/minimax_m27.yaml",
        "--benchmark",
        "personamem",
        "--output-root",
        str(output_root),
        "--limit",
        "1",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["summary_path"]).exists()
    assert Path(payload["run_dir"]).exists()
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["status"] in {"completed", "blocked_provider_not_configured"}
