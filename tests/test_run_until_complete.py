from pathlib import Path
from types import SimpleNamespace
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scripts.run_until_complete as run_until_complete


def test_supervisor_retries_until_predictions_reach_expected(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "demo_personamem"
    run_dir.mkdir(parents=True)
    predictions_path = run_dir / "predictions.jsonl"
    launches = {"count": 0}
    sleeps: list[float] = []

    def fake_run(cmd, cwd, check):
        launches["count"] += 1
        if launches["count"] == 1:
            predictions_path.write_text(
                json.dumps({"sample_id": "q1"}) + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=1)
        predictions_path.write_text(
            "\n".join([json.dumps({"sample_id": "q1"}), json.dumps({"sample_id": "q2"})]) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_until_complete.subprocess, "run", fake_run)
    monkeypatch.setattr(run_until_complete.time, "sleep", lambda seconds: sleeps.append(seconds))

    args = SimpleNamespace(
        benchmark="personamem",
        config="configs/gemini_flash.yaml",
        run_dir=str(run_dir),
        expected_count=2,
        limit=2,
        cooldown_seconds=7.0,
        max_launches=5,
        max_idle_launches=3,
        chunk_size=None,
        questions_csv="data/personamem/questions_32k.csv",
        contexts_jsonl="data/personamem/shared_contexts_32k.jsonl",
        dataset_json=None,
    )

    result = run_until_complete.supervise_run(args)

    assert result == 0
    assert launches["count"] == 2
    assert sleeps == [7.0]


def test_supervisor_stops_after_max_idle_launches(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "demo_longmemeval"
    run_dir.mkdir(parents=True)
    sleeps: list[float] = []

    monkeypatch.setattr(
        run_until_complete.subprocess,
        "run",
        lambda cmd, cwd, check: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(run_until_complete.time, "sleep", lambda seconds: sleeps.append(seconds))

    args = SimpleNamespace(
        benchmark="longmemeval",
        config="configs/gemini_flash.yaml",
        run_dir=str(run_dir),
        expected_count=1,
        limit=1,
        cooldown_seconds=3.0,
        max_launches=5,
        max_idle_launches=2,
        chunk_size=None,
        questions_csv=None,
        contexts_jsonl=None,
        dataset_json="data/longmemeval/longmemeval_s_cleaned.json",
    )

    result = run_until_complete.supervise_run(args)

    assert result == 1
    assert sleeps == [3.0]


def test_build_runner_command_uses_chunk_limited_prefix():
    args = SimpleNamespace(
        benchmark="personamem",
        config="configs/gemini_flash.yaml",
        run_dir="outputs/runs/demo_personamem",
        limit=589,
        chunk_size=5,
        questions_csv="data/personamem/questions_32k.csv",
        contexts_jsonl="data/personamem/shared_contexts_32k.jsonl",
        dataset_json=None,
    )

    cmd = run_until_complete.build_runner_command(args, completed_before=22)

    assert "--limit" in cmd
    assert cmd[cmd.index("--limit") + 1] == "27"
