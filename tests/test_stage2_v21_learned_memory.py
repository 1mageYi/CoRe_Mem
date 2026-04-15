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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_stage2_v21_learned_memory_verifier(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "learned-memory-first\n更好的 latent\nstage2_v21_learned_memory_score\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 N：V2.1 Learned-Memory-First Pivot\ncheckpoint-backed learned memory path\nonline-aligned learned path\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-029`\nActive workstreams: `WS-015`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-029` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "learned_memory_related_work.md").write_text(
        "End-To-End Memory Networks\nMemorizing Transformers\nRETRO\nLongMem\nSlot Attention\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "memory_mode='learned_memory'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "use_learned_memory = True\n",
        encoding="utf-8",
    )
    (repo_root / "configs" / "stage2_train.yaml").write_text(
        "online_aligned: true\n",
        encoding="utf-8",
    )
    _write_json(
        repo_root / "outputs_v2" / "evals_local" / "demo_stage2_local_eval.json",
        {"trained_eval": {"metrics": {"token_f1": 0.2}}},
    )
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json", {"ok": True})

    result = _run("scripts/verify_stage2_v21_learned_memory.py", "--root", str(repo_root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == payload["total"] == 12


def test_stage2_v21_learned_memory_verifier_accepts_td030_runtime_truth(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-029` 的 learned-memory-first plumbing 已完成\n更好的 latent\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 N：V2.1 Learned-Memory-First Pivot\ncheckpoint-backed learned memory path\nonline-aligned learned path\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-030`\nActive workstreams: `WS-016`\n`WS-015` `[done]`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text(
        "`TD-030` `[doing]`\n`TD-029` `[done]`\n`TD-029` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "learned_memory_related_work.md").write_text(
        "End-To-End Memory Networks\nMemorizing Transformers\nRETRO\nLongMem\nSlot Attention\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "memory_mode='learned_memory'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "use_learned_memory = True\n",
        encoding="utf-8",
    )
    (repo_root / "configs" / "stage2_train.yaml").write_text(
        "online_aligned: true\n",
        encoding="utf-8",
    )
    _write_json(
        repo_root / "outputs_v2" / "evals_local" / "demo_stage2_local_eval.json",
        {"trained_eval": {"metrics": {"token_f1": 0.2}}},
    )
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json", {"ok": True})

    result = _run("scripts/verify_stage2_v21_learned_memory.py", "--root", str(repo_root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == payload["total"] == 12
