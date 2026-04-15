"""Mechanical verifier for the long-run complete-v2 milestone set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _contains_all(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_stage2_canary(root: Path, benchmark: str) -> Path | None:
    candidates = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if payload.get("benchmark") == benchmark and payload.get("status") == "completed":
            return path
    return None


def _latest_analysis(root: Path, benchmark: str) -> Path | None:
    latest = root / "outputs_v2" / "artifacts" / f"latest_{benchmark}_stage2_canary_analysis.json"
    return latest if latest.exists() else None


def _run_score(script: Path) -> int:
    if not script.exists():
        return 0
    result = subprocess.run(
        ["python3", str(script), "--score-only"],
        cwd=script.parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _current_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else None


def _latest_non_tiny_train(root: Path) -> dict[str, Any] | None:
    candidates = sorted((root / "outputs_v2" / "runs").glob("*_stage2_train_exec/execution_summary.json"))
    for path in reversed(candidates):
        summary = _read_json(path)
        config_snapshot = path.parent / "config_snapshot.yaml"
        if not config_snapshot.exists():
            continue
        config_text = config_snapshot.read_text(encoding="utf-8")
        if "__tiny_debug_seq2seq__" in config_text:
            continue
        checkpoint_dir = Path(str(summary.get("checkpoint_dir", "")))
        if checkpoint_dir and not checkpoint_dir.is_absolute():
            checkpoint_dir = (root / checkpoint_dir).resolve()
        return {
            "summary_path": str(path),
            "checkpoint_dir": str(checkpoint_dir),
            "config_snapshot_path": str(config_snapshot),
            "summary": summary,
        }
    return None


def _has_non_tiny_checkpoint(train_payload: dict[str, Any] | None) -> bool:
    if not train_payload:
        return False
    checkpoint_dir = Path(train_payload["checkpoint_dir"])
    if not checkpoint_dir.exists():
        return False
    if (checkpoint_dir / "tiny_model.pt").exists():
        return False
    return any(path.is_file() for path in checkpoint_dir.rglob("*"))


def _positive_trained_eval(root: Path) -> dict[str, Any] | None:
    candidates = sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        trained = payload.get("trained_eval") or {}
        metrics = trained.get("metrics") or {}
        exact = float(metrics.get("exact_match", 0.0))
        token_f1 = float(metrics.get("token_f1", 0.0))
        if exact > 0.0 or token_f1 > 0.0:
            return {
                "path": str(path),
                "exact_match": exact,
                "token_f1": token_f1,
            }
    return None


def _canary_completed(root: Path, benchmark: str) -> bool:
    summary_path = _latest_stage2_canary(root, benchmark)
    if summary_path is None:
        return False
    payload = _read_json(summary_path)
    current_head = _current_head(root)
    commit_matches = True
    if current_head is not None:
        commit_matches = str(payload.get("commit_hash", "")) == current_head
    return (
        payload.get("status") == "completed"
        and int(payload.get("sample_count", 0)) >= 64
        and int(payload.get("live_predictions_completed", 0)) >= 64
        and bool(payload.get("provider_configured", False))
        and commit_matches
    )


def _runner_avoids_shortcuts(root: Path) -> bool:
    runner = root / "scripts" / "run_stage2_memory_canary.py"
    if not runner.exists():
        return False
    text = runner.read_text(encoding="utf-8")
    banned = [
        "Latent matcher candidate:",
        "_finalize_personamem_provider_prediction",
        "_best_personamem_option_label(",
    ]
    return not any(pattern in text for pattern in banned)


def compute_v2_completion(root: Path) -> dict[str, Any]:
    current_status_path = root / "docs" / "current_status.md"
    implementation_plan_path = root / "docs" / "implementation_plan.md"
    project_index_path = root / ".agent-os" / "project-index.md"
    todo_path = root / ".agent-os" / "todo.md"

    latent_status = _run_score(root / "scripts" / "verify_stage2_latent_status.py")
    latent_core = _run_score(root / "scripts" / "verify_stage2_latent_core_quality.py")
    non_tiny_train = _latest_non_tiny_train(root)
    trained_eval = _positive_trained_eval(root)

    checks = {
        "current_status_tracks_complete_v2_goal": _contains_all(
            current_status_path,
            [
                "完整的 `v2`",
                "不要做任何偷懒兜底 fallback",
            ],
        ),
        "implementation_plan_tracks_stage_l_milestones": _contains_all(
            implementation_plan_path,
            [
                "阶段 L：完整 V2 长跑里程碑",
                "不允许 benchmark-specific heuristic / fallback 成为 retained 收益",
            ],
        ),
        "project_index_tracks_td027": _contains(project_index_path, "`TD-027` `[doing]`"),
        "todo_tracks_td027": _contains(todo_path, "`TD-027` `[doing]`"),
        "latent_readiness_9_of_9": latent_status >= 9,
        "latent_core_quality_10_of_10": latent_core >= 10,
        "personamem_live_canary_completed": _canary_completed(root, "personamem"),
        "personamem_analysis_exists": _latest_analysis(root, "personamem") is not None,
        "longmemeval_live_canary_completed": _canary_completed(root, "longmemeval"),
        "longmemeval_analysis_exists": _latest_analysis(root, "longmemeval") is not None,
        "non_tiny_train_artifact_exists": non_tiny_train is not None,
        "non_tiny_checkpoint_exists": _has_non_tiny_checkpoint(non_tiny_train),
        "trained_eval_positive": trained_eval is not None,
        "benchmark_runner_avoids_shortcuts": _runner_avoids_shortcuts(root),
    }

    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "latent_readiness_score": latent_status,
        "latent_core_quality_score": latent_core,
        "latest_non_tiny_train": non_tiny_train,
        "positive_trained_eval": trained_eval,
        "latest_personamem_canary": str(_latest_stage2_canary(root, "personamem")) if _latest_stage2_canary(root, "personamem") else None,
        "latest_longmemeval_canary": str(_latest_stage2_canary(root, "longmemeval")) if _latest_stage2_canary(root, "longmemeval") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_v2_completion(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
