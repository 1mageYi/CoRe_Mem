"""Mechanical verifier for the learned-memory-first v2.1 pivot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contains_all(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def _contains_any(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner_avoids_shortcuts(root: Path) -> bool:
    runner = root / "scripts" / "run_stage2_memory_canary.py"
    if not runner.exists():
        return False
    text = runner.read_text(encoding="utf-8")
    banned = [
        "Latent matcher candidate:",
        "_finalize_personamem_provider_prediction",
        "_best_personamem_option_label(",
        "blank-provider fallback",
    ]
    return not any(pattern in text for pattern in banned)


def _best_trained_eval_token_f1(root: Path) -> float:
    best = 0.0
    for path in sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json")):
        payload = _read_json(path)
        metrics = ((payload.get("trained_eval") or {}).get("metrics") or {})
        best = max(best, float(metrics.get("token_f1", 0.0)))
    return best


def _artifact_exists(path: Path) -> bool:
    return path.exists()


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in patterns):
            return True
    return False


def compute_learned_memory_score(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    related_work = root / "docs" / "learned_memory_related_work.md"
    learned_gain = root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json"

    checks = {
        "current_status_tracks_learned_memory_first": _contains_all(
            current_status,
            [
                "learned-memory-first",
                "更好的 latent",
                "stage2_v21_learned_memory_score",
            ],
        ),
        "implementation_plan_tracks_stage_n": _contains_all(
            implementation_plan,
            [
                "阶段 N：V2.1 Learned-Memory-First Pivot",
                "checkpoint-backed learned memory path",
                "online-aligned learned path",
            ],
        ),
        "agentos_tracks_td029_ws015": _contains_all(
            project_index,
            [
                "Top next action: `TD-029`",
                "Active workstreams: `WS-015`",
            ],
        ) and _contains_any(todo, ["`TD-029` `[doing]`"]),
        "related_work_note_exists": _artifact_exists(related_work),
        "related_work_mentions_primary_inspirations": _contains_all(
            related_work,
            [
                "End-To-End Memory Networks",
                "Memorizing Transformers",
                "RETRO",
                "LongMem",
                "Slot Attention",
            ],
        ),
        "benchmark_runner_avoids_shortcuts": _runner_avoids_shortcuts(root),
        "historical_learned_gain_artifact_exists": _artifact_exists(learned_gain),
        "non_tiny_trained_eval_token_f1_ge_0_14": _best_trained_eval_token_f1(root) >= 0.14,
        "online_system_supports_learned_memory_toggle": _code_contains_any(
            root,
            [
                "src/core_mem/v2/system.py",
                "scripts/run_stage2_memory_canary.py",
                "configs/stage2_train.yaml",
                "configs/stage2_train_tiny.yaml",
            ],
            [
                "learned_memory",
                "memory_mode",
                "use_learned_memory",
                "online_learned",
            ],
        ),
        "training_supports_online_aligned_learned_variant": _code_contains_any(
            root,
            [
                "src/core_mem/v2/training.py",
                "scripts/train_stage2.py",
                "configs/stage2_train.yaml",
            ],
            [
                "online_aligned",
                "learned_retrieval",
                "belief_ranking",
                "memory_write",
            ],
        ),
        "current_head_personamem_learned_canary_exists": _artifact_exists(
            root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json"
        ),
        "current_head_longmemeval_learned_canary_exists": _artifact_exists(
            root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json"
        ),
    }

    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "best_trained_eval_token_f1": _best_trained_eval_token_f1(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_learned_memory_score(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
