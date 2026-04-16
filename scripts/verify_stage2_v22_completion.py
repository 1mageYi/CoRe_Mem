"""Mechanical verifier for the v2.2 full-data semantic-first plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _runner_avoids_shortcuts(root: Path) -> bool:
    text = _read_text(root / "scripts" / "run_stage2_memory_canary.py")
    banned = [
        "Latent matcher candidate:",
        "_finalize_personamem_provider_prediction",
        "_best_personamem_option_label(",
        "blank-provider fallback",
    ]
    return bool(text) and not any(pattern in text for pattern in banned)


def _online_system_avoids_symbolic_fallback(root: Path) -> bool:
    text = _read_text(root / "src" / "core_mem" / "v2" / "system.py")
    banned = [
        "symbolic_fallback",
        'return fallback, "symbolic"',
        'return fallback, "symbolic_fallback"',
    ]
    return bool(text) and not any(pattern in text for pattern in banned)


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        text = _read_text(root / rel)
        if text and any(pattern in text for pattern in patterns):
            return True
    return False


def _artifact_matches_head(path: Path, head: str | None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if head is not None and str(payload.get("commit_hash", "")) != head:
        return None
    return payload


def _full_train_payload(root: Path) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json", _current_head(root))


def _full_eval_payload(root: Path) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json", _current_head(root))


def _semantic_canary_payload(root: Path, benchmark: str) -> dict[str, Any] | None:
    name = {
        "personamem": "latest_personamem_stage2_semantic_canary.json",
        "longmemeval": "latest_longmemeval_stage2_semantic_canary.json",
    }[benchmark]
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / name, _current_head(root))


def _semantic_analysis_payload(root: Path) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_analysis.json", _current_head(root))


def _semantic_gain_payload(root: Path) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_online_gain.json", _current_head(root))


def compute_v22_completion(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v22_plan = root / "docs" / "v22_plan.md"

    full_train = _full_train_payload(root)
    full_eval = _full_eval_payload(root)
    personamem = _semantic_canary_payload(root, "personamem")
    longmemeval = _semantic_canary_payload(root, "longmemeval")
    analysis = _semantic_analysis_payload(root)
    gain = _semantic_gain_payload(root)

    trained_metrics = ((full_eval or {}).get("trained_eval") or {}).get("metrics") or {}
    trained_per_task = ((full_eval or {}).get("trained_eval") or {}).get("per_task") or {}

    checks = {
        "current_status_tracks_td032_v22": _contains_all(
            current_status,
            ["`TD-032`", "`v2.2`", "full-data", "LongMemEval-S"],
        ),
        "implementation_plan_tracks_stage_q": _contains_all(
            implementation_plan,
            ["阶段 Q：V2.2 Full-Data Semantic Latent", "full-data", "LongMemEval-S"],
        ),
        "agentos_tracks_td032_ws018": _contains_all(
            project_index,
            ["Top next action: `TD-032`", "Active workstreams: `WS-018`"],
        ) and _contains_all(todo, ["`TD-032` `[doing]`"]),
        "v22_plan_exists": v22_plan.exists(),
        "v22_plan_mentions_full_data_no_fallback": _contains_all(
            v22_plan,
            ["full-data", "LongMemEval-S", "no fallback", "no shortcut"],
        ),
        "benchmark_runner_avoids_shortcuts": _runner_avoids_shortcuts(root),
        "online_system_avoids_symbolic_fallback": _online_system_avoids_symbolic_fallback(root),
        "semantic_eval_metrics_exist": _code_contains_any(
            root,
            ["src/core_mem/v2/training.py", "src/core_mem/v2/semantic_outputs.py", "scripts/eval_stage2_local.py"],
            ["semantic_validity_rate", "field_accuracy", "field_f1", "json_validity_rate"],
        ),
        "training_not_locked_to_raw_json_exactness": not _code_contains_any(
            root,
            ["src/core_mem/v2/training.py", "src/core_mem/v2/system.py"],
            ["return only compact JSON that matches the target schema"],
        ),
        "current_head_full_data_train_exists": bool(full_train),
        "current_head_full_data_train_uses_gpu3": bool(full_train) and str(full_train.get("cuda_visible_devices", "")) == "3",
        "current_head_full_data_train_num_examples_ge_700": bool(full_train) and int(full_train.get("num_examples", 0)) >= 700,
        "current_head_full_data_eval_token_f1_ge_0_85": float(trained_metrics.get("token_f1", 0.0)) >= 0.85,
        "current_head_full_data_eval_retrieval_f1_ge_0_90": float(((trained_per_task.get("retrieval_alignment") or {}).get("token_f1", 0.0))) >= 0.90,
        "current_head_personamem_semantic_128_exists": bool(personamem) and int(personamem.get("sample_count", 0)) >= 128 and personamem.get("status") == "completed",
        "current_head_longmemeval_semantic_64_exists": bool(longmemeval) and int(longmemeval.get("sample_count", 0)) >= 64 and longmemeval.get("status") == "completed",
        "current_head_longmemeval_semantic_128_exists": bool(longmemeval) and int(longmemeval.get("sample_count", 0)) >= 128 and longmemeval.get("status") == "completed",
        "current_head_longmemeval_analysis_exists": bool(analysis),
        "current_head_semantic_online_gain_positive": bool(gain) and bool(gain.get("positive_gain")),
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v22_completion_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "full_train": full_train or {},
        "trained_eval_metrics": trained_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v22_completion(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
