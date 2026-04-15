"""Mechanical verifier for the learned-model-first v2.1 long run."""

from __future__ import annotations

import argparse
import json
import subprocess
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


def _online_system_avoids_symbolic_fallback(root: Path) -> bool:
    system = root / "src" / "core_mem" / "v2" / "system.py"
    if not system.exists():
        return False
    text = system.read_text(encoding="utf-8")
    banned = [
        "symbolic_fallback",
        'return fallback, "symbolic"',
        'return fallback, "symbolic_fallback"',
    ]
    return not any(pattern in text for pattern in banned)


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in patterns):
            return True
    return False


def _best_trained_eval_token_f1(root: Path) -> float:
    best = 0.0
    for path in sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json")):
        payload = _read_json(path)
        metrics = ((payload.get("trained_eval") or {}).get("metrics") or {})
        best = max(best, float(metrics.get("token_f1", 0.0)))
    return best


def _positive_learned_gain(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _read_json(path)
    deltas = [
        float(payload.get("delta_provider_exact_match", 0.0)),
        float(payload.get("delta_provider_label_prefix_match", 0.0)),
        float(payload.get("delta_local_exact_match", 0.0)),
    ]
    return bool(payload.get("positive_gain")) or any(delta > 0.0 for delta in deltas)


def _learned_alias_ok(root: Path, *, benchmark: str, min_samples: int) -> bool:
    alias_name = {
        "personamem": "latest_personamem_stage2_learned_canary.json",
        "longmemeval": "latest_longmemeval_stage2_learned_canary.json",
    }[benchmark]
    path = root / "outputs_v2" / "artifacts" / alias_name
    if not path.exists():
        return False
    payload = _read_json(path)
    benchmark_aliases = {
        "personamem": {"personamem"},
        "longmemeval": {"longmemeval", "longmemeval_s"},
    }
    if str(payload.get("benchmark", "")).strip().lower() not in benchmark_aliases[benchmark]:
        return False
    if int(payload.get("sample_count", 0)) < min_samples:
        return False
    if int(payload.get("live_predictions_completed", 0)) < min_samples:
        return False
    if payload.get("status") != "completed":
        return False
    if not bool(payload.get("provider_configured", False)):
        return False
    if str(payload.get("memory_mode", "")) != "learned_memory":
        return False
    if not bool(payload.get("use_learned_memory", False)):
        return False
    current_head = _current_head(root)
    if current_head is not None and str(payload.get("commit_hash", "")) != current_head:
        return False
    return True


def _layered_analysis_ok(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _read_json(path)
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return False
    required = {"parser", "retrieval", "belief", "projection", "provider"}
    return required.issubset(layers.keys())


def compute_v21_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    longrun_plan = root / "docs" / "v21_longrun_plan.md"
    learned_gain = root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json"
    longmemeval_layered = root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_layered_analysis.json"
    best_token_f1 = _best_trained_eval_token_f1(root)

    checks = {
        "current_status_tracks_td030_longrun": _contains_all(
            current_status,
            [
                "`TD-030`",
                "stage2_v21_longrun_score",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ],
        ),
        "implementation_plan_tracks_stage_o": _contains_all(
            implementation_plan,
            [
                "阶段 O：V2.1 Learned-Model Long Run",
                "更加注重 learned model",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ],
        ),
        "agentos_tracks_td030_ws016": _contains_all(
            project_index,
            [
                "Top next action: `TD-030`",
                "Active workstreams: `WS-016`",
            ],
        ) and _contains_any(todo, ["`TD-030` `[doing]`"]),
        "longrun_plan_exists": longrun_plan.exists(),
        "longrun_plan_mentions_phases_and_constraints": _contains_all(
            longrun_plan,
            [
                "阶段 A：把 learned path 从“存在”推进到“可测”",
                "阶段 B：让训练真正服务在线链路",
                "阶段 C：专项提升 LongMemEval-S",
                "阶段 D：扩大样本，验证 robustness",
                "阶段 E：收口成 v2.1",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ],
        ),
        "benchmark_runner_avoids_shortcuts": _runner_avoids_shortcuts(root),
        "online_system_avoids_symbolic_fallback": _online_system_avoids_symbolic_fallback(root),
        "online_system_supports_learned_memory_toggle": _code_contains_any(
            root,
            [
                "src/core_mem/v2/system.py",
                "scripts/run_stage2_memory_canary.py",
            ],
            [
                "memory_mode",
                "learned_memory",
                "use_learned_memory",
                "learned_memory_checkpoint_dir",
            ],
        ),
        "training_supports_online_aligned": _code_contains_any(
            root,
            [
                "src/core_mem/v2/training.py",
                "scripts/train_stage2.py",
                "configs/stage2_train.yaml",
                "configs/stage2_train_tiny.yaml",
            ],
            [
                "online_aligned",
                "belief_selection",
                "retrieval",
                "composition_to_belief",
            ],
        ),
        "learned_online_gain_is_positive": _positive_learned_gain(learned_gain),
        "non_tiny_trained_eval_token_f1_ge_0_14": best_token_f1 >= 0.14,
        "non_tiny_trained_eval_token_f1_ge_0_20": best_token_f1 >= 0.20,
        "current_head_personamem_learned_64_exists": _learned_alias_ok(
            root,
            benchmark="personamem",
            min_samples=64,
        ),
        "current_head_longmemeval_learned_64_exists": _learned_alias_ok(
            root,
            benchmark="longmemeval",
            min_samples=64,
        ),
        "current_head_personamem_learned_128_exists": _learned_alias_ok(
            root,
            benchmark="personamem",
            min_samples=128,
        ),
        "longmemeval_layered_analysis_exists": _layered_analysis_ok(longmemeval_layered),
    }

    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "best_trained_eval_token_f1": best_token_f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_v21_longrun(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
