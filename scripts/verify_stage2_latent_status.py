"""Mechanical verifier for stage-2 latent-memory readiness."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.decoder import BeliefDecoder
from core_mem.v2.encoder import QueryEncoder, SlotEncoder
from core_mem.v2.resampler import LightResampler
from core_mem.v2.system import StructuredMemorySystem


def _contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _contains_any(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern in text for pattern in patterns)


def compute_latent_status(root: Path) -> dict[str, Any]:
    requirements_path = root / "docs" / "requirements.md"
    current_status_path = root / "docs" / "current_status.md"
    benchmark_runner_path = root / "scripts" / "run_stage2_memory_canary.py"
    benchmark_outputs = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary*.json"))

    query_encoder_src = inspect.getsource(QueryEncoder.encode)
    slot_encoder_src = inspect.getsource(SlotEncoder.encode)
    resampler_src = inspect.getsource(LightResampler.compose)
    decoder_src = inspect.getsource(BeliefDecoder.decode)
    system_query_src = inspect.getsource(StructuredMemorySystem.query)

    checks = {
        "requirements_latent_priority_documented": _contains(
            requirements_path,
            "第二阶段当前执行优先级必须先放在 **把真正的 latent memory 主链路做实**",
        ),
        "current_status_latent_priority_documented": _contains_any(
            current_status_path,
            [
                "当前最重要的下一步已经从“继续补 local eval 接口 / canary 记录”切换为“先把真正的 latent memory 主链路做实”",
                "以系统/模型/latent 本体更强、更稳健为锚点提升 local intrinsic 质量",
            ],
        ),
        "query_encoder_not_hash_only": "hash_text_to_vector" not in query_encoder_src,
        "slot_encoder_not_hash_only": "hash_text_to_vector" not in slot_encoder_src,
        "resampler_not_mean_only_skeleton": "mean_vectors" not in resampler_src,
        "decoder_accepts_composed_memory": "composed_memory" in decoder_src,
        "system_passes_composed_memory_to_decoder": bool(
            re.search(r"decoder\.decode\([^)]*composed_memory", system_query_src, re.DOTALL)
        ),
        "stage2_memory_canary_runner_exists": benchmark_runner_path.exists(),
        "stage2_memory_canary_artifact_exists": bool(benchmark_outputs),
    }
    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "benchmark_outputs": [str(path) for path in benchmark_outputs[-5:]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_latent_status(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
