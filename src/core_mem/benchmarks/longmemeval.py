"""LongMemEval-S adapter skeleton for stage-1 integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongMemEvalConfig:
    subset: str = "s"


class LongMemEvalAdapter:
    def __init__(self, data_root: str | Path = "data/longmemeval") -> None:
        self.data_root = Path(data_root)

    def planned_subset_path(self, subset: str = "s") -> Path:
        return self.data_root / subset

