"""PersonaMem adapter skeleton for stage-1 integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaMemConfig:
    variant: str = "32k"
    task: str = "multiple_choice"


class PersonaMemAdapter:
    def __init__(self, data_root: str | Path = "data/personamem") -> None:
        self.data_root = Path(data_root)

    def planned_variant_path(self, variant: str = "32k") -> Path:
        return self.data_root / variant

