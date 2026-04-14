"""Memory data structures for vanilla CoRe Memory."""

from core_mem.embeddings.base import TextEmbedder
from core_mem.memory.core import (
    CoreMemoryState,
    CoreMemoryUpdater,
    MemoryObservation,
    MemoryReadResult,
    MemoryReader,
    MemoryWriter,
    ResidualMemoryManager,
    ResidualSlot,
    VanillaCoreMemory,
)

__all__ = [
    "CoreMemoryState",
    "CoreMemoryUpdater",
    "MemoryObservation",
    "MemoryReadResult",
    "MemoryReader",
    "MemoryWriter",
    "ResidualMemoryManager",
    "ResidualSlot",
    "TextEmbedder",
    "VanillaCoreMemory",
]
