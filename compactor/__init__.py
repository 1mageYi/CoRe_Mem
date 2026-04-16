"""Memory compaction: Router v1 + gated MemoryCompactor (see DESIGN.md)."""

from compactor.encoding import encode_texts, encode_texts_numpy, load_encoder
from compactor.fusion import MemoryCompactor
from compactor.inference import ingest_embedding, route_only
from compactor.memory_bank import MemoryBank
from compactor.router import RouterV1, apply_cand_mask, predict_decision
from compactor.types import IngestResult, RouteDecision

__all__ = [
    "MemoryBank",
    "MemoryCompactor",
    "RouterV1",
    "RouteDecision",
    "IngestResult",
    "apply_cand_mask",
    "predict_decision",
    "route_only",
    "ingest_embedding",
    "load_encoder",
    "encode_texts",
    "encode_texts_numpy",
]
