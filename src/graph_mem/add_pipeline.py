from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .embedder import BgeM3Embedder
from .formatter import format_structured_text
from .graph_store import MemoryGraphStore
from .schemas import MemoryEdge, MemoryNode, StructuredRecord

MergeStrategy = Literal["hybrid", "latest_wins"]


@dataclass(slots=True)
class AddConfig:
    merge_threshold: float = 0.88
    semantic_edge_threshold: float = 0.78
    semantic_edge_topk: int = 8
    # latest_wins: high-sim always overwrites nearest node; hybrid: only update_type==update merges
    merge_strategy: MergeStrategy = "hybrid"


@dataclass(slots=True)
class AddPipeline:
    graph_store: MemoryGraphStore
    embedder: BgeM3Embedder
    cfg: AddConfig = field(default_factory=AddConfig)

    def add_record(self, record: StructuredRecord, *, now_ts: int) -> str:
        structured_text = format_structured_text(record)
        # Embed raw text so semantic similarity is driven by content, not
        # the shared metadata prefix that would compress all embeddings together.
        emb = self.embedder.encode(record.text)

        nearest_id, nearest_sim = self._nearest_neighbor(emb)
        if nearest_id is None:
            node_id = self._create_node(record, structured_text, emb, now_ts)
            self._attach_semantic_edges(node_id, now_ts)
            return node_id

        if nearest_sim >= self.cfg.merge_threshold and self.cfg.merge_strategy == "latest_wins":
            node = self.graph_store.get_node(nearest_id)
            node.structured_record = record
            node.structured_text = structured_text
            node.embedding = emb
            node.updated_at = now_ts
            node.version += 1
            self._attach_semantic_edges(nearest_id, now_ts)
            return nearest_id

        if nearest_sim >= self.cfg.merge_threshold and record.update_type == "update":
            node = self.graph_store.get_node(nearest_id)
            node.structured_record = record
            node.structured_text = structured_text
            node.embedding = emb
            node.updated_at = now_ts
            node.version += 1
            self._attach_semantic_edges(nearest_id, now_ts)
            return nearest_id

        # Hybrid conflict policy:
        # - revision/conflict/new -> create new node
        # - link supersedes when it is a revision over highly similar node.
        new_id = self._create_node(record, structured_text, emb, now_ts)
        if nearest_sim >= self.cfg.merge_threshold and record.update_type in ("revision", "conflict"):
            old = self.graph_store.get_node(nearest_id)
            old.state = "superseded"
            old.valid_to = now_ts
            self.graph_store.get_node(new_id).supersedes = nearest_id
            self.graph_store.add_edge(
                MemoryEdge(
                    src_node_id=nearest_id,
                    dst_node_id=new_id,
                    edge_type="supersedes",
                    weight=1.0,
                    created_at=now_ts,
                    updated_at=now_ts,
                )
            )

        # Temporal edge from nearest context node to newest node.
        self.graph_store.add_edge(
            MemoryEdge(
                src_node_id=nearest_id,
                dst_node_id=new_id,
                edge_type="temporal",
                weight=1.0,
                created_at=now_ts,
                updated_at=now_ts,
            )
        )
        self._attach_semantic_edges(new_id, now_ts)
        return new_id

    def _create_node(self, record: StructuredRecord, text: str, emb: list[float], now_ts: int) -> str:
        node_id = f"n_{len(self.graph_store.nodes) + 1}"
        node = MemoryNode(
            node_id=node_id,
            structured_record=record,
            structured_text=text,
            embedding=emb,
            created_at=now_ts,
            updated_at=now_ts,
            state="active",
            valid_from=record.time_index,
        )
        self.graph_store.add_node(node)
        return node_id

    def _nearest_neighbor(self, emb: list[float]) -> tuple[str | None, float]:
        best_id: str | None = None
        best_sim = -1.0
        for node in self.graph_store.all_nodes():
            sim = self.embedder.cosine(emb, node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = node.node_id
        return best_id, max(best_sim, 0.0)

    def _attach_semantic_edges(self, node_id: str, now_ts: int) -> None:
        node = self.graph_store.get_node(node_id)
        candidates: list[tuple[str, float]] = []
        for other in self.graph_store.all_nodes():
            if other.node_id == node_id:
                continue
            sim = self.embedder.cosine(node.embedding, other.embedding)
            if sim >= self.cfg.semantic_edge_threshold:
                candidates.append((other.node_id, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for other_id, sim in candidates[: self.cfg.semantic_edge_topk]:
            self.graph_store.add_edge(
                MemoryEdge(
                    src_node_id=node_id,
                    dst_node_id=other_id,
                    edge_type="semantic",
                    weight=sim,
                    created_at=now_ts,
                    updated_at=now_ts,
                )
            )
            self.graph_store.add_edge(
                MemoryEdge(
                    src_node_id=other_id,
                    dst_node_id=node_id,
                    edge_type="semantic",
                    weight=sim,
                    created_at=now_ts,
                    updated_at=now_ts,
                )
            )
