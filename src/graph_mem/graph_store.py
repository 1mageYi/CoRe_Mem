from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .schemas import MemoryEdge, MemoryNode


@dataclass(slots=True)
class MemoryGraphStore:
    """
    Lightweight in-memory graph store for MVP.

    - DiGraph enables directed temporal/supersedes edges.
    - Bidirectional semantic/co-usage edges are stored as two directed edges.
    """

    graph: nx.DiGraph = field(init=False)
    nodes: dict[str, MemoryNode] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes = {}

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)

    def get_node(self, node_id: str) -> MemoryNode:
        return self.nodes[node_id]

    def all_nodes(self) -> list[MemoryNode]:
        return list(self.nodes.values())

    def add_edge(self, edge: MemoryEdge) -> None:
        self.graph.add_edge(
            edge.src_node_id,
            edge.dst_node_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
            usage_count=edge.usage_count,
        )

    def has_edge(self, src: str, dst: str) -> bool:
        return self.graph.has_edge(src, dst)

    def increment_co_usage(self, src: str, dst: str, now_ts: int) -> None:
        data = self.graph.get_edge_data(src, dst)
        if not data:
            return
        data["usage_count"] = int(data.get("usage_count", 1)) + 1
        data["weight"] = float(data.get("weight", 0.0)) + 1.0
        data["updated_at"] = now_ts

    def centrality(self) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, alpha=0.85)

    def neighbors_by_type(self, node_id: str, edge_type: str, limit: int) -> list[str]:
        out: list[str] = []
        for nxt in self.graph.successors(node_id):
            data = self.graph.get_edge_data(node_id, nxt)
            if data and data.get("edge_type") == edge_type:
                out.append(nxt)
            if len(out) >= limit:
                break
        return out
