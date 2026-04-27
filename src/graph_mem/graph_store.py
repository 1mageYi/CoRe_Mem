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
    _co_usage_hit_counts: dict[tuple[str, str], int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes = {}
        self._co_usage_hit_counts = {}

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

    def register_co_usage_between(
        self,
        a: str,
        b: str,
        *,
        now_ts: int,
        min_count_to_solidify: int,
    ) -> None:
        """
        Count co-appearance events per unordered pair; only add bidirectional co_usage edges
        after `min_count_to_solidify` joint hits (unless edges already exist).
        """
        key = tuple(sorted((a, b)))
        self._co_usage_hit_counts[key] = self._co_usage_hit_counts.get(key, 0) + 1
        h = self._co_usage_hit_counts[key]

        da_ab = self.graph.has_edge(a, b) and self.graph.get_edge_data(a, b).get("edge_type") == "co_usage"
        if da_ab:
            self.increment_co_usage(a, b, now_ts)
            if self.graph.has_edge(b, a):
                self.increment_co_usage(b, a, now_ts)
            return

        if h >= min_count_to_solidify:
            self.add_edge(
                MemoryEdge(
                    src_node_id=a,
                    dst_node_id=b,
                    edge_type="co_usage",
                    weight=float(h),
                    created_at=now_ts,
                    updated_at=now_ts,
                    usage_count=h,
                )
            )
            self.add_edge(
                MemoryEdge(
                    src_node_id=b,
                    dst_node_id=a,
                    edge_type="co_usage",
                    weight=float(h),
                    created_at=now_ts,
                    updated_at=now_ts,
                    usage_count=h,
                )
            )

    def decay_co_usage_edges(self, decay: float, prune_below: float, now_ts: int) -> None:
        """Multiply co_usage edge weights by decay; drop edges weaker than prune_below."""
        if decay >= 1.0 and prune_below <= 0.0:
            return
        edges = list(self.graph.edges(data=True))
        for u, v, data in edges:
            if data.get("edge_type") != "co_usage":
                continue
            if decay < 1.0:
                w = float(data.get("weight", 0.0)) * decay
                data["weight"] = w
                data["updated_at"] = now_ts
            else:
                w = float(data.get("weight", 0.0))
            if prune_below > 0.0 and w < prune_below:
                self.graph.remove_edge(u, v)

    def centrality(self) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, alpha=0.85)

    def compute_core_scores(self) -> dict[str, float]:
        """
        Combine three signals into a stable core score:
        - pagerank centrality
        - in-degree
        - retrieval usage count
        """
        if not self.nodes:
            return {}
        pr = self.centrality()
        indeg = dict(self.graph.in_degree())
        usage = {nid: node.retrieve_count for nid, node in self.nodes.items()}

        pr_norm = _minmax_normalize(pr)
        indeg_norm = _minmax_normalize(indeg)
        usage_norm = _minmax_normalize(usage)

        scores: dict[str, float] = {}
        for nid, node in self.nodes.items():
            score = (
                0.50 * float(pr_norm.get(nid, 0.0))
                + 0.30 * float(indeg_norm.get(nid, 0.0))
                + 0.20 * float(usage_norm.get(nid, 0.0))
            )
            node.core_score = score
            scores[nid] = score
        return scores

    def neighbors_by_type(
        self,
        node_id: str,
        edge_type: str,
        limit: int,
        *,
        min_usage_count: int = 1,
    ) -> list[str]:
        out: list[str] = []
        for nxt in self.graph.successors(node_id):
            data = self.graph.get_edge_data(node_id, nxt)
            if not data or data.get("edge_type") != edge_type:
                continue
            if edge_type == "co_usage" and int(data.get("usage_count", 1)) < min_usage_count:
                continue
            out.append(nxt)
            if len(out) >= limit:
                break
        return out

    def bump_retrieve_counts(self, node_ids: list[str]) -> None:
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            node.retrieve_count += 1


def _minmax_normalize(values: dict[str, float | int]) -> dict[str, float]:
    if not values:
        return {}
    floats = {k: float(v) for k, v in values.items()}
    vmin = min(floats.values())
    vmax = max(floats.values())
    if vmax <= vmin:
        return {k: 0.0 for k in floats}
    return {k: (v - vmin) / (vmax - vmin) for k, v in floats.items()}
