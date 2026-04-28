from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from .schemas import MemoryEdge, MemoryNode


def _tokenize(text: str) -> list[str]:
    """Lowercase + whitespace tokenisation for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


@dataclass(slots=True)
class MemoryGraphStore:
    """
    Lightweight in-memory graph store for MVP.

    - DiGraph enables directed temporal/supersedes edges.
    - Bidirectional semantic/co-usage edges are stored as two directed edges.
    - BM25 index is built lazily on first retrieval call and invalidated on
      every ``add_node`` so it stays current without manual management.
    """

    graph: nx.DiGraph = field(init=False)
    nodes: dict[str, MemoryNode] = field(init=False, default_factory=dict)
    _co_usage_hit_counts: dict[tuple[str, str], int] = field(init=False, default_factory=dict)
    # BM25 lazy index
    _bm25_index: object = field(init=False, repr=False)
    _bm25_node_ids: list = field(init=False, repr=False, default_factory=list)
    _bm25_dirty: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes = {}
        self._co_usage_hit_counts = {}
        self._bm25_index = None
        self._bm25_node_ids = []
        self._bm25_dirty = True

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)
        self._bm25_dirty = True  # invalidate index on every new node

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> None:
        """(Re)build the BM25 index over all current node texts."""
        from rank_bm25 import BM25Okapi  # lazy import keeps startup fast

        node_list = list(self.nodes.values())
        self._bm25_node_ids = [n.node_id for n in node_list]
        corpus = [_tokenize(n.structured_record.text) for n in node_list]
        self._bm25_index = BM25Okapi(corpus)
        self._bm25_dirty = False

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to *top_k* (node_id, bm25_score) pairs sorted by score."""
        if self._bm25_dirty or self._bm25_index is None:
            self.build_bm25_index()
        tokens = _tokenize(query)
        import numpy as _np  # already a transitive dep
        scores: list[float] = self._bm25_index.get_scores(tokens).tolist()  # type: ignore[union-attr]
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._bm25_node_ids[i], scores[i]) for i in indices]

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

    def ppr_scores(
        self,
        personalization: dict[str, float],
        alpha: float = 0.85,
    ) -> dict[str, float]:
        """Personalized PageRank from given anchor node weights.

        personalization: {node_id: weight} — anchor nodes with relative
            importance weights (will be L1-normalised internally).
        alpha: restart probability (probability of jumping back to anchor).
            Higher alpha = stays closer to anchors = more local.

        Returns dict of {node_id: ppr_score} for all nodes in the graph.
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        # Keep only anchors that exist in graph; L1-normalise
        pers = {nid: w for nid, w in personalization.items() if nid in self.graph and w > 0}
        if not pers:
            return {}
        total = sum(pers.values())
        pers = {nid: w / total for nid, w in pers.items()}
        return nx.pagerank(self.graph, alpha=alpha, personalization=pers)

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
