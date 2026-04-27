from __future__ import annotations

from dataclasses import dataclass, field

from .embedder import BgeM3Embedder
from .graph_store import MemoryGraphStore
from .ranking import RankingConfig, fuse_score
from .schemas import MemoryEdge


@dataclass(slots=True)
class SearchConfig:
    seed_topk: int = 12
    expand_hop: int = 1
    expand_max_neighbors_per_type: int = 6
    final_topn_evidence: int = 8
    co_usage_min_count: int = 2
    core_top_ratio: float = 0.2


@dataclass(slots=True)
class SearchPipeline:
    graph_store: MemoryGraphStore
    embedder: BgeM3Embedder
    rank_cfg: RankingConfig = field(default_factory=RankingConfig)
    cfg: SearchConfig = field(default_factory=SearchConfig)

    def search(self, query: str, *, now_ts: int) -> list[str]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec)
        expanded = self._expand(seeds)
        ranked = self._rank(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        return [self.graph_store.get_node(nid).structured_text for nid in top_ids]

    def search_with_debug(self, query: str, *, now_ts: int) -> tuple[list[str], dict]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec)
        expanded = self._expand(seeds)
        ranked, breakdown = self._rank_with_breakdown(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        evidence = [self.graph_store.get_node(nid).structured_text for nid in top_ids]
        debug = {
            "seeds": seeds,
            "expanded": expanded,
            "ranked_top_ids": top_ids,
            "score_breakdown": breakdown,
        }
        return evidence, debug

    def _seed_retrieve(self, qvec: list[float]) -> list[str]:
        scores = []
        for node in self.graph_store.all_nodes():
            sim = self.embedder.cosine(qvec, node.embedding)
            scores.append((node.node_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scores[: self.cfg.seed_topk]]

    def _expand(self, seeds: list[str]) -> list[str]:
        visited = set(seeds)
        frontier = list(seeds)
        for _ in range(self.cfg.expand_hop):
            nxt_frontier: list[str] = []
            for nid in frontier:
                for edge_type in ("semantic", "temporal", "co_usage"):
                    neighbors = self.graph_store.neighbors_by_type(
                        nid, edge_type=edge_type, limit=self.cfg.expand_max_neighbors_per_type
                    )
                    for nb in neighbors:
                        if nb not in visited:
                            visited.add(nb)
                            nxt_frontier.append(nb)
            frontier = nxt_frontier
        return list(visited)

    def _rank(self, candidates: list[str], qvec: list[float]) -> list[tuple[str, float]]:
        centrality = self.graph_store.centrality()
        core_scores = self.graph_store.compute_core_scores()
        ranked: list[tuple[str, float]] = []
        for nid in candidates:
            node = self.graph_store.get_node(nid)
            semantic = self.embedder.cosine(qvec, node.embedding)
            cent = float(core_scores.get(nid, 0.0))
            edge_ev = self._edge_evidence_score(nid)
            temporal_fit = 1.0 if node.state == "active" else 0.5
            is_core = cent >= self._core_threshold(core_scores)
            score = fuse_score(
                semantic_score=semantic,
                centrality_score=cent,
                edge_evidence_score=edge_ev,
                temporal_fit_score=temporal_fit,
                is_core=is_core,
                is_active=node.state == "active",
                cfg=self.rank_cfg,
            )
            ranked.append((nid, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _rank_with_breakdown(self, candidates: list[str], qvec: list[float]) -> tuple[list[tuple[str, float]], list[dict]]:
        centrality = self.graph_store.centrality()
        core_scores = self.graph_store.compute_core_scores()
        ranked: list[tuple[str, float]] = []
        breakdown: list[dict] = []
        cth = self._core_threshold(core_scores)
        for nid in candidates:
            node = self.graph_store.get_node(nid)
            semantic = self.embedder.cosine(qvec, node.embedding)
            cent = float(core_scores.get(nid, 0.0))
            edge_ev = self._edge_evidence_score(nid)
            temporal_fit = 1.0 if node.state == "active" else 0.5
            is_core = cent >= cth
            score = fuse_score(
                semantic_score=semantic,
                centrality_score=cent,
                edge_evidence_score=edge_ev,
                temporal_fit_score=temporal_fit,
                is_core=is_core,
                is_active=node.state == "active",
                cfg=self.rank_cfg,
            )
            ranked.append((nid, score))
            breakdown.append(
                {
                    "node_id": nid,
                    "semantic": round(semantic, 6),
                    "centrality": round(float(centrality.get(nid, 0.0)), 6),
                    "core_score": round(cent, 6),
                    "retrieve_count": node.retrieve_count,
                    "edge_evidence": round(edge_ev, 6),
                    "temporal_fit": round(temporal_fit, 6),
                    "is_core": is_core,
                    "is_active": node.state == "active",
                    "final_score": round(score, 6),
                }
            )
        ranked.sort(key=lambda x: x[1], reverse=True)
        breakdown.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked, breakdown

    def _edge_evidence_score(self, node_id: str) -> float:
        out_degree = self.graph_store.graph.out_degree(node_id)
        if out_degree == 0:
            return 0.0
        # Normalize with a simple bounded transform.
        return min(1.0, out_degree / 10.0)

    def _core_threshold(self, core_scores: dict[str, float]) -> float:
        if not core_scores:
            return 0.0
        vals = sorted(core_scores.values(), reverse=True)
        top_n = max(1, int(len(vals) * self.cfg.core_top_ratio))
        return vals[top_n - 1]

    def _update_co_usage(self, top_ids: list[str], now_ts: int) -> None:
        self.graph_store.bump_retrieve_counts(top_ids)
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                a, b = top_ids[i], top_ids[j]
                if self.graph_store.has_edge(a, b):
                    self.graph_store.increment_co_usage(a, b, now_ts)
                    self.graph_store.increment_co_usage(b, a, now_ts)
                    continue
                edge = MemoryEdge(
                    src_node_id=a,
                    dst_node_id=b,
                    edge_type="co_usage",
                    weight=1.0,
                    created_at=now_ts,
                    updated_at=now_ts,
                    usage_count=1,
                )
                back = MemoryEdge(
                    src_node_id=b,
                    dst_node_id=a,
                    edge_type="co_usage",
                    weight=1.0,
                    created_at=now_ts,
                    updated_at=now_ts,
                    usage_count=1,
                )
                self.graph_store.add_edge(edge)
                self.graph_store.add_edge(back)
