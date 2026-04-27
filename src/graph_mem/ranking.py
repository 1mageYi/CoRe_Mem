from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RankingConfig:
    alpha_semantic: float = 0.60
    beta_centrality: float = 0.20
    gamma_edge_evidence: float = 0.15
    delta_temporal_fit: float = 0.05
    lambda_core: float = 0.20
    inactive_penalty: float = 0.35


def fuse_score(
    *,
    semantic_score: float,
    centrality_score: float,
    edge_evidence_score: float,
    temporal_fit_score: float,
    is_core: bool,
    is_active: bool,
    cfg: RankingConfig,
) -> float:
    base = (
        cfg.alpha_semantic * semantic_score
        + cfg.beta_centrality * centrality_score
        + cfg.gamma_edge_evidence * edge_evidence_score
        + cfg.delta_temporal_fit * temporal_fit_score
    )
    if is_core:
        base *= 1.0 + cfg.lambda_core
    if not is_active:
        base *= 1.0 - cfg.inactive_penalty
    return base
