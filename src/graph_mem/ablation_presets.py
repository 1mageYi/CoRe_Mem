"""Phase 3 ablation presets: edge expansion modes and merge strategies."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .add_pipeline import AddConfig, MergeStrategy
from .search_pipeline import SearchConfig

GraphEdgeMode = Literal["full", "semantic_only", "semantic_temporal", "semantic_co_usage"]


def search_config_for_edge_mode(mode: GraphEdgeMode, base: SearchConfig | None = None) -> SearchConfig:
    """Control which edge types participate in 1-hop expansion (seeds always included)."""
    sc = base or SearchConfig()
    if mode == "full":
        return sc
    if mode == "semantic_only":
        return replace(
            sc,
            expand_use_semantic=True,
            expand_use_temporal=False,
            expand_use_co_usage=False,
        )
    if mode == "semantic_temporal":
        return replace(
            sc,
            expand_use_semantic=True,
            expand_use_temporal=True,
            expand_use_co_usage=False,
        )
    if mode == "semantic_co_usage":
        return replace(
            sc,
            expand_use_semantic=True,
            expand_use_temporal=False,
            expand_use_co_usage=True,
        )
    raise ValueError(f"unknown graph_edge_mode: {mode}")


def add_config_for_merge(strategy: MergeStrategy, base: AddConfig | None = None) -> AddConfig:
    ac = base or AddConfig()
    return replace(ac, merge_strategy=strategy)
