"""Compactor: gated fusion of two embeddings (DESIGN.md §3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCompactor(nn.Module):
    """Element-wise gated blend of ``e_old`` and ``e_new``, then L2 normalize.

    ``g = sigmoid(W [e_old; e_new])``, ``e_sum = normalize(g * e_old + (1-g) * e_new)``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, e_old: torch.Tensor, e_new: torch.Tensor) -> torch.Tensor:
        if e_old.shape != e_new.shape:
            raise ValueError("e_old and e_new must have the same shape")
        x = torch.cat([e_old, e_new], dim=-1)
        g = torch.sigmoid(self.gate(x))
        out = g * e_old + (1.0 - g) * e_new
        return F.normalize(out, p=2, dim=-1)
