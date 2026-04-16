"""Router v1: score K candidates + new_slot (K+1-way). See DESIGN.md §2.2, §3.1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair_features(e_new: torch.Tensor, e_old: torch.Tensor) -> torch.Tensor:
    """Concat [e_new, e_old, e_new*e_old, |e_new-e_old|] -> (..., 4d)."""
    return torch.cat([e_new, e_old, e_new * e_old, torch.abs(e_new - e_old)], dim=-1)


class RouterV1(nn.Module):
    """Per-candidate MLP logits + a global reject (new_slot) logit.

    Forward returns raw logits of shape (B, K+1): classes ``0..K-1`` merge into that
    slot, class ``K`` means new_slot. Invalid candidate positions should be masked
    to a large negative value before softmax or argmax (see ``apply_cand_mask``).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        in_dim = embed_dim * 4
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reject_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        e_new: torch.Tensor,
        e_cands: torch.Tensor,
        cand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        e_new : (B, d)
        e_cands : (B, K, d)
        cand_mask : (B, K) bool, True = valid candidate (padding False)

        Returns
        -------
        logits : (B, K+1) — last column is new_slot.
        """
        if e_cands.dim() != 3:
            raise ValueError("e_cands must be (B, K, d)")
        b, k, d = e_cands.shape
        if e_new.shape != (b, d):
            raise ValueError("e_new must be (B, d) matching e_cands")
        en = e_new.unsqueeze(1).expand(-1, k, -1)
        pair = _pair_features(en, e_cands)
        cand_logits = self.pair_mlp(pair).squeeze(-1)  # (B, K)
        reject = self.reject_mlp(e_new)  # (B, 1)
        logits = torch.cat([cand_logits, reject], dim=-1)  # (B, K+1)

        if cand_mask is not None:
            logits = apply_cand_mask(logits, cand_mask)
        return logits


def apply_cand_mask(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid candidate columns (0..K-1) to -inf; leave new_slot column untouched."""
    if logits.dim() != 2:
        raise ValueError("logits must be (B, K+1)")
    b, kp1 = logits.shape
    k = kp1 - 1
    if cand_mask.shape != (b, k):
        raise ValueError("cand_mask must be (B, K)")
    out = logits.clone()
    neg = torch.finfo(out.dtype).min / 2  # avoid -inf in edge cases with softmax
    out[:, :k] = out[:, :k].masked_fill(~cand_mask, neg)
    return out


def predict_decision(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    logits : (B, K+1)

    Returns
    -------
    pred_class : (B,) long in ``0..K`` (``K`` means new_slot)
    probs : (B, K+1) softmax probabilities
    """
    probs = F.softmax(logits, dim=-1)
    return torch.argmax(probs, dim=-1), probs
