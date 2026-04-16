"""Tensor datasets for Router / Compactor training (synthetic or pre-encoded)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RouterTensorDataset(Dataset):
    """Fixed-K Router batches: ``e_new``, ``e_cands``, ``cand_mask``, label ``0..K`` (``K`` = new_slot)."""

    def __init__(
        self,
        e_new: torch.Tensor,
        e_cands: torch.Tensor,
        cand_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Parameters
        ----------
        e_new : (N, d)
        e_cands : (N, K, d)
        cand_mask : (N, K) bool
        labels : (N,) long in ``0..K`` (inclusive; ``K`` codes new_slot)
        """
        if not (e_new.shape[0] == e_cands.shape[0] == cand_mask.shape[0] == labels.shape[0]):
            raise ValueError("batch dimension mismatch")
        self.e_new = e_new
        self.e_cands = e_cands
        self.cand_mask = cand_mask
        self.labels = labels

    def __len__(self) -> int:
        return self.e_new.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.e_new[idx],
            self.e_cands[idx],
            self.cand_mask[idx],
            self.labels[idx],
        )


def make_synthetic_router_data(
    *,
    n: int,
    k: int,
    d: int,
    seed: int = 42,
    device: torch.device | None = None,
) -> RouterTensorDataset:
    """Random unit vectors + random labels (sanity / smoke only)."""
    g = torch.Generator(device=device or torch.device("cpu"))
    g.manual_seed(seed)
    e_new = F.normalize(torch.randn(n, d, generator=g, device=device), dim=-1)
    e_cands = F.normalize(torch.randn(n, k, d, generator=g, device=device), dim=-1)
    cand_mask = torch.ones(n, k, dtype=torch.bool, device=device)
    labels = torch.randint(0, k + 1, (n,), generator=g, device=device)
    return RouterTensorDataset(e_new, e_cands, cand_mask, labels)


class CompactorTensorDataset(Dataset):
    """Triples (e_old, e_new, e_target) for cosine regression."""

    def __init__(self, e_old: torch.Tensor, e_new: torch.Tensor, e_target: torch.Tensor):
        if e_old.shape != e_new.shape or e_old.shape != e_target.shape:
            raise ValueError("e_old, e_new, e_target must match shapes (N, d)")
        self.e_old = e_old
        self.e_new = e_new
        self.e_target = e_target

    def __len__(self) -> int:
        return self.e_old.shape[0]

    def __getitem__(self, idx: int):
        return self.e_old[idx], self.e_new[idx], self.e_target[idx]


def make_synthetic_compactor_data(
    *,
    n: int,
    d: int,
    seed: int = 42,
    device: torch.device | None = None,
) -> CompactorTensorDataset:
    """Weak teacher: ``e_target = normalize(0.5*e_old + 0.5*e_new)``."""
    g = torch.Generator(device=device or torch.device("cpu"))
    g.manual_seed(seed)
    e_old = F.normalize(torch.randn(n, d, generator=g, device=device), dim=-1)
    e_new = F.normalize(torch.randn(n, d, generator=g, device=device), dim=-1)
    e_target = F.normalize(0.5 * e_old + 0.5 * e_new, dim=-1)
    return CompactorTensorDataset(e_old, e_new, e_target)
