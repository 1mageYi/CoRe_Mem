"""In-memory bank of embedding vectors (one tensor per slot)."""

from __future__ import annotations

import torch


class MemoryBank:
    """Stores L2-normalized memory vectors on CPU or a single device."""

    def __init__(self, device: torch.device | None = None):
        self._slots: list[torch.Tensor] = []
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return len(self._slots)

    def clear(self) -> None:
        self._slots.clear()

    def append(self, e: torch.Tensor) -> int:
        """Append one (d,) or (1,d) embedding; returns new index."""
        if e.dim() == 2 and e.shape[0] == 1:
            e = e.squeeze(0)
        if e.dim() != 1:
            raise ValueError("expected shape (d,) or (1,d)")
        x = e.detach().to(dtype=torch.float32, device=self.device)
        idx = len(self._slots)
        self._slots.append(x)
        return idx

    def replace(self, index: int, e: torch.Tensor) -> None:
        if e.dim() == 2 and e.shape[0] == 1:
            e = e.squeeze(0)
        self._slots[index] = e.detach().to(dtype=torch.float32, device=self.device)

    def get(self, index: int) -> torch.Tensor:
        return self._slots[index]

    def stack_for_router(self, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (e_cands, cand_mask) with shapes (1, K, d) and (1, K) for batch size 1."""
        if len(self._slots) == 0:
            raise ValueError("empty memory bank")
        dev = device or self.device
        t = torch.stack(self._slots, dim=0).to(dev)  # (K, d)
        t = t.unsqueeze(0)  # (1, K, d)
        k = t.shape[1]
        mask = torch.ones(1, k, dtype=torch.bool, device=dev)
        return t, mask

    def slots_tensor(self, *, device: torch.device | None = None) -> torch.Tensor:
        """(K, d) stack of all slots."""
        if len(self._slots) == 0:
            raise ValueError("empty memory bank")
        dev = device or self.device
        return torch.stack(self._slots, dim=0).to(dev)
