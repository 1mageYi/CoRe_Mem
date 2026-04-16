"""High-level route + compact ingest API (DESIGN.md §6)."""

from __future__ import annotations

import torch

from compactor.fusion import MemoryCompactor
from compactor.memory_bank import MemoryBank
from compactor.router import RouterV1, predict_decision
from compactor.types import IngestResult, RouteDecision


@torch.no_grad()
def route_only(
    e_new: torch.Tensor,
    bank: MemoryBank,
    router: RouterV1,
    *,
    device: torch.device | None = None,
) -> RouteDecision:
    """
    Parameters
    ----------
    e_new : (d,) or (1, d)
    """
    dev = device or next(router.parameters()).device
    if e_new.dim() == 1:
        e_new_b = e_new.unsqueeze(0).to(dev)
    else:
        e_new_b = e_new.to(dev)
    if len(bank) == 0:
        return RouteDecision(is_new_slot=True, slot_index=None, logits=None)

    e_cands, cand_mask = bank.stack_for_router(device=dev)
    logits = router(e_new_b, e_cands, cand_mask=cand_mask)
    pred, probs = predict_decision(logits)
    k = int(pred.item())
    kmax = logits.shape[-1] - 1
    if k == kmax:
        return RouteDecision(is_new_slot=True, slot_index=None, logits=logits.squeeze(0))
    return RouteDecision(is_new_slot=False, slot_index=k, logits=logits.squeeze(0))


@torch.no_grad()
def ingest_embedding(
    e_new: torch.Tensor,
    bank: MemoryBank,
    router: RouterV1,
    compactor: MemoryCompactor,
    *,
    device: torch.device | None = None,
) -> IngestResult:
    """Route ``e_new``; on merge path run compactor and replace slot; else append new slot."""
    dev = device or next(router.parameters()).device
    if e_new.dim() == 1:
        e_new = e_new.unsqueeze(0)
    e_new = e_new.to(dev)

    if len(bank) == 0:
        bank.append(e_new.squeeze(0))
        return IngestResult(
            decision=RouteDecision(is_new_slot=True, slot_index=None, logits=None),
            memory_updated=False,
            new_slot_appended=True,
        )

    e_cands, cand_mask = bank.stack_for_router(device=dev)
    logits = router(e_new, e_cands, cand_mask=cand_mask)
    pred, _ = predict_decision(logits)
    k = int(pred.item())
    kmax = logits.shape[-1] - 1

    if k == kmax:
        bank.append(e_new.squeeze(0))
        return IngestResult(
            decision=RouteDecision(is_new_slot=True, slot_index=None, logits=logits.squeeze(0)),
            memory_updated=False,
            new_slot_appended=True,
        )

    e_old = bank.get(k).to(dev)
    if e_old.dim() == 1:
        e_old = e_old.unsqueeze(0)
    e_sum = compactor(e_old, e_new)
    bank.replace(k, e_sum.squeeze(0))
    return IngestResult(
        decision=RouteDecision(is_new_slot=False, slot_index=k, logits=logits.squeeze(0)),
        memory_updated=True,
        new_slot_appended=False,
    )
