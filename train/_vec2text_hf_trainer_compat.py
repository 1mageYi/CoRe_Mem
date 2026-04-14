"""
transformers>=4.46 Trainer.training_step passes num_items_in_batch= to compute_loss.
vec2text Corrector.compute_loss predates this keyword — accept and ignore it.
"""
from __future__ import annotations


def apply_corrector_compute_loss_compat() -> None:
    import vec2text.trainers.corrector as cm

    if getattr(cm.Corrector, "_hf_compute_loss_patched", False):
        return

    _orig = cm.Corrector.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return _orig(self, model, inputs, return_outputs=return_outputs)

    cm.Corrector.compute_loss = compute_loss  # type: ignore[assignment]
    cm.Corrector._hf_compute_loss_patched = True  # type: ignore[attr-defined]
