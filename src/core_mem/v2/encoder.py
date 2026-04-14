"""Slot/query encoders for the stage-2 skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from core_mem.v2.schemas import Observation, SlotRecord, SoftRoleScores
from core_mem.v2.vector_ops import hash_text_to_vector


@dataclass(frozen=True)
class SlotEncoderConfig:
    retrieval_dim: int = 8
    slot_tokens: int = 8
    token_dim: int = 8


@dataclass(frozen=True)
class QueryEncoder:
    dimension: int = 8

    def encode(self, text: str) -> list[float]:
        return hash_text_to_vector(text, self.dimension)


@dataclass(frozen=True)
class SlotEncoder:
    config: SlotEncoderConfig = SlotEncoderConfig()

    def encode(
        self,
        observation: Observation,
        *,
        timestamp: str,
        existing_slot: SlotRecord | None = None,
        bank: str = "residual",
        revision_parent: str | None = None,
    ) -> SlotRecord:
        retrieval_key = hash_text_to_vector(
            f"{observation.relation}|{observation.value}",
            self.config.retrieval_dim,
        )
        latent_tokens = [
            hash_text_to_vector(
                f"{observation.canonical_gloss}|{token_idx}",
                self.config.token_dim,
            )
            for token_idx in range(self.config.slot_tokens)
        ]
        digest = hashlib.sha1(f"{observation.obs_id}|{bank}|{timestamp}".encode("utf-8")).hexdigest()[:12]
        return SlotRecord(
            slot_id=f"slot_{digest}",
            bank=bank,
            entity=observation.entity,
            relation=observation.relation,
            retrieval_key=retrieval_key,
            latent_tokens=latent_tokens,
            soft_role_scores=self._soft_roles(observation),
            confidence=observation.confidence,
            first_seen_ts=existing_slot.first_seen_ts if existing_slot else timestamp,
            last_update_ts=timestamp,
            revision_count=(existing_slot.revision_count + 1) if existing_slot else 0,
            active_flag=True,
            revision_parent=revision_parent,
            canonical_gloss=observation.canonical_gloss,
        )

    @staticmethod
    def _soft_roles(observation: Observation) -> SoftRoleScores:
        relation = observation.relation
        return SoftRoleScores(
            stable=0.85 if observation.time_scope == "current" and observation.status_hint == "active" else 0.35,
            preference=0.95 if relation.endswith("preference") or relation == "hobby" else 0.05,
            constraint=0.95 if relation == "constraint" or observation.value_type == "constraint" else 0.0,
            goal=0.95 if relation == "goal" or observation.value_type == "goal" else 0.0,
            temporal=0.9 if observation.time_scope in {"past", "future", "recent_change"} else 0.1,
            social=0.9 if relation == "relationship" else 0.0,
        )
