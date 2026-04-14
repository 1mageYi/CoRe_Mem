"""Stage-2 V2.0 structured latent-slot memory primitives."""

from core_mem.v2.consolidation import ConsolidationManager
from core_mem.v2.decoder import BeliefDecoder
from core_mem.v2.encoder import QueryEncoder, SlotEncoder, SlotEncoderConfig
from core_mem.v2.eval_local import evaluate_local, module_inventory_payload
from core_mem.v2.datasets import (
    STAGE2_TASK_DATASETS,
    STAGE2_TASK_MODULES,
    build_demo_prepared_payload,
    build_task_registry,
)
from core_mem.v2.lifecycle import LifecycleDecision, LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.projection import AnswerProjection
from core_mem.v2.public_data import (
    normalize_mquake_dataset,
    normalize_multiwoz24_dataset,
    normalize_personachat_dataset,
    normalize_recoe_dataset,
    normalize_sgd_dataset,
)
from core_mem.v2.relations import DEFAULT_RELATION_SET, normalize_relation
from core_mem.v2.resampler import LightResampler
from core_mem.v2.schemas import (
    BeliefItem,
    BeliefState,
    Observation,
    SlotRecord,
    SoftRoleScores,
)
from core_mem.v2.system import QueryResult, StructuredMemoryState, StructuredMemorySystem

__all__ = [
    "DEFAULT_RELATION_SET",
    "STAGE2_TASK_DATASETS",
    "STAGE2_TASK_MODULES",
    "BeliefDecoder",
    "BeliefItem",
    "BeliefState",
    "ConsolidationManager",
    "evaluate_local",
    "LifecycleDecision",
    "LifecycleManager",
    "LightResampler",
    "module_inventory_payload",
    "normalize_mquake_dataset",
    "normalize_multiwoz24_dataset",
    "normalize_personachat_dataset",
    "normalize_recoe_dataset",
    "normalize_sgd_dataset",
    "Observation",
    "AnswerProjection",
    "QueryEncoder",
    "QueryResult",
    "SlotEncoder",
    "SlotEncoderConfig",
    "SlotRecord",
    "StructuredMemoryState",
    "StructuredMemorySystem",
    "SoftRoleScores",
    "Stage2ObservationParser",
    "build_demo_prepared_payload",
    "build_task_registry",
    "normalize_relation",
]
