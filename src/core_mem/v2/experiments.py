"""Stage-2 experiment variants and registry helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _pool_dataset_aliases() -> dict[str, set[str]]:
    return {
        "state_belief": {
            "sgd",
            "multiwoz24",
            "sgdderivedstatechanges",
            "multiwozderivedstatechanges",
        },
        "persona_preference": {
            "personachat",
            "trainingmillionsofpersonalizeddialogueagents",
        },
        "update_propagation": {
            "mquake",
            "recoe",
            "sgdderivedstatechanges",
            "multiwozderivedstatechanges",
        },
        "minimal_synthetic": {
            "synthetic",
            "minimalsyntheticpool",
            "overwritestress",
            "budgetstress",
            "conflictbundles",
            "multifactcomposition",
        },
    }


POOL_DATASET_ALIASES = _pool_dataset_aliases()


@dataclass(frozen=True)
class Stage2ExperimentVariant:
    resampler_type: str = "light"
    decoder_type: str = "belief_json"
    overwrite_mode: str = "merge_overwrite"
    bank_mode: str = "dual"
    assignment_mode: str = "default"
    disabled_pools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disabled_pools"] = list(self.disabled_pools)
        return payload


@dataclass(frozen=True)
class Stage2ExperimentSpec:
    experiment_id: str
    description: str
    variant: Stage2ExperimentVariant = field(default_factory=Stage2ExperimentVariant)
    budgets: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    notes: str = ""


EXPERIMENT_SPECS = {
    "mainline": Stage2ExperimentSpec(
        experiment_id="mainline",
        description="Default stage-2 mainline training plus local eval.",
        notes="Default light-resampler + belief-JSON route on the prepared public-data manifest.",
    ),
    "ablation_t5_vs_optimus": Stage2ExperimentSpec(
        experiment_id="ablation_t5_vs_optimus",
        description="T5 route vs Optimus-like.",
        variant=Stage2ExperimentVariant(decoder_type="optimus_like"),
        notes="Training path stays seq2seq; local eval swaps the belief decoder to an Optimus-like surrogate.",
    ),
    "ablation_mean_pooling_vs_light_resampler": Stage2ExperimentSpec(
        experiment_id="ablation_mean_pooling_vs_light_resampler",
        description="Mean pooling vs light resampler.",
        variant=Stage2ExperimentVariant(resampler_type="mean_pooling"),
    ),
    "ablation_belief_json_vs_direct_answer": Stage2ExperimentSpec(
        experiment_id="ablation_belief_json_vs_direct_answer",
        description="Belief JSON decoding vs direct answer decoding.",
        variant=Stage2ExperimentVariant(decoder_type="direct_answer"),
    ),
    "ablation_merge_only_vs_merge_overwrite": Stage2ExperimentSpec(
        experiment_id="ablation_merge_only_vs_merge_overwrite",
        description="Merge-only vs merge+overwrite.",
        variant=Stage2ExperimentVariant(overwrite_mode="merge_only"),
    ),
    "ablation_single_bank_vs_core_residual": Stage2ExperimentSpec(
        experiment_id="ablation_single_bank_vs_core_residual",
        description="Single-bank vs core+residual.",
        variant=Stage2ExperimentVariant(bank_mode="single"),
    ),
    "ablation_assignment_randomization": Stage2ExperimentSpec(
        experiment_id="ablation_assignment_randomization",
        description="Core/residual assignment randomization.",
        variant=Stage2ExperimentVariant(assignment_mode="randomized"),
    ),
    "ablation_budget_sweep": Stage2ExperimentSpec(
        experiment_id="ablation_budget_sweep",
        description="Budget sweep.",
        budgets=[1, 2, 4, 8, 12],
    ),
    "ablation_without_update_pool": Stage2ExperimentSpec(
        experiment_id="ablation_without_update_pool",
        description="Without update pool.",
        variant=Stage2ExperimentVariant(disabled_pools=["update_propagation"]),
    ),
    "ablation_without_persona_pool": Stage2ExperimentSpec(
        experiment_id="ablation_without_persona_pool",
        description="Without persona pool.",
        variant=Stage2ExperimentVariant(disabled_pools=["persona_preference"]),
    ),
    "ablation_without_state_pool": Stage2ExperimentSpec(
        experiment_id="ablation_without_state_pool",
        description="Without state pool.",
        variant=Stage2ExperimentVariant(disabled_pools=["state_belief"]),
    ),
    "ablation_without_synthetic_pool": Stage2ExperimentSpec(
        experiment_id="ablation_without_synthetic_pool",
        description="Without synthetic pool.",
        variant=Stage2ExperimentVariant(disabled_pools=["minimal_synthetic"]),
    ),
}


def spec_for_experiment(experiment_id: str) -> Stage2ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[experiment_id]
    except KeyError as exc:
        raise ValueError(f"Unknown stage-2 experiment id: {experiment_id}") from exc


def variant_payload_for_experiment(experiment_id: str) -> dict[str, Any]:
    return spec_for_experiment(experiment_id).variant.to_dict()


def dataset_allowed_for_variant(dataset_name: str, disabled_pools: list[str] | None) -> bool:
    if not disabled_pools:
        return True
    normalized_dataset = _normalize_name(dataset_name)
    for pool_name in disabled_pools:
        if normalized_dataset in POOL_DATASET_ALIASES.get(_normalize_name(pool_name), set()):
            return False
    return True


def stage2_experiment_index_path(output_root: Path) -> Path:
    return output_root / "artifacts" / "stage2_experiment_index.json"


def load_stage2_experiment_index(output_root: Path) -> dict[str, Any]:
    path = stage2_experiment_index_path(output_root)
    if not path.exists():
        return {"experiments": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON in {path}")
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        payload["experiments"] = {}
    return payload


def register_stage2_experiment(
    output_root: Path,
    *,
    experiment_id: str,
    train_run_dir: str,
    checkpoint_dir: str,
    local_eval_path: str,
    summary_table_path: str,
    budget_table_path: str,
    variant: dict[str, Any],
    budgets: list[int],
    notes: str,
    metrics_path: str | None = None,
) -> Path:
    payload = load_stage2_experiment_index(output_root)
    experiments = payload.setdefault("experiments", {})
    experiments[experiment_id] = {
        "completed": True,
        "train_run_dir": train_run_dir,
        "checkpoint_dir": checkpoint_dir,
        "local_eval_path": local_eval_path,
        "summary_table_path": summary_table_path,
        "budget_table_path": budget_table_path,
        "metrics_path": metrics_path,
        "variant": dict(variant),
        "budgets": list(budgets),
        "notes": notes,
    }
    path = stage2_experiment_index_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
