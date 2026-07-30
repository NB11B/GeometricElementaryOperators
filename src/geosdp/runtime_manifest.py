from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


LINEAR_CAPABILITIES: FrozenSet[str] = frozenset({"linear"})
CORE_CAPABILITIES: FrozenSet[str] = frozenset(
    {"linear", "add", "mul", "scale", "rms_norm"}
)
ACTIVATION_CAPABILITIES: FrozenSet[str] = frozenset({"gelu", "silu_mul"})
POSITION_CAPABILITIES: FrozenSet[str] = frozenset({"build_rope", "apply_rope"})
ATTENTION_CAPABILITIES: FrozenSet[str] = frozenset({"causal_attention"})
LOSS_CAPABILITIES: FrozenSet[str] = frozenset({"cross_entropy"})
EMBEDDING_CAPABILITIES: FrozenSet[str] = frozenset({"embedding"})
OPTIMIZER_CAPABILITIES: FrozenSet[str] = frozenset({"adamw"})
ACTIVATION_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(CORE_CAPABILITIES) | set(ACTIVATION_CAPABILITIES)
)
POSITION_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(ACTIVATION_STAGE_CAPABILITIES) | set(POSITION_CAPABILITIES)
)
ATTENTION_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(POSITION_STAGE_CAPABILITIES) | set(ATTENTION_CAPABILITIES)
)
LOSS_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(ATTENTION_STAGE_CAPABILITIES) | set(LOSS_CAPABILITIES)
)
MODEL_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(LOSS_STAGE_CAPABILITIES) | set(EMBEDDING_CAPABILITIES)
)
TRAINING_STAGE_CAPABILITIES: FrozenSet[str] = frozenset(
    set(MODEL_STAGE_CAPABILITIES) | set(OPTIMIZER_CAPABILITIES)
)
FULL_MODEL_CAPABILITIES: FrozenSet[str] = MODEL_STAGE_CAPABILITIES

RUNTIME_STAGE_CAPABILITIES: dict[str, FrozenSet[str]] = {
    "linear": LINEAR_CAPABILITIES,
    "core": CORE_CAPABILITIES,
    "activation": ACTIVATION_STAGE_CAPABILITIES,
    "position": POSITION_STAGE_CAPABILITIES,
    "attention": ATTENTION_STAGE_CAPABILITIES,
    "loss": LOSS_STAGE_CAPABILITIES,
    "model": MODEL_STAGE_CAPABILITIES,
    "transformer": FULL_MODEL_CAPABILITIES,
    "training": TRAINING_STAGE_CAPABILITIES,
}


@dataclass(frozen=True)
class RuntimeReadiness:
    stage: str
    available: FrozenSet[str]
    required: FrozenSet[str]

    @property
    def missing(self) -> FrozenSet[str]:
        return self.required - self.available

    @property
    def ready(self) -> bool:
        return not self.missing


def evaluate_runtime(stage: str, available: set[str] | frozenset[str]) -> RuntimeReadiness:
    try:
        required = RUNTIME_STAGE_CAPABILITIES[stage]
    except KeyError as exc:
        raise ValueError(f"unknown GEO runtime stage: {stage}") from exc
    return RuntimeReadiness(stage=stage, available=frozenset(available), required=required)
