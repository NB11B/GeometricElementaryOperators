from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CalibrationConfig:
    variant_name: str
    seed: int
    data_order_seed: int

    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float

    warmup_steps: int
    total_steps: int
    scheduler: str
    minimum_lr_ratio: float

    max_grad_norm: float | None

    batch_size: int
    sequence_length: int
    checkpoint_steps: tuple[int, ...]

    batch_sampling_mode: str

    def config_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
