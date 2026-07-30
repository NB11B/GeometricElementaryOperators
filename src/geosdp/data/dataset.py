from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch


class ResumableDatasetLoader:
    def __init__(
        self,
        inputs: List[List[int]],
        targets: List[List[int]],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.device = device
        self.cursor = 0
        self.total_samples = len(inputs)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size > self.total_samples:
            self.cursor = 0  # Wrap around epoch

        batch_x = self.inputs[self.cursor : self.cursor + self.batch_size]
        batch_y = self.targets[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size

        x_tensor = torch.tensor(batch_x, dtype=torch.int64, device=self.device)
        y_tensor = torch.tensor(batch_y, dtype=torch.int64, device=self.device)
        return x_tensor, y_tensor

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "total_samples": self.total_samples}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = int(state.get("cursor", 0))
