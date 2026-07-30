from __future__ import annotations

import math
import torch


class GeoLRScheduler:
    """Cosine Learning Rate Scheduler with Linear Warmup for GEO Language Models."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_lr: float = 1e-3, min_lr: float = 1e-4, warmup_steps: int = 50, max_steps: int = 500):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self) -> float:
        """Update optimizer learning rate according to linear warmup + cosine decay schedule."""
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / max(1, self.warmup_steps))
        elif self.current_step > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state: dict):
        self.current_step = state.get("current_step", 0)
        self.max_lr = state.get("max_lr", 1e-3)
        self.min_lr = state.get("min_lr", 1e-4)
        self.warmup_steps = state.get("warmup_steps", 50)
        self.max_steps = state.get("max_steps", 500)
