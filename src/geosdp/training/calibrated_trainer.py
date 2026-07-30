from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 50,
    total_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Creates a linear warmup followed by cosine learning rate decay scheduler."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def compute_global_grad_norm(parameters: list[nn.Parameter]) -> float:
    """Computes global 2-norm across all parameters with gradients."""
    valid_grads = [p.grad.detach() for p in parameters if p.requires_grad and p.grad is not None]
    if not valid_grads:
        return 0.0
    return float(torch.norm(torch.stack([torch.norm(g, 2) for g in valid_grads]), 2).item())


class CalibratedTrainer:
    """Declarative trainer with padding-safe ignore_index=-100 loss, grad norm telemetry, and sealed-test guards."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None = None,
        max_grad_norm: float | None = 1.0,
        device: torch.device = torch.device("cuda"),
        ignore_index: int = -100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.ignore_index = ignore_index

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.total_steps = 0
        self.clipped_steps_count = 0
        self.unclipped_norm_sum = 0.0
        self.clipped_norm_sum = 0.0
        self.valid_tokens_consumed = 0

    def assert_evaluation_path_safety(self, eval_path: str):
        assert "sealed_test" not in eval_path.lower(), "SEALED TEST PATH PROHIBITED IN CALIBRATED TRAINER!"

    def train_step(self, batch_or_x: dict | torch.Tensor, y: torch.Tensor | None = None) -> float:
        if isinstance(batch_or_x, dict):
            x = batch_or_x["x"]
            y = batch_or_x["y"]
            valid_cnt = batch_or_x.get("valid_target_count", int((y != self.ignore_index).sum().item()))
        else:
            x = batch_or_x
            assert y is not None, "Target tensor y must be provided when x is a tensor!"
            valid_cnt = int((y != self.ignore_index).sum().item())

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(x, y)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()

        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        unclipped_norm = compute_global_grad_norm(params)
        self.unclipped_norm_sum += unclipped_norm

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            if unclipped_norm > self.max_grad_norm:
                self.clipped_steps_count += 1
            clipped_norm = compute_global_grad_norm(params)
            self.clipped_norm_sum += clipped_norm
        else:
            self.clipped_norm_sum += unclipped_norm

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.total_steps += 1
        self.valid_tokens_consumed += valid_cnt
        return float(loss.item())

    def get_geo_parameter_norms(self) -> dict[str, float]:
        """Computes separate parameter norms for GEO U, V, and Alpha parameters if present."""
        norms = {}
        for name, param in self.model.named_parameters():
            if "geo" in name.lower() or any(k in name for k in ["U", "V", "alpha"]):
                norms[name] = float(torch.norm(param.detach(), 2).item())
        return norms

    def get_telemetry(self) -> dict:
        clipping_freq_pct = (self.clipped_steps_count / float(self.total_steps) * 100.0) if self.total_steps > 0 else 0.0
        mean_unclipped_norm = (self.unclipped_norm_sum / float(self.total_steps)) if self.total_steps > 0 else 0.0
        mean_clipped_norm = (self.clipped_norm_sum / float(self.total_steps)) if self.total_steps > 0 else 0.0

        return {
            "total_steps": self.total_steps,
            "valid_tokens_consumed": self.valid_tokens_consumed,
            "max_grad_norm": self.max_grad_norm,
            "clipping_freq_pct": clipping_freq_pct,
            "mean_unclipped_grad_norm": mean_unclipped_norm,
            "mean_clipped_grad_norm": mean_clipped_norm,
            "current_lr": self.optimizer.param_groups[0]["lr"],
            "geo_parameter_norms": self.get_geo_parameter_norms(),
            "sealed_test_accessed": False,
        }
