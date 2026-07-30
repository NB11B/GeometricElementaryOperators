from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TelemetryLogger:
    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                self.log_path.unlink()

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_seen: int,
        step_time_ms: float,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, Any]:
        param_norm = 0.0
        grad_norm = 0.0
        if model is not None:
            total_p_norm = 0.0
            total_g_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    param_norm_sq = p.detach().norm(2).item() ** 2
                    total_p_norm += param_norm_sq
                    if p.grad is not None:
                        total_g_norm += p.grad.detach().norm(2).item() ** 2
            param_norm = total_p_norm ** 0.5
            grad_norm = total_g_norm ** 0.5

        clipped_grad_norm = min(grad_norm, grad_clip) if grad_clip is not None else grad_norm

        gpu_alloc = 0
        gpu_res = 0
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated(device)
            gpu_res = torch.cuda.memory_reserved(device)

        tokens_per_sec = (tokens_seen / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0.0

        record = {
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": round(loss, 6),
            "learning_rate": lr,
            "gradient_norm": round(grad_norm, 6),
            "clipped_gradient_norm": round(clipped_grad_norm, 6),
            "parameter_norm": round(param_norm, 6),
            "tokens_per_second": round(tokens_per_sec, 2),
            "step_time_ms": round(step_time_ms, 2),
            "gpu_allocated_bytes": gpu_alloc,
            "gpu_reserved_bytes": gpu_res,
            "nonfinite_gradients": 0 if torch.isfinite(torch.tensor(grad_norm)) else 1,
        }

        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record
