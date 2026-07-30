from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


class CalibrationCheckpointManager:
    """Manages high-resolution evaluation checkpoints (including step 0) with strict domain-validation selection rules and full state restoration."""

    RESOLUTIONS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000]

    def __init__(
        self,
        checkpoint_dir: Path,
        variant_name: str,
        seed: int,
        calibration_config: dict | None = None,
        repository_commits: dict | None = None,
        tokenizer_sha256: str = "",
        split_hashes: dict | None = None,
    ):
        assert "sealed_test" not in checkpoint_dir.name.lower(), "SEALED TEST ACCESS PROHIBITED IN CHECKPOINT MANAGER!"

        self.checkpoint_dir = checkpoint_dir
        self.variant_name = variant_name
        self.seed = seed
        self.calibration_config = calibration_config or {}
        self.repository_commits = repository_commits or {}
        self.tokenizer_sha256 = tokenizer_sha256
        self.split_hashes = split_hashes or {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_domain_nll = float("inf")
        self.best_general_nll = float("inf")
        self.best_step = 0
        self.best_checkpoint_path: Path | None = None
        self.best_checkpoint_hash: str | None = None

    def is_checkpoint_step(self, step: int) -> bool:
        return step in self.RESOLUTIONS

    def evaluate_and_maybe_save(
        self,
        step: int,
        domain_nll: float,
        general_nll: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None = None,
        tokens_consumed: int = 0,
    ) -> bool:
        """Evaluates domain/general NLL and saves state_dict directly if a new minimum is achieved."""
        improved = False
        tie_tol = 1e-6

        # Selection rule: Primary lowest domain NLL, Tie-breaker 1: lowest general NLL, Tie-breaker 2: earliest step
        if (domain_nll < self.best_domain_nll - tie_tol) or (
            math.isclose(domain_nll, self.best_domain_nll, abs_tol=tie_tol) and general_nll < self.best_general_nll - tie_tol
        ):
            self.best_domain_nll = domain_nll
            self.best_general_nll = general_nll
            self.best_step = step
            improved = True

            ckpt_path = self.checkpoint_dir / f"{self.variant_name}_seed{self.seed}_best.pt"

            state_payload = {
                "variant_name": self.variant_name,
                "seed": self.seed,
                "global_step": step,
                "tokens_consumed": tokens_consumed,
                "domain_nll": domain_nll,
                "general_nll": general_nll,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler and hasattr(scheduler, "state_dict") else None,
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "calibration_config": self.calibration_config,
                "repository_commits": self.repository_commits,
                "tokenizer_sha256": self.tokenizer_sha256,
                "split_hashes": self.split_hashes,
            }

            torch.save(state_payload, ckpt_path)
            self.best_checkpoint_path = ckpt_path
            self.best_checkpoint_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

        return improved

    @staticmethod
    def restore_checkpoint(
        ckpt_path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
    ) -> dict:
        """Restores exact model, optimizer, scheduler, and RNG states from disk."""
        assert ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model.load_state_dict(payload["model_state_dict"])

        if optimizer and "optimizer_state_dict" in payload and payload["optimizer_state_dict"]:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in payload and payload["scheduler_state_dict"]:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        if "torch_cpu_rng_state" in payload:
            torch.set_rng_state(payload["torch_cpu_rng_state"])
        if "torch_cuda_rng_state" in payload and payload["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(payload["torch_cuda_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])

        return payload
