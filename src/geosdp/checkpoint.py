from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


SCHEMA_VERSION = "1.0"
CHECKPOINT_FORMAT_VERSION = "1.0"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: Any,
    config: ModelConfig,
    tokens_seen: int = 0,
    dataset_position: int = 0,
    tokenizer_identity: str = "synthetic_32k",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_dir / f"checkpoint_step_{step}.pt.tmp"
    final_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "tokens_seen": tokens_seen,
        "dataset_position": dataset_position,
        "tokenizer_identity": tokenizer_identity,
        "config": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "ffn_hidden": config.ffn_hidden,
            "seq_len": config.seq_len,
            "ple_dim": config.ple_dim,
            "rope_theta": config.rope_theta,
            "dropout": config.dropout,
        },
        "model_state_dict": model.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        "extra_metadata": extra_metadata or {},
    }

    if hasattr(optimizer, "state_dict"):
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, temp_path)
    if final_path.exists():
        final_path.unlink()
    temp_path.rename(final_path)

    # Write checksum file
    sha256 = compute_file_sha256(final_path)
    meta_path = checkpoint_dir / f"checkpoint_step_{step}.json"
    meta_data = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "tokens_seen": tokens_seen,
        "dataset_position": dataset_position,
        "tokenizer_identity": tokenizer_identity,
        "file": final_path.name,
        "sha256": sha256,
        "bytes": final_path.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
    return final_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    expected_config: Optional[ModelConfig] = None,
) -> Tuple[int, ModelConfig, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Verify checksum if .json metadata exists
    meta_path = checkpoint_path.parent / f"{checkpoint_path.stem}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        current_sha256 = compute_file_sha256(checkpoint_path)
        if current_sha256 != meta.get("sha256"):
            raise ValueError(f"Checkpoint corruption detected in {checkpoint_path}")

    map_location = device if device is not None else "cpu"
    state = torch.load(checkpoint_path, map_location=map_location)

    schema_version = state.get("schema_version", "1.0")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible checkpoint schema_version: expected {SCHEMA_VERSION}, got {schema_version}"
        )

    cfg_dict = state["config"]
    config = ModelConfig(**cfg_dict)

    if expected_config is not None:
        if (
            config.vocab_size != expected_config.vocab_size
            or config.d_model != expected_config.d_model
            or config.n_layers != expected_config.n_layers
            or config.n_heads != expected_config.n_heads
        ):
            raise ValueError(
                f"Checkpoint config mismatch: saved {config} vs expected {expected_config}"
            )

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    rng = state.get("rng_state", {})
    if "python" in rng and rng["python"]:
        random.setstate(rng["python"])
    if "torch_cpu" in rng and rng["torch_cpu"] is not None:
        torch.set_rng_state(rng["torch_cpu"].cpu())
    if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["torch_cuda"].cpu())

    step = int(state["step"])
    extra = state.get("extra_metadata", {})
    extra["tokens_seen"] = state.get("tokens_seen", 0)
    extra["dataset_position"] = state.get("dataset_position", 0)
    extra["tokenizer_identity"] = state.get("tokenizer_identity", "synthetic_32k")
    return step, config, extra
    return step, config, extra
