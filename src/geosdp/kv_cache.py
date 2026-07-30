from __future__ import annotations

import torch
import torch.nn as nn


class GeoKVCache:
    """Production Key-Value cache engine for fast autoregressive decoding in GEO models."""

    def __init__(self, n_layers: int, batch_size: int, n_heads: int, max_seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.seq_len: int = 0

        self.reset()

    def reset(self):
        """Reset and pre-allocate cache memory slots across all layers."""
        self.k_cache = [
            torch.zeros(self.batch_size, self.n_heads, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        self.v_cache = [
            torch.zeros(self.batch_size, self.n_heads, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        self.seq_len = 0

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a given layer with new Key and Value tensors.
        
        Args:
            layer_idx: Index of transformer layer [0..n_layers-1]
            k_new: New key tensor of shape [B, H, T_new, D]
            v_new: New value tensor of shape [B, H, T_new, D]
            
        Returns:
            Tuple of (full_cached_k, full_cached_v) up to current active sequence length.
        """
        B, H, T_new, D = k_new.shape
        start_pos = self.seq_len
        end_pos = start_pos + T_new

        if end_pos > self.max_seq_len:
            raise ValueError(f"KV cache overflow: sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}")

        self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_new
        self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_new

        # Return active cached slice [0..end_pos]
        return (
            self.k_cache[layer_idx][:, :, :end_pos, :],
            self.v_cache[layer_idx][:, :, :end_pos, :]
        )

    def advance(self, num_tokens: int):
        """Advance current active sequence length after updating all layers for a step."""
        self.seq_len += num_tokens
