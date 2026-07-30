from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class GeoBackend(ABC):
    """All model mathematics crosses this boundary."""

    name: str
    native: bool

    @abstractmethod
    def embedding(self, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def scale(self, x: torch.Tensor, value: float) -> torch.Tensor: ...
    @abstractmethod
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...
    @abstractmethod
    def build_rope(self, seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def gelu(self, x: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor: ...
    @abstractmethod
    def implicit_linear(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        perm_indices: torch.Tensor,
        inv_perm_indices: torch.Tensor,
        sign_mask: torch.Tensor,
    ) -> torch.Tensor: ...
