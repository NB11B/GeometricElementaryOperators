from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import GeoBackend


class ReferenceGeoBackend(GeoBackend):
    """Correctness-only backend; never permitted for production training."""

    name = "reference"
    native = False

    def embedding(self, indices, weight): return F.embedding(indices, weight)
    def linear(self, x, weight): return torch.matmul(x, weight.transpose(-1, -2))
    def add(self, a, b): return torch.add(a, b)
    def mul(self, a, b): return torch.mul(a, b)
    def scale(self, x, value): return torch.mul(x, value)

    def rms_norm(self, x, weight, eps):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return torch.mul(torch.mul(x, torch.rsqrt(torch.add(mean_square, eps))), weight)

    def build_rope(self, seq_len, head_dim, theta, device) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inv = torch.pow(torch.tensor(theta, device=device), -idx / head_dim)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(self, x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        c = cos[: x.shape[-2], :][None, None, :, :]
        s = sin[: x.shape[-2], :][None, None, :, :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

    def causal_attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        t = q.shape[-2]
        mask = torch.triu(torch.ones((t, t), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def silu_mul(self, gate, up): return F.silu(gate) * up
    def gelu(self, x): return F.gelu(x)
    def cross_entropy(self, logits, targets, ignore_index=-1):
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    def implicit_linear(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        perm_indices: torch.Tensor,
        inv_perm_indices: torch.Tensor,
        sign_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.matmul(x, v.T.to(x.dtype))
        h_scaled = h * alpha.to(x.dtype)
        out = torch.matmul(h_scaled, u.to(x.dtype))
        return out
