from __future__ import annotations

import math

import torch
import torch.nn as nn

from .backends.base import GeoBackend
from .config import ModelConfig


class GeoEmbedding(nn.Module):
    def __init__(self, backend: GeoBackend, vocabulary: int, dimension: int) -> None:
        super().__init__()
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(vocabulary, dimension))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.backend.embedding(indices, self.weight)


class GeoLinear(nn.Module):
    def __init__(self, backend: GeoBackend, in_features: int, out_features: int) -> None:
        super().__init__()
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.linear(x, self.weight)


class GeoRMSNorm(nn.Module):
    def __init__(self, backend: GeoBackend, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.backend = backend
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.rms_norm(x, self.weight, self.eps)


class GeoAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: GeoBackend) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.qkv = GeoLinear(backend, cfg.d_model, 3 * cfg.d_model)
        self.proj = GeoLinear(backend, cfg.d_model, cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        heads, head_dim = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        q = self.backend.apply_rope(q, cos, sin)
        k = self.backend.apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        out = self.backend.causal_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().reshape(batch, tokens, channels)
        return self.proj(out)


class GeoSwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: GeoBackend) -> None:
        super().__init__()
        self.backend = backend
        self.gate = GeoLinear(backend, cfg.d_model, cfg.ffn_hidden)
        self.up = GeoLinear(backend, cfg.d_model, cfg.ffn_hidden)
        self.down = GeoLinear(backend, cfg.ffn_hidden, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.backend.silu_mul(self.gate(x), self.up(x)))


class GeoBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: GeoBackend) -> None:
        super().__init__()
        self.backend = backend
        self.attn_norm = GeoRMSNorm(backend, cfg.d_model)
        self.attn = GeoAttention(cfg, backend)
        self.ffn_norm = GeoRMSNorm(backend, cfg.d_model)
        self.ffn = GeoSwiGLU(cfg, backend)
        self.ple_gate = GeoLinear(backend, cfg.d_model, cfg.ple_dim)
        self.ple_proj = GeoLinear(backend, cfg.ple_dim, cfg.d_model)
        self.ple_norm = GeoRMSNorm(backend, cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ple: torch.Tensor,
        kv_cache: any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = self.backend.add(x, self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache, layer_idx=layer_idx))
        x = self.backend.add(x, self.ffn(self.ffn_norm(x)))
        gated = self.backend.gelu(self.ple_gate(x))
        h = self.ple_proj(self.backend.mul(gated, ple))
        return self.backend.add(x, self.ple_norm(h))


class GeoDomainLM(nn.Module):
    """ESP32-AIv2-shaped PLE model with a strict GEO math boundary."""

    def __init__(self, cfg: ModelConfig, backend: GeoBackend) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.backend = backend
        self.tok_emb = GeoEmbedding(backend, cfg.vocab_size, cfg.d_model)
        self.ple_table = GeoEmbedding(backend, cfg.vocab_size, cfg.n_layers * cfg.ple_dim)
        self.ple_model_proj = GeoLinear(backend, cfg.d_model, cfg.n_layers * cfg.ple_dim)
        self.ple_norm = GeoRMSNorm(backend, cfg.ple_dim)
        self.blocks = nn.ModuleList([GeoBlock(cfg, backend) for _ in range(cfg.n_layers)])
        self.out_norm = GeoRMSNorm(backend, cfg.d_model)
        self.head = GeoLinear(backend, cfg.d_model, cfg.vocab_size)
        self.head.weight = self.tok_emb.weight
        for block in self.blocks:
            nn.init.zeros_(block.ple_norm.weight)
        cos, sin = backend.build_rope(
            cfg.seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache: any | None = None, start_pos: int = 0):
        batch, tokens = idx.shape
        end_pos = start_pos + tokens
        if end_pos > self.cfg.seq_len:
            raise ValueError(f"sequence length {end_pos} exceeds configured limit {self.cfg.seq_len}")
        x = self.tok_emb(idx)
        projected = self.ple_model_proj(x).reshape(
            batch, tokens, self.cfg.n_layers, self.cfg.ple_dim
        )
        projected = self.ple_norm(projected)
        table = self.ple_table(idx).reshape(
            batch, tokens, self.cfg.n_layers, self.cfg.ple_dim
        )
        ple = self.backend.scale(
            self.backend.add(
                projected,
                self.backend.scale(table, math.sqrt(self.cfg.ple_dim)),
            ),
            1.0 / math.sqrt(2.0),
        )
        cos_step = self.cos[start_pos:end_pos]
        sin_step = self.sin[start_pos:end_pos]

        for layer, block in enumerate(self.blocks):
            x = block(x, cos_step, sin_step, ple[:, :, layer], kv_cache=kv_cache, layer_idx=layer)

        if kv_cache is not None:
            kv_cache.advance(tokens)

        logits = self.head(self.out_norm(x))
        loss = None
        if targets is not None:
            loss = self.backend.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), -1
            )
        return logits, loss

