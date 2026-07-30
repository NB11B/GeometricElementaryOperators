from __future__ import annotations

import math
import torch
import torch.nn as nn

from geosdp import ModelConfig, GeoDomainLM
from geosdp.backends.base import GeoBackend
from geosdp.backends.native import NativeGeoBackend
from geosdp.models.geo_implicit_linear import GeoImplicitStructuredLinear, GeoImplicitSparseResidualLinear


class StandardLowRankLinear(nn.Module):
    """Sequential UV^T low-rank projection layer matching GEO rank and dimensions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.u = nn.Parameter(torch.randn(rank, out_features, device=device, dtype=dtype) / math.sqrt(out_features))
        self.v = nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) / math.sqrt(in_features))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(torch.matmul(x_in, self.v.T.to(x_in.dtype)), self.u.to(x_in.dtype))
        return out.to(x_in.dtype)


def build_decoder_variant(
    variant_name: str,
    cfg: ModelConfig,
    backend: GeoBackend,
    rank: int = 4,
    sparsity: float = 0.05,
    device: torch.device | None = None,
) -> tuple[GeoDomainLM, dict]:
    """Centralized builder for decoder variants enforcing the complete 16-layer replacement map."""
    model = GeoDomainLM(cfg, backend).to(device)
    replacement_map = {}

    if variant_name == "GeoDenseDecoder":
        pass  # Standard dense baseline

    elif variant_name == "StandardLowRankDecoder":
        for idx, block in enumerate(model.blocks):
            block.attn.proj = StandardLowRankLinear(cfg.d_model, cfg.d_model, rank=rank, device=device)
            replacement_map[f"blocks.{idx}.attn.proj"] = {
                "in_features": cfg.d_model, "out_features": cfg.d_model, "rank": rank, "operator": "low_rank"
            }

            block.ffn.down = StandardLowRankLinear(cfg.ffn_hidden, cfg.d_model, rank=rank, device=device)
            replacement_map[f"blocks.{idx}.ffn.down"] = {
                "in_features": cfg.ffn_hidden, "out_features": cfg.d_model, "rank": rank, "operator": "low_rank"
            }

    elif variant_name == "GeoCompactDecoder":
        for idx, block in enumerate(model.blocks):
            block.attn.proj = GeoImplicitStructuredLinear(backend, cfg.d_model, cfg.d_model, rank=rank, device=device)
            replacement_map[f"blocks.{idx}.attn.proj"] = {
                "in_features": cfg.d_model, "out_features": cfg.d_model, "rank": rank, "operator": "native_geo_compact"
            }

            block.ffn.down = GeoImplicitStructuredLinear(backend, cfg.ffn_hidden, cfg.d_model, rank=rank, device=device)
            replacement_map[f"blocks.{idx}.ffn.down"] = {
                "in_features": cfg.ffn_hidden, "out_features": cfg.d_model, "rank": rank, "operator": "native_geo_compact"
            }

    elif variant_name == "GeoSparseResidualDecoder":
        for idx, block in enumerate(model.blocks):
            block.attn.proj = GeoImplicitSparseResidualLinear(backend, cfg.d_model, cfg.d_model, rank=rank, sparsity=sparsity, device=device)
            replacement_map[f"blocks.{idx}.attn.proj"] = {
                "in_features": cfg.d_model, "out_features": cfg.d_model, "rank": rank, "sparsity": sparsity, "operator": "native_geo_sparse"
            }

            block.ffn.down = GeoImplicitSparseResidualLinear(backend, cfg.ffn_hidden, cfg.d_model, rank=rank, sparsity=sparsity, device=device)
            replacement_map[f"blocks.{idx}.ffn.down"] = {
                "in_features": cfg.ffn_hidden, "out_features": cfg.d_model, "rank": rank, "sparsity": sparsity, "operator": "native_geo_sparse"
            }

    else:
        raise ValueError(f"Unknown decoder variant name: {variant_name}")

    return model, replacement_map


def introspect_decoder_parameters(model: GeoDomainLM) -> dict:
    """Detailed parameter and byte breakdown by decoder subsystem."""
    breakdown = {
        "embedding_lm_head": 0,
        "attention": 0,
        "ffn": 0,
        "normalization": 0,
        "geo_factors": 0,
        "sparse_residual": 0,
        "other": 0,
        "total_trainable": 0,
        "total_persistent_bytes": 0,
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        p_bytes = numel * p.element_size()
        breakdown["total_trainable"] += numel
        breakdown["total_persistent_bytes"] += p_bytes

        if "tok_emb" in name or "head" in name or "ple_table" in name:
            breakdown["embedding_lm_head"] += numel
        elif "geo_implicit" in name or ".u" in name or ".v" in name or ".alpha" in name:
            breakdown["geo_factors"] += numel
        elif "sparse_val" in name:
            breakdown["sparse_residual"] += numel
        elif ".attn." in name:
            breakdown["attention"] += numel
        elif ".ffn." in name:
            breakdown["ffn"] += numel
        elif "norm" in name:
            breakdown["normalization"] += numel
        else:
            breakdown["other"] += numel

    # Add buffer bytes
    for name, b in model.named_buffers():
        breakdown["total_persistent_bytes"] += b.numel() * b.element_size()

    return breakdown
