from __future__ import annotations

import math
import torch
import torch.nn as nn

from geosdp.backends.base import GeoBackend


class GeoParameterizedLinear(nn.Module):
    """Geometric Parameterized Linear Layer supporting Dense, Structured GEO, and Hybrid Modes."""

    def __init__(
        self,
        backend: GeoBackend,
        in_features: int,
        out_features: int,
        mode: str = "dense",
        num_bases: int = 4,
    ):
        super().__init__()
        self.backend = backend
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.num_bases = num_bases

        if mode == "dense":
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.normal_(self.weight, std=0.02)
        elif mode == "geo_structured":
            # Compact Geometric Basis Matrices G_i and Coefficients alpha_i
            self.bases = nn.Parameter(torch.empty(num_bases, out_features, in_features))
            self.alpha = nn.Parameter(torch.ones(num_bases) / math.sqrt(num_bases))
            nn.init.normal_(self.bases, std=0.02 / math.sqrt(num_bases))
        elif mode == "hybrid_residual":
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.normal_(self.weight, std=0.02)
            self.bases = nn.Parameter(torch.empty(num_bases, out_features, in_features))
            self.alpha = nn.Parameter(torch.zeros(num_bases))
            nn.init.normal_(self.bases, std=0.01)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_effective_weight(self) -> torch.Tensor:
        """Compute full effective weight matrix from GEO bases and coefficients."""
        if self.mode == "dense":
            return self.weight
        elif self.mode == "geo_structured":
            # W = sum(alpha_i * G_i)
            return torch.sum(self.alpha.view(-1, 1, 1) * self.bases, dim=0)
        elif self.mode == "hybrid_residual":
            geo_corr = torch.sum(self.alpha.view(-1, 1, 1) * self.bases, dim=0)
            return self.weight + geo_corr
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def parameter_count(self) -> int:
        """Count total trainable parameters in this layer."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.get_effective_weight()
        return self.backend.linear(x, w_eff)
