from __future__ import annotations

import math
import torch
import torch.nn as nn

from geosdp.backends.base import GeoBackend


class GeoCompactStructuredLinear(nn.Module):
    """Truly Compact Sub-Dense Geometric Parameterized Linear Layer.
    
    W_GEO = sum_{i=1}^k alpha_i * (P_i * (u_i @ v_i^T))
    
    Provides > 10x to 60x parameter compression relative to a dense linear layer.
    """

    def __init__(
        self,
        backend: GeoBackend,
        in_features: int,
        out_features: int,
        rank: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.backend = backend
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank factor vectors u_i [rank, out_features] and v_i [rank, in_features]
        self.u = nn.Parameter(torch.empty(rank, out_features, device=device, dtype=dtype))
        self.v = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.alpha = nn.Parameter(torch.ones(rank, device=device, dtype=dtype) / math.sqrt(rank))

        nn.init.normal_(self.u, std=1.0 / math.sqrt(out_features))
        nn.init.normal_(self.v, std=1.0 / math.sqrt(in_features))

        # Fixed sign-permutation bases P_i [rank, out_features, in_features]
        # P_i is fixed and non-trainable (zero parameter overhead)
        gen = torch.Generator(device=device if device else torch.device("cpu")).manual_seed(42)
        p_bases = torch.randint(0, 2, (rank, out_features, in_features), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0
        self.register_buffer("P_bases", p_bases, persistent=False)

    @property
    def dense_parameter_count(self) -> int:
        """Parameters of an equivalent dense matrix."""
        return self.out_features * self.in_features

    @property
    def trainable_parameter_count(self) -> int:
        """Actual trainable parameters in this compact GEO layer."""
        return sum(p.numel() for p in self.parameters())

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to dense matrix."""
        return self.dense_parameter_count / max(1, self.trainable_parameter_count)

    def get_effective_weight(self) -> torch.Tensor:
        """Compute full effective weight matrix: sum(alpha_i * P_i * (u_i @ v_i^T))."""
        # u: [rank, out], v: [rank, in] -> rank outer products: [rank, out, in]
        outer = torch.bmm(self.u.unsqueeze(2), self.v.unsqueeze(1))
        # Elementwise multiplication with fixed sign-permutation P_bases
        structured_bases = self.P_bases * outer
        # Weighted sum by alpha_i
        w_eff = torch.sum(self.alpha.view(-1, 1, 1) * structured_bases, dim=0)
        return w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.get_effective_weight()
        return self.backend.linear(x, w_eff)
