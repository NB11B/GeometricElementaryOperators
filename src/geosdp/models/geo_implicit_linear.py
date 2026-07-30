from __future__ import annotations

import math
import torch
import torch.nn as nn
from geosdp.backends.base import GeoBackend


class GeoImplicitStructuredLinear(nn.Module):
    """Native Low-Rank Implicit Linear Projection Layer.
    
    Computes direct low-rank implicit transformation y = (x @ V^T @ U) * alpha
    without constructing dense W matrix. Supports native C/CUDA runtime dispatch
    with fallback to PyTorch tensor operations.
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

        self.u = nn.Parameter(torch.randn(rank, out_features, device=device, dtype=dtype) / math.sqrt(out_features))
        self.v = nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) / math.sqrt(in_features))
        self.alpha = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        # Structured permutation buffers
        perm_indices = torch.randperm(in_features, device=device, dtype=torch.int32)
        inv_perm_indices = torch.zeros_like(perm_indices)
        inv_perm_indices[perm_indices] = torch.arange(in_features, device=device, dtype=torch.int32)
        sign_mask = torch.where(torch.rand(in_features, device=device) > 0.5, 1, -1).to(torch.int8)

        self.register_buffer("perm_indices", perm_indices, persistent=False)
        self.register_buffer("inv_perm_indices", inv_perm_indices, persistent=False)
        self.register_buffer("sign_mask", sign_mask, persistent=False)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def persistent_bytes(self) -> int:
        """Total memory footprint of trainable parameters and compact 1D buffers."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = self.perm_indices.numel() * self.perm_indices.element_size() + self.inv_perm_indices.numel() * self.inv_perm_indices.element_size() + self.sign_mask.numel() * self.sign_mask.element_size()
        return param_bytes + buffer_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct implicit geometric forward pass via native runtime without constructing weight matrix W."""
        try:
            import geo_dl_runtime
            return geo_dl_runtime.implicit_linear(x, self.u, self.v, self.alpha, self.perm_indices, self.inv_perm_indices, self.sign_mask)
        except ImportError:
            # Reference PyTorch evaluation path when C/CUDA runtime module is not compiled
            out = torch.matmul(torch.matmul(x, self.v.T.to(x.dtype)), self.u.to(x.dtype))
            return (out * self.alpha).to(x.dtype)


class GeoImplicitSparseResidualLinear(nn.Module):
    """Implicit Structured Linear + Real Sparse Residual Matrix (S).
    
    y = GeoImplicitStructuredLinear(x) + (x @ S_sparse^T)
    """

    def __init__(
        self,
        backend: GeoBackend,
        in_features: int,
        out_features: int,
        rank: int = 4,
        sparsity: float = 0.05,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.implicit = GeoImplicitStructuredLinear(backend, in_features, out_features, rank=rank, device=device, dtype=dtype)
        
        # Build initial sparse mask
        total_elements = out_features * in_features
        num_nonzeros = int(total_elements * sparsity)
        indices = torch.randint(0, total_elements, (num_nonzeros,), device=device)
        rows = indices // in_features
        cols = indices % in_features
        sparse_indices = torch.stack([rows, cols])
        values = torch.randn(num_nonzeros, device=device, dtype=dtype) * 0.01
        
        self.sparse_weight = nn.Parameter(torch.sparse_coo_tensor(sparse_indices, values, (out_features, in_features), device=device, dtype=dtype).coalesce())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        implicit_out = self.implicit(x)
        sparse_out = torch.matmul(x, self.sparse_weight.to_dense().T.to(x.dtype))
        return implicit_out + sparse_out
