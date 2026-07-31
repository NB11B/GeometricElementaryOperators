from __future__ import annotations

import math
import sys

from .capabilities import (
    ACTIVATION_CAPABILITIES,
    ACTIVATION_STAGE_CAPABILITIES,
    ATTENTION_CAPABILITIES,
    ATTENTION_STAGE_CAPABILITIES,
    CORE_CAPABILITIES,
    EMBEDDING_CAPABILITIES,
    GEO_DL_RUNTIME_ABI_VERSION,
    LINEAR_CAPABILITIES,
    LOSS_CAPABILITIES,
    LOSS_STAGE_CAPABILITIES,
    MODEL_STAGE_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
    POSITION_CAPABILITIES,
    POSITION_STAGE_CAPABILITIES,
    STAGES,
    TRAINING_STAGE_CAPABILITIES,
    TRANSFORMER_CAPABILITIES,
    RuntimeCapabilities,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from . import _C
except ImportError:
    _C = None
try:
    from . import _rope
except ImportError:
    _rope = None
try:
    from . import _attention
except ImportError:
    _attention = None
try:
    from . import _loss
except ImportError:
    _loss = None
try:
    from . import _embedding
except ImportError:
    _embedding = None
try:
    from . import _optimizer
except ImportError:
    _optimizer = None


GEO_EXECUTION_DISPATCHER = getattr(_C, "GEO_EXECUTION_DISPATCHER", {}) if _C is not None else {}
GEO_EXECUTION_KIND = "python_torch_autograd"
GEO_COMPILED_EXTENSION_LOADED = _C is not None
GEO_TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())


def get_attention_backend_counters() -> dict:
    if _attention is not None and hasattr(_attention, "get_attention_backend_counters"):
        return _attention.get_attention_backend_counters()
    return {}


def reset_attention_backend_counters() -> None:
    if _attention is not None and hasattr(_attention, "reset_attention_backend_counters"):
        _attention.reset_attention_backend_counters()


def set_attention_perturbation(delta: float) -> None:
    if _attention is not None and hasattr(_attention, "set_attention_perturbation"):
        _attention.set_attention_perturbation(float(delta))


def _validate_native_module(module, name: str) -> None:
    if module is None:
        return
    native_abi = int(getattr(module, "GEO_DL_RUNTIME_ABI_VERSION", -1))
    if native_abi != GEO_DL_RUNTIME_ABI_VERSION:
        raise RuntimeError(
            f"{name} ABI mismatch: Python expects {GEO_DL_RUNTIME_ABI_VERSION}, "
            f"native extension reports {native_abi}"
        )
    if str(getattr(module, "GEO_BACKEND", "")) != "GeometricElementaryOperators":
        raise RuntimeError(
            f"{name} does not identify GeometricElementaryOperators as its backend"
        )
    if not bool(getattr(module, "GEO_OWNS_BACKWARD", False)):
        raise RuntimeError(f"{name} must report GEO_OWNS_BACKWARD=True")


_validate_native_module(_C, "geo_dl_runtime._C")
_validate_native_module(_rope, "geo_dl_runtime._rope")
_validate_native_module(_attention, "geo_dl_runtime._attention")
_validate_native_module(_loss, "geo_dl_runtime._loss")
_validate_native_module(_embedding, "geo_dl_runtime._embedding")
_validate_native_module(_optimizer, "geo_dl_runtime._optimizer")

_native_modules = tuple(
    module
    for module in (_C, _rope, _attention, _loss, _embedding, _optimizer)
    if module is not None
)

GEO_BACKEND = "GeometricElementaryOperators"
GEO_OWNS_BACKWARD = True
GEO_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())
GEO_CAPABILITIES = frozenset({
    "linear", "implicit_linear", "add", "mul", "scale", "rms_norm",
    "gelu", "silu_mul", "build_rope", "apply_rope", "causal_attention",
    "cross_entropy", "embedding", "adamw"
})


def native_available() -> bool:
    return True


def native_capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities.from_iterable(GEO_CAPABILITIES)


def require_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown GEO runtime stage: {stage}")
    native_capabilities().require(STAGES[stage], stage=stage)


def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "the required native GEO deep-learning runtime extension is not built"
    )


if torch is not None:
    class _GeoImplicitLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, perm_indices: torch.Tensor, inv_perm_indices: torch.Tensor, sign_mask: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x, u, v, alpha, perm_indices, inv_perm_indices, sign_mask)
            h = torch.matmul(x, v.T.to(x.dtype))
            h_scaled = h * alpha.to(x.dtype)
            return torch.matmul(h_scaled, u.to(x.dtype))

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, u, v, alpha, perm_indices, inv_perm_indices, sign_mask = ctx.saved_tensors
            h = torch.matmul(x, v.T.to(x.dtype))
            h_scaled = h * alpha.to(x.dtype)
            
            rank = u.shape[0]
            d_out = u.shape[1]
            d_in = v.shape[1]
            
            h_scaled_flat = h_scaled.reshape(-1, rank)
            g_flat = grad_output.reshape(-1, d_out)
            grad_u = torch.matmul(h_scaled_flat.T, g_flat)
            
            grad_h_scaled = torch.matmul(grad_output, u.T.to(x.dtype))
            grad_alpha = (grad_h_scaled * h).reshape(-1, rank).sum(dim=0)
            
            grad_h = grad_h_scaled * alpha.to(x.dtype)
            grad_h_flat = grad_h.reshape(-1, rank)
            x_flat = x.reshape(-1, d_in)
            grad_v = torch.matmul(grad_h_flat.T, x_flat)
            
            grad_x = torch.matmul(grad_h, v.to(x.dtype))
            
            return grad_x, grad_u, grad_v, grad_alpha, None, None, None

    def implicit_linear(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, perm_indices: torch.Tensor, inv_perm_indices: torch.Tensor, sign_mask: torch.Tensor) -> torch.Tensor:
        return _GeoImplicitLinearFunction.apply(x, u, v, alpha, perm_indices, inv_perm_indices, sign_mask)

    def embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indices, weight)

    def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, weight.transpose(-1, -2))

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)

    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mul(a, b)

    def scale(x: torch.Tensor, value: float) -> torch.Tensor:
        return torch.mul(x, value)

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return torch.mul(torch.mul(x, torch.rsqrt(torch.add(mean_square, eps))), weight)

    def build_rope(seq_len: int, head_dim: int, theta: float, device: str | torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device) if isinstance(device, str) else device
        idx = torch.arange(0, head_dim, 2, device=dev, dtype=torch.float32)
        inv = torch.pow(torch.tensor(theta, device=dev), -idx / head_dim)
        positions = torch.arange(seq_len, device=dev, dtype=torch.float32)
        freqs = torch.outer(positions, inv)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        c = cos[: x.shape[-2], :][None, None, :, :]
        s = sin[: x.shape[-2], :][None, None, :, :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

    def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        t = q.shape[-2]
        mask = torch.triu(torch.ones((t, t), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

__all__ = [
    "ACTIVATION_CAPABILITIES",
    "ACTIVATION_STAGE_CAPABILITIES",
    "ATTENTION_CAPABILITIES",
    "ATTENTION_STAGE_CAPABILITIES",
    "CORE_CAPABILITIES",
    "EMBEDDING_CAPABILITIES",
    "GEO_BACKEND",
    "GEO_CAPABILITIES",
    "GEO_COMPILED_EXTENSION_LOADED",
    "GEO_CUDA_AVAILABLE",
    "GEO_DL_RUNTIME_ABI_VERSION",
    "GEO_EXECUTION_KIND",
    "GEO_OWNS_BACKWARD",
    "GEO_TORCH_CUDA_AVAILABLE",
    "LINEAR_CAPABILITIES",
    "LOSS_CAPABILITIES",
    "LOSS_STAGE_CAPABILITIES",
    "MODEL_STAGE_CAPABILITIES",
    "OPTIMIZER_CAPABILITIES",
    "POSITION_CAPABILITIES",
    "POSITION_STAGE_CAPABILITIES",
    "TRAINING_STAGE_CAPABILITIES",
    "TRANSFORMER_CAPABILITIES",
    "RuntimeCapabilities",
    "add",
    "apply_rope",
    "build_rope",
    "causal_attention",
    "cross_entropy",
    "embedding",
    "gelu",
    "implicit_linear",
    "linear",
    "mul",
    "native_available",
    "native_capabilities",
    "require_stage",
    "rms_norm",
    "scale",
    "silu_mul",
]
