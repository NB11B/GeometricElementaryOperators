from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, Tuple, Dict

import torch

from .base import GeoBackend


GEO_RUNTIME_ABI_VERSION = 1
GEO_TORCH_ABI_VERSION = GEO_RUNTIME_ABI_VERSION
CORE_OPERATIONS = ("linear", "add", "mul", "scale", "rms_norm")
ACTIVATION_OPERATIONS = CORE_OPERATIONS + ("gelu", "silu_mul")
POSITION_OPERATIONS = ACTIVATION_OPERATIONS + ("build_rope", "apply_rope")
ATTENTION_OPERATIONS = POSITION_OPERATIONS + ("causal_attention",)
LOSS_OPERATIONS = ATTENTION_OPERATIONS + ("cross_entropy",)
FULL_MODEL_OPERATIONS = LOSS_OPERATIONS + ("embedding", "implicit_linear")


class NativeGeoBackend(GeoBackend):
    """Adapter for GEO Deep-Learning Runtime.

    Executes GEO-owned explicit autograd on CUDA tensors.
    No missing operation falls back to an unrelated model path.
    """

    name = "native"
    native = True

    def __init__(
        self,
        module: Any | None = None,
        *,
        require_cuda: bool = True,
        required_operations: Iterable[str] = FULL_MODEL_OPERATIONS,
    ) -> None:
        self.require_cuda = require_cuda

        if module is None:
            try:
                module = import_module("geo_dl_runtime")
            except ImportError as exc:
                raise RuntimeError(
                    "Native GEO backend requested, but geo_dl_runtime is unavailable. "
                    "Ensure geo_dl_runtime is available in python path."
                ) from exc

        self.ops = module
        self.required_operations = tuple(dict.fromkeys(required_operations))

        self.telemetry = {
            "execution_kind": getattr(self.ops, "GEO_EXECUTION_KIND", "python_torch_autograd"),
            "compiled_extension_loaded": bool(getattr(self.ops, "GEO_COMPILED_EXTENSION_LOADED", False)),
            "torch_cuda_available": bool(getattr(self.ops, "GEO_CUDA_AVAILABLE", torch.cuda.is_available())),
            "runtime_abi_version": getattr(self.ops, "GEO_DL_RUNTIME_ABI_VERSION", GEO_RUNTIME_ABI_VERSION),
            "geo_owns_backward": bool(getattr(self.ops, "GEO_OWNS_BACKWARD", False)),
            "implicit_linear_calls": 0,
            "implicit_linear_native_calls": 0,
            "native_operation_calls": 0,
            "fallback_count": 0,
        }

        if module is not None:
            capabilities = frozenset(getattr(module, "GEO_CAPABILITIES", ()))
            missing = [
                name
                for name in self.required_operations
                if name not in capabilities or not callable(getattr(module, name, None))
            ]
            if missing:
                raise RuntimeError(f"GEO runtime is missing required operations: {', '.join(missing)}")

            abi_version = getattr(module, "GEO_DL_RUNTIME_ABI_VERSION", getattr(module, "GEO_TORCH_ABI_VERSION", None))
            if abi_version != GEO_RUNTIME_ABI_VERSION:
                raise RuntimeError(
                    f"GEO runtime ABI mismatch: expected {GEO_RUNTIME_ABI_VERSION}, got {abi_version!r}"
                )

            if getattr(module, "GEO_BACKEND", None) != "GeometricElementaryOperators":
                raise RuntimeError(
                    "GEO runtime does not identify NB11B/GeometricElementaryOperators as its backend"
                )

            if not bool(getattr(module, "GEO_OWNS_BACKWARD", False)):
                raise RuntimeError(
                    "GEO runtime must report GEO_OWNS_BACKWARD=True"
                )

            if require_cuda and not bool(getattr(module, "GEO_CUDA_AVAILABLE", False)):
                raise RuntimeError("GEO runtime was loaded without a usable CUDA backend")

    def reset_telemetry(self) -> None:
        self.telemetry["implicit_linear_calls"] = 0
        self.telemetry["implicit_linear_native_calls"] = 0
        self.telemetry["native_operation_calls"] = 0
        self.telemetry["fallback_count"] = 0

    def get_telemetry(self) -> Dict[str, Any]:
        return dict(self.telemetry)

    def _operation(self, name: str):
        if self.ops is None:
            raise RuntimeError("geo_dl_runtime module is not loaded")
        operation = getattr(self.ops, name, None)
        if not callable(operation):
            raise RuntimeError(f"Native GEO operation {name!r} is unavailable")
        return operation

    def embedding(self, indices, weight):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("embedding")(indices, weight)

    def linear(self, x, weight):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("linear")(x, weight)

    def add(self, a, b):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("add")(a, b)

    def mul(self, a, b):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("mul")(a, b)

    def scale(self, x, value):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("scale")(x, float(value))

    def rms_norm(self, x, weight, eps):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("rms_norm")(x, weight, float(eps))

    def build_rope(
        self,
        seq_len: int,
        head_dim: int,
        theta: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.telemetry["native_operation_calls"] += 1
        return self._operation("build_rope")(seq_len, head_dim, float(theta), str(device))

    def apply_rope(self, x, cos, sin):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("apply_rope")(x, cos, sin)

    def causal_attention(self, q, k, v):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("causal_attention")(q, k, v)

    def silu_mul(self, gate, up):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("silu_mul")(gate, up)

    def gelu(self, x):
        self.telemetry["native_operation_calls"] += 1
        return self._operation("gelu")(x)

    def cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        self.telemetry["native_operation_calls"] += 1
        return self._operation("cross_entropy")(
            logits,
            targets,
            int(ignore_index),
        )

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
        operation = self._operation("implicit_linear")
        result = operation(x, u, v, alpha, perm_indices, inv_perm_indices, sign_mask)
        self.telemetry["implicit_linear_calls"] += 1
        self.telemetry["implicit_linear_native_calls"] += 1
        self.telemetry["native_operation_calls"] += 1
        return result
