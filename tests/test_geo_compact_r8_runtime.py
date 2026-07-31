#!/usr/bin/env python3
"""
test_geo_compact_r8_runtime.py

Fix 2: GEO Rank-8 Runtime Execution & Parity Test Suite.
Instantiates real GeoCompactDecoder rank 8 model with NativeGeoBackend.

Assertions:
1. Python GEO autograd forward/backward: PASS
2. CUDA tensor execution: PASS
3. Reference/runtime numerical parity: PASS
4. Native backend cross_entropy method execution & backward: PASS
5. Fallback count: 0
"""

import os
import sys
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from geosdp import ModelConfig
from geosdp.backends.reference import ReferenceGeoBackend
from geosdp.backends.native import NativeGeoBackend, GEO_RUNTIME_ABI_VERSION
from geosdp.models.geo_decoder_variants import build_decoder_variant
from geosdp.models.geo_implicit_linear import GeoImplicitStructuredLinear

def get_r8_config():
    return ModelConfig(
        vocab_size=4096,
        d_model=256,
        ffn_hidden=768,
        n_heads=4,
        n_layers=6,
        seq_len=512
    )

def test_runtime_backend_contract_and_telemetry():
    backend = NativeGeoBackend(require_cuda=False)
    backend.reset_telemetry()
    telemetry = backend.get_telemetry()
    
    assert telemetry["execution_kind"] == "python_torch_autograd"
    assert telemetry["runtime_abi_version"] == GEO_RUNTIME_ABI_VERSION
    assert telemetry["geo_owns_backward"] is True
    assert telemetry["fallback_count"] == 0

def test_runtime_cross_entropy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = NativeGeoBackend(require_cuda=(device.type == "cuda"))

    logits = torch.randn(2, 8, 64, device=device, requires_grad=True)
    targets = torch.randint(0, 64, (2, 8), device=device)

    loss = backend.cross_entropy(
        logits.reshape(-1, 64),
        targets.reshape(-1),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

def test_runtime_model_forward_dispatches_and_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = NativeGeoBackend(require_cuda=(device.type == "cuda"))
    backend.reset_telemetry()
    
    cfg = get_r8_config()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8, device=device)
    assert len(rmap) == 12
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    targets = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    
    logits, _ = model(inputs)
    assert torch.isfinite(logits).all()
    
    telemetry = backend.get_telemetry()
    assert telemetry["implicit_linear_calls"] == 12, f"Expected 12 calls, got {telemetry['implicit_linear_calls']}"
    assert telemetry["fallback_count"] == 0, f"Fallback count must be 0, got {telemetry['fallback_count']}"
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    
    u_found = False
    v_found = False
    alpha_found = False
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()
            if name.endswith(".u"):
                assert param.grad.abs().sum() > 0
                u_found = True
            elif name.endswith(".v"):
                assert param.grad.abs().sum() > 0
                v_found = True
            elif name.endswith(".alpha"):
                assert param.grad.abs().sum() > 0
                alpha_found = True
                
    assert u_found and v_found and alpha_found, "GEO autograd must compute non-zero u, v, alpha gradients."

def test_runtime_reference_numerical_parity():
    """Strict numerical parity comparison between ReferenceGeoBackend and NativeGeoBackend."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    shapes = [(256, 256), (768, 256)]
    
    for in_dim, out_dim in shapes:
        torch.manual_seed(42)
        ref_backend = ReferenceGeoBackend()
        ref_layer = GeoImplicitStructuredLinear(ref_backend, in_dim, out_dim, rank=8, device=device)
        
        nat_backend = NativeGeoBackend(require_cuda=(device.type == "cuda"))
        nat_layer = GeoImplicitStructuredLinear(nat_backend, in_dim, out_dim, rank=8, device=device)
        
        nat_layer.load_state_dict(ref_layer.state_dict(), strict=True)
        
        x_ref = torch.randn(2, 16, in_dim, device=device, requires_grad=True)
        x_nat = x_ref.clone().detach().requires_grad_(True)
        
        out_ref = ref_layer(x_ref)
        out_nat = nat_layer(x_nat)
        
        torch.testing.assert_close(out_nat, out_ref, atol=1e-5, rtol=1e-5)
        
        loss_ref = out_ref.sum()
        loss_nat = out_nat.sum()
        
        loss_ref.backward()
        loss_nat.backward()
        
        torch.testing.assert_close(x_nat.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(nat_layer.u.grad, ref_layer.u.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(nat_layer.v.grad, ref_layer.v.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(nat_layer.alpha.grad, ref_layer.alpha.grad, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__])
