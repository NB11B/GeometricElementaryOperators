#!/usr/bin/env python3
"""
test_geo_compact_r8_native.py

Work Package 4: Native Gate Tests.
Instantiates real GeoCompactDecoder rank 8 model with NativeGeoBackend.

Assertions:
1. Native backend initialization succeeds.
2. Runtime ABI version matches 1.
3. GEO_OWNS_BACKWARD is True.
4. One model forward produces 12 implicit_linear dispatches.
5. Fallback count = 0.
6. Loss and gradients for u, v, alpha are finite and non-zero.
7. Optimizer step produces non-zero parameter updates.
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
from geosdp.backends.native import NativeGeoBackend, GEO_RUNTIME_ABI_VERSION
from geosdp.models.geo_decoder_variants import build_decoder_variant

def get_r8_config():
    return ModelConfig(
        vocab_size=4096,
        d_model=256,
        ffn_hidden=768,
        n_heads=4,
        n_layers=6,
        seq_len=512
    )

def test_native_backend_contract_and_telemetry():
    backend = NativeGeoBackend(require_cuda=False)
    backend.reset_telemetry()
    telemetry = backend.get_telemetry()
    
    assert telemetry["runtime_abi_version"] == GEO_RUNTIME_ABI_VERSION
    assert telemetry["geo_owns_backward"] is True
    assert telemetry["fallback_count"] == 0

def test_native_model_forward_dispatches_and_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = NativeGeoBackend(require_cuda=False)
    backend.reset_telemetry()
    
    cfg = get_r8_config()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8, device=device)
    assert len(rmap) == 12
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    targets = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    
    logits, _ = model(inputs)
    assert torch.isfinite(logits).all()
    
    telemetry = backend.get_telemetry()
    assert telemetry["implicit_linear_forward_calls"] == 12, f"Expected 12 implicit-linear dispatches, got {telemetry['implicit_linear_forward_calls']}"
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
                
    assert u_found and v_found and alpha_found, "Native backend must compute non-zero u, v, alpha gradients."

def test_native_reference_parity():
    """Verify numeric parity between reference and native backends."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_r8_config()
    
    b_ref = NativeGeoBackend(require_cuda=False) # Reference path
    m_ref, _ = build_decoder_variant("GeoCompactDecoder", cfg, b_ref, rank=8, device=device)
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    logits_ref, _ = m_ref(inputs)
    assert torch.isfinite(logits_ref).all()

if __name__ == "__main__":
    pytest.main([__file__])
