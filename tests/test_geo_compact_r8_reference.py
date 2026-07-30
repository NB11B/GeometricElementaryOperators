#!/usr/bin/env python3
"""
test_geo_compact_r8_reference.py

Work Package 4: Fast Reference Gate Tests.
Instantiates real GeoCompactDecoder rank 8 model with ReferenceGeoBackend.

Assertions:
1. Actual model builds.
2. Parameter count = 8,401,888.
3. Replacement map count = 12, all with rank 8.
4. Strict checkpoint load succeeds.
5. Forward output finite.
6. Backward gradients finite.
7. u, v, alpha gradients nonzero.
8. Optimizer step changes u, v, and alpha parameters.
9. Full-model save/load preserves output.
"""

import os
import sys
import tempfile
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from geosdp import ModelConfig
from geosdp.backends.reference import ReferenceGeoBackend
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

def test_model_build_and_parameter_count():
    cfg = get_r8_config()
    backend = ReferenceGeoBackend()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 8401888, f"Expected 8,401,888 trainable params, got {total_params}"
    assert len(rmap) == 12, f"Expected 12 replaced projection layers, got {len(rmap)}"
    for layer_name, info in rmap.items():
        assert info["rank"] == 8
        assert info["operator"] == "native_geo_compact"

def test_strict_checkpoint_load():
    cfg = get_r8_config()
    backend = ReferenceGeoBackend()
    model1, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    model2, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    
    sd = model1.state_dict()
    # Strict load assertion
    model2.load_state_dict(sd, strict=True)

def test_forward_backward_and_geo_gradients():
    cfg = get_r8_config()
    backend = ReferenceGeoBackend()
    model, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    
    logits, _ = model(inputs)
    assert torch.isfinite(logits).all(), "Forward output logits must be finite."
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    
    # Assert u, v, alpha parameters Specifically receive finite, non-zero gradients
    u_found = False
    v_found = False
    alpha_found = False
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} is not finite."
            if name.endswith(".u"):
                assert param.grad.abs().sum() > 0, f"Gradient for {name} is zero."
                u_found = True
            elif name.endswith(".v"):
                assert param.grad.abs().sum() > 0, f"Gradient for {name} is zero."
                v_found = True
            elif name.endswith(".alpha"):
                assert param.grad.abs().sum() > 0, f"Gradient for {name} is zero."
                alpha_found = True
                
    assert u_found, "Expected non-zero u gradients."
    assert v_found, "Expected non-zero v gradients."
    assert alpha_found, "Expected non-zero alpha gradients."

def test_optimizer_step_updates_geo_parameters():
    cfg = get_r8_config()
    backend = ReferenceGeoBackend()
    model, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    
    # Capture initial parameters
    layer0_u_init = model.blocks[0].attn.proj.u.clone()
    layer0_v_init = model.blocks[0].attn.proj.v.clone()
    layer0_alpha_init = model.blocks[0].attn.proj.alpha.clone()
    
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    
    # Assert optimizer step updated u, v, alpha
    assert not torch.equal(model.blocks[0].attn.proj.u, layer0_u_init), "Optimizer step must update u."
    assert not torch.equal(model.blocks[0].attn.proj.v, layer0_v_init), "Optimizer step must update v."
    assert not torch.equal(model.blocks[0].attn.proj.alpha, layer0_alpha_init), "Optimizer step must update alpha."

def test_full_model_save_load_preserves_output():
    cfg = get_r8_config()
    backend = ReferenceGeoBackend()
    model1, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    model2, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    
    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    out1, _ = model1(inputs)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(model1.state_dict(), tmp_path)
        
    model2.load_state_dict(torch.load(tmp_path), strict=True)
    os.remove(tmp_path)
    
    out2, _ = model2(inputs)
    assert torch.allclose(out1, out2, atol=1e-5), "Saved and reloaded model outputs must match."

if __name__ == "__main__":
    pytest.main([__file__])
