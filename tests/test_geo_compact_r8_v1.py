#!/usr/bin/env python3
"""
test_geo_compact_r8_v1.py

Unit test suite for the GeoCompactDecoder rank 8 platform preset.
Instantiates the ACTUAL GeoCompactDecoder model from the geosdp codebase.

Verifies:
1. Configuration loading matching configs/geo_compact_r8_v1.json.
2. Trainable parameter count matching recorded 8,401,804 target (8,401,888 byte budget).
3. 12 replaced linear projection layers (attention proj + ffn down) with rank 8.
4. Forward/backward execution and non-zero gradient computation.
5. State dictionary save/load serialization round-trips.
"""

import os
import sys
import json
import tempfile
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from geosdp import ModelConfig
from geosdp.backends.reference import ReferenceGeoBackend
from geosdp.models.geo_decoder_variants import build_decoder_variant, GeoImplicitStructuredLinear

CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "geo_compact_r8_v1.json")

def load_platform_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_config_structure():
    """Verify platform preset config matches rank 8 specification."""
    cfg_json = load_platform_config()
    assert cfg_json["architecture"] == "GeoCompactDecoder"
    assert cfg_json["rank"] == 8
    assert cfg_json["vocab_size"] == 4096
    assert cfg_json["d_model"] == 256
    assert cfg_json["ffn_hidden"] == 768
    assert cfg_json["n_heads"] == 4
    assert cfg_json["n_layers"] == 6
    assert cfg_json["seq_len"] == 512

def test_real_model_instantiation_and_parameter_budget():
    """Instantiate actual GeoCompactDecoder and verify parameter budget matches 8,401,804 params."""
    cfg_json = load_platform_config()
    cfg = ModelConfig(
        vocab_size=cfg_json["vocab_size"],
        d_model=cfg_json["d_model"],
        ffn_hidden=cfg_json["ffn_hidden"],
        n_heads=cfg_json["n_heads"],
        n_layers=cfg_json["n_layers"],
        seq_len=cfg_json["seq_len"]
    )
    backend = ReferenceGeoBackend()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=cfg_json["rank"])
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 8401804, f"Expected 8,401,804 trainable params, got {trainable_params}"
    assert len(rmap) == 12, f"Expected 12 replaced projection layers, got {len(rmap)}"
    
    for layer_name, info in rmap.items():
        assert info["rank"] == 8
        assert info["operator"] == "native_geo_compact"

def test_forward_backward_and_gradients():
    """Run forward and backward pass on real GeoCompactDecoder model."""
    cfg_json = load_platform_config()
    cfg = ModelConfig(
        vocab_size=cfg_json["vocab_size"],
        d_model=cfg_json["d_model"],
        ffn_hidden=cfg_json["ffn_hidden"],
        n_heads=cfg_json["n_heads"],
        n_layers=cfg_json["n_layers"],
        seq_len=16
    )
    backend = ReferenceGeoBackend()
    model, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=cfg_json["rank"])
    
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    
    logits, _ = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()
            if param.grad.abs().sum() > 0:
                has_grad = True
    assert has_grad, "Expected non-zero finite gradients across model parameters."

def test_serialization_round_trip():
    """Verify save/load state_dict round-trips preserve tensor outputs."""
    cfg_json = load_platform_config()
    cfg = ModelConfig(
        vocab_size=cfg_json["vocab_size"],
        d_model=cfg_json["d_model"],
        ffn_hidden=cfg_json["ffn_hidden"],
        n_heads=cfg_json["n_heads"],
        n_layers=2,
        seq_len=16
    )
    backend = ReferenceGeoBackend()
    model1, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=cfg_json["rank"])
    model2, _ = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=cfg_json["rank"])
    
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out1, _ = model1(x)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(model1.state_dict(), tmp_path)
        
    model2.load_state_dict(torch.load(tmp_path))
    os.remove(tmp_path)
    
    out2, _ = model2(x)
    assert torch.allclose(out1, out2, atol=1e-5), "Loaded model outputs must match original."

if __name__ == "__main__":
    pytest.main([__file__])
