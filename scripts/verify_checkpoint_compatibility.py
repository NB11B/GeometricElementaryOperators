#!/usr/bin/env python3
"""
verify_checkpoint_compatibility.py

Fix 1: Canonical Architecture Fixture Compatibility Verifier.
Strictly loads all 35 generated architecture compatibility fixtures from
artifacts/canonical_checkpoint_fixtures_v1/checkpoints/.

Outputs artifacts/platform_readiness_v4/canonical_checkpoint_compatibility.json.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from geosdp import ModelConfig
from geosdp.backends.reference import ReferenceGeoBackend
from geosdp.models.geo_decoder_variants import build_decoder_variant

EXPECTED_MATRIX = [
    ("GeoDenseDecoder", 0),
    ("StandardLowRankDecoder", 4),
    ("StandardLowRankDecoder", 8),
    ("StandardLowRankDecoder", 16),
    ("GeoCompactDecoder", 4),
    ("GeoCompactDecoder", 8),
    ("GeoCompactDecoder", 16),
]

SEEDS = [42, 43, 44, 45, 46]

def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify Canonical Architecture Checkpoint Fixture Compatibility")
    parser.add_argument("--bundle", type=str, default="artifacts/canonical_checkpoint_fixtures_v1", help="Fixture bundle path")
    parser.add_argument("--output", type=str, default="artifacts/platform_readiness_v4/canonical_checkpoint_compatibility.json", help="Output path")
    args = parser.parse_args()
    
    bundle_path = Path(args.bundle)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=256,
        ffn_hidden=768,
        n_heads=4,
        n_layers=6,
        seq_len=512
    )
    backend = ReferenceGeoBackend()
    
    results = []
    total_checkpoints = 0
    strict_pass_count = 0
    r8_geo_param_count = 0
    
    ckpt_dir = bundle_path / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory missing: {ckpt_dir}")
        
    for v_base, rank in EXPECTED_MATRIX:
        v_key = f"{v_base}_r{rank}" if rank > 0 else v_base
        for seed in SEEDS:
            total_checkpoints += 1
            ckpt_name = f"{v_key}_s{seed}.pt"
            ckpt_path = ckpt_dir / ckpt_name
            
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Expected architecture fixture not found: {ckpt_path}")
                
            model, rmap = build_decoder_variant(v_base, cfg, backend, rank=rank)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if v_base == "GeoCompactDecoder" and rank == 8:
                r8_geo_param_count = total_params
                
            ckpt_hash = compute_file_sha256(ckpt_path)
            sd_obj = torch.load(ckpt_path, map_location="cpu")
            state_dict = sd_obj.get("state_dict", sd_obj.get("model", sd_obj))
            
            model.load_state_dict(state_dict, strict=True)
            strict_pass_count += 1
            
            results.append({
                "fixture_path": str(ckpt_path),
                "fixture_sha256": ckpt_hash,
                "variant_name": v_key,
                "architecture": v_base,
                "rank": rank,
                "seed": seed,
                "parameters": total_params,
                "strict_load_result": "PASS"
            })
            
    summary_package = {
        "artifact_type": "generated_checkpoint_compatibility_fixture",
        "trained_weights": False,
        "scientific_evidence": False,
        "engineering_evidence": True,
        "strict_loads": f"{strict_pass_count}/{total_checkpoints}",
        "verified_checkpoints": f"{strict_pass_count}/{total_checkpoints}",
        "all_checkpoints_strict_loaded": (strict_pass_count == total_checkpoints),
        "geo_compact_r8_parameters": r8_geo_param_count,
        "total_missing_keys": 0,
        "total_unexpected_keys": 0,
        "total_shape_mismatches": 0,
        "runs": results
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_package, f, indent=2)
        
    print(f"\nCanonical Architecture Checkpoint Fixture Strict Load Verification:")
    print(f"Passed: {strict_pass_count}/{total_checkpoints} canonical architecture fixtures strict-loaded")
    print(f"GeoCompactDecoder r8 parameters: {r8_geo_param_count}")
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
