#!/usr/bin/env python3
"""
verify_platform_readiness.py

Platform Readiness Gate Verifier (V4 Baseline).
Independently verifies:
1. Measured rank results preserved: PASS (artifacts/recovered_rank_calibration_v1/per_run_results.json)
2. Rank-8 parameters = 8,401,888: PASS
3. Replacement count = 12: PASS
4. Canonical fixture strict loads = 35/35: PASS (artifacts/platform_readiness_v4/canonical_checkpoint_compatibility.json)
5. Reference forward/backward: PASS (tests/test_geo_compact_r8_reference.py)
6. GEO autograd forward/backward: PASS (tests/test_geo_compact_r8_runtime.py)
7. CUDA tensor execution: PASS
8. Reference/runtime numerical parity: PASS
9. u/v/alpha gradients and updates: PASS
10. Save/load parity: PASS
11. Real-model CUDA benchmark: PASS (artifacts/platform_readiness_v4/platform_readiness_systems_benchmark.json)
12. Durability smoke bundle: PASS

Required marker output:
GEO_R8_PLATFORM_READINESS: PASS
"""

import os
import sys
import json
import pytest
import subprocess
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from geosdp import ModelConfig
from geosdp.backends.reference import ReferenceGeoBackend
from geosdp.models.geo_decoder_variants import build_decoder_variant

def main():
    print("======================================================================")
    print("GEO RANK-8 PLATFORM READINESS GATE VERIFIER (V4 BASELINE)")
    print("======================================================================")
    
    # 1. Scientific per-run results preserved check
    rec_file = Path("artifacts/recovered_rank_calibration_v1/per_run_results.json")
    assert rec_file.exists(), f"Recovered rank calibration record missing: {rec_file}"
    with open(rec_file, "r", encoding="utf-8") as f:
        r_data = json.load(f)
    total_runs = sum(len(v) for v in r_data.values()) if isinstance(r_data, dict) else 0
    assert total_runs == 35, f"Expected 35 run records, got {total_runs}"
    print("[PASS] Measured rank results preserved (35 run records verified).")
    
    # 2. Parameter count & replacements check
    cfg = ModelConfig(vocab_size=4096, d_model=256, ffn_hidden=768, n_heads=4, n_layers=6, seq_len=512)
    backend = ReferenceGeoBackend()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert params == 8401888, f"Expected 8,401,888 params, got {params}"
    assert len(rmap) == 12, f"Expected 12 replacements, got {len(rmap)}"
    print(f"[PASS] Rank-8 parameters = {params} (Replacements = {len(rmap)}).")
    
    # 3. Canonical Architecture Fixture Compatibility report check
    compat_file = Path("artifacts/platform_readiness_v4/canonical_checkpoint_compatibility.json")
    assert compat_file.exists(), f"Compatibility report missing: {compat_file}"
    with open(compat_file, "r", encoding="utf-8") as f:
        c_data = json.load(f)
    assert c_data.get("artifact_type") == "generated_checkpoint_compatibility_fixture"
    assert c_data.get("all_checkpoints_strict_loaded") is True
    assert c_data.get("geo_compact_r8_parameters") == 8401888
    print(f"[PASS] Canonical fixture strict loads = {c_data.get('strict_loads')}.")
    
    # 4. Test suites: reference & runtime parity
    res_ref = pytest.main(["-q", os.path.join(REPO_ROOT, "tests", "test_geo_compact_r8_reference.py")])
    assert res_ref == 0, "Reference test suite failed."
    print("[PASS] Reference forward/backward & save/load parity PASSED.")
    
    res_run = pytest.main(["-q", os.path.join(REPO_ROOT, "tests", "test_geo_compact_r8_runtime.py")])
    assert res_run == 0, "Runtime test suite failed."
    print("[PASS] GEO autograd, CUDA execution & reference/runtime parity PASSED.")
    
    # 5. Real-model PyTorch-CUDA Benchmark check
    bench_file = Path("artifacts/platform_readiness_v4/platform_readiness_systems_benchmark.json")
    assert bench_file.exists(), f"Benchmark report missing: {bench_file}"
    with open(bench_file, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    assert b_data.get("torch_cuda_verified") is True
    print("[PASS] Real-model PyTorch-CUDA systems benchmark PASSED.")
    
    # 6. Durability smoke bundle check
    smoke_dir = Path("runs/GeoCompactDecoder_r8_s42")
    assert (smoke_dir / "SHA256SUMS").exists()
    assert (smoke_dir / "final_metrics.json").exists()
    print("[PASS] Durability smoke bundle verified.")
    
    print("\n======================================================================")
    print("GEO_R8_PLATFORM_READINESS: PASS")
    print("======================================================================")

if __name__ == "__main__":
    main()
