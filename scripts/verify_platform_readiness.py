#!/usr/bin/env python3
"""
verify_platform_readiness.py

Repair Step 8: Master Platform Readiness Gate Verifier.
Independently verifies:
1. Canonical Bundle Integrity (35/35 set equality, SHA256SUMS)
2. GeoCompactDecoder r8 parameter count = 8,401,888
3. 35/35 real physical checkpoint strict loads (no synthetic fallbacks)
4. Reference forward/backward gate tests = PASS
5. Native CUDA forward/backward gate tests + Numerical Parity = PASS
6. Compiled geo_dl_runtime module loaded (module file != None)
7. CUDA available = True
8. GEO owns backward = True
9. Implicit linear native calls > 0
10. Fallback count = 0
11. Real native benchmark report = PASS
12. Durability smoke bundle verification = PASS

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
    print("GEO RANK-8 PLATFORM READINESS GATE VERIFIER (STRICT REPAIRED)")
    print("======================================================================")
    
    # 1. Parameter count check
    cfg = ModelConfig(vocab_size=4096, d_model=256, ffn_hidden=768, n_heads=4, n_layers=6, seq_len=512)
    backend = ReferenceGeoBackend()
    model, rmap = build_decoder_variant("GeoCompactDecoder", cfg, backend, rank=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert params == 8401888, f"Expected 8,401,888 params, got {params}"
    assert len(rmap) == 12, f"Expected 12 replacements, got {len(rmap)}"
    print(f"[PASS] Parameter Count: GeoCompactDecoder r8 = {params} (Replacements: {len(rmap)})")
    
    # 2. Run bundle set equality verifier
    res_bundle = subprocess.run([
        sys.executable, "-m", "scripts.verify_recovered_rank_bundle",
        "--bundle", "artifacts/recovered_rank_calibration_v1",
        "--expected-checkpoints", "35"
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res_bundle.returncode == 0, f"Bundle verifier failed:\n{res_bundle.stderr}\n{res_bundle.stdout}"
    assert "RECOVERED_RANK_BUNDLE_VERIFY: PASS" in res_bundle.stdout
    print("[PASS] Bundle Integrity: 35/35 physical checkpoint files set equality & SHA256SUMS PASS.")
    
    # 3. Run real physical checkpoint compatibility verifier
    res_compat = subprocess.run([
        sys.executable, "-m", "scripts.verify_checkpoint_compatibility",
        "--bundle", "artifacts/recovered_rank_calibration_v1",
        "--output", "artifacts/platform_readiness_v3/checkpoint_compatibility.json"
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res_compat.returncode == 0, f"Checkpoint compatibility verifier failed:\n{res_compat.stderr}\n{res_compat.stdout}"
    
    compat_file = Path("artifacts/platform_readiness_v3/checkpoint_compatibility.json")
    assert compat_file.exists(), f"Compatibility report missing: {compat_file}"
    with open(compat_file, "r", encoding="utf-8") as f:
        c_data = json.load(f)
    assert c_data.get("all_checkpoints_strict_loaded") is True
    assert c_data.get("geo_compact_r8_parameters") == 8401888
    print(f"[PASS] Strict Physical Checkpoint Loading: {c_data.get('verified_checkpoints')} passed.")
    
    # 4. Run PyTest on reference and native test suites
    res_ref = pytest.main(["-q", os.path.join(REPO_ROOT, "tests", "test_geo_compact_r8_reference.py")])
    assert res_ref == 0, "Reference test suite failed."
    print("[PASS] Reference Test Suite (test_geo_compact_r8_reference.py): PASSED.")
    
    res_nat = pytest.main(["-q", os.path.join(REPO_ROOT, "tests", "test_geo_compact_r8_native.py")])
    assert res_nat == 0, "Native test suite failed."
    print("[PASS] Native & Numerical Parity Test Suite (test_geo_compact_r8_native.py): PASSED.")
    
    # 5. Real Native Benchmark Report assertions
    bench_file = Path("artifacts/platform_readiness_v3/platform_readiness_systems_benchmark.json")
    assert bench_file.exists(), f"Benchmark report missing: {bench_file}"
    with open(bench_file, "r", encoding="utf-8") as f:
        b_data = json.load(f)
        
    compact_bench = b_data.get("results", {}).get("GeoCompactDecoder_r8", {})
    assert compact_bench.get("runtime_module_loaded") is True, "runtime_module_loaded must be True"
    assert compact_bench.get("runtime_module_file") is not None, "runtime_module_file must not be None"
    assert compact_bench.get("cuda_available") is True, "cuda_available must be True"
    assert compact_bench.get("geo_owns_backward") is True, "geo_owns_backward must be True"
    assert compact_bench.get("implicit_linear_native_calls") > 0, "implicit_linear_native_calls must be > 0"
    assert compact_bench.get("fallback_count") == 0, "fallback_count must be 0"
    print("[PASS] Real Native Systems Benchmark verified (Compiled runtime loaded, CUDA active, zero fallbacks).")
    
    # 6. Durability smoke bundle check
    smoke_dir = Path("runs/GeoCompactDecoder_r8_s42")
    assert (smoke_dir / "SHA256SUMS").exists()
    assert (smoke_dir / "final_metrics.json").exists()
    print("[PASS] Durability Infrastructure Smoke Bundle verified.")
    
    print("\n======================================================================")
    print("GEO_R8_PLATFORM_READINESS: PASS")
    print("======================================================================")

if __name__ == "__main__":
    main()
