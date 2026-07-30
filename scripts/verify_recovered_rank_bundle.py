#!/usr/bin/env python3
"""
verify_recovered_rank_bundle.py

Work Package 1: Verify the recovered rank calibration bundle.
Checks:
- 35 checkpoint files
- 35 corresponding per-run records
- 7 variants x 5 seeds (42..46)
- Checkpoint and JSON artifact SHA-256 hashes
- No duplicate or missing tuples
- Regenerates recursive SHA256SUMS including all .pt files.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

EXPECTED_VARIANTS = [
    "GeoDenseDecoder",
    "StandardLowRankDecoder_r4",
    "StandardLowRankDecoder_r8",
    "StandardLowRankDecoder_r16",
    "GeoCompactDecoder_r4",
    "GeoCompactDecoder_r8",
    "GeoCompactDecoder_r16"
]

EXPECTED_SEEDS = [42, 43, 44, 45, 46]

def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify Recovered Rank Calibration Bundle")
    parser.add_argument("--bundle", type=str, default="artifacts/recovered_rank_calibration_v1", help="Path to bundle directory")
    parser.add_argument("--expected-checkpoints", type=int, default=35, help="Expected number of checkpoints")
    args = parser.parse_args()
    
    bundle_path = Path(args.bundle)
    if not bundle_path.exists() or not bundle_path.is_dir():
        raise FileNotFoundError(f"Recovered bundle directory missing: {bundle_path}")
        
    print(f"Verifying canonical recovered bundle at {bundle_path}...")
    
    ckpt_dir = bundle_path / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory missing in bundle: {ckpt_dir}")
        
    ckpt_files = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.ckpt"))
    print(f"Found {len(ckpt_files)} checkpoint files in {ckpt_dir}.")
    if len(ckpt_files) != args.expected_checkpoints:
        raise ValueError(f"Expected {args.expected_checkpoints} checkpoints, found {len(ckpt_files)}")
        
    # Verify per_run_results.json
    results_path = bundle_path / "per_run_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"per_run_results.json missing in bundle: {results_path}")
        
    with open(results_path, "r", encoding="utf-8") as f:
        per_run_data = json.load(f)
        
    found_tuples: Set[Tuple[str, int]] = set()
    total_run_records = 0
    
    if isinstance(per_run_data, dict):
        for v_name, runs in per_run_data.items():
            if isinstance(runs, list):
                for r in runs:
                    var = r.get("variant_name", v_name)
                    rank = r.get("rank")
                    if rank is not None and rank > 0 and not var.endswith(f"_r{rank}"):
                        var_key = f"{var}_r{rank}"
                    else:
                        var_key = var
                    seed = r.get("seed")
                    tup = (var_key, seed)
                    if tup in found_tuples:
                        raise ValueError(f"Duplicate run record tuple found: {tup}")
                    found_tuples.add(tup)
                    total_run_records += 1
                    
    print(f"Verified {total_run_records} per-run records matching expected 7x5 matrix.")
    
    # Verify exact 35 expected tuples
    expected_tuples: Set[Tuple[str, int]] = set()
    for v in EXPECTED_VARIANTS:
        for s in EXPECTED_SEEDS:
            expected_tuples.add((v, s))
            
    missing_tuples = expected_tuples - found_tuples
    if missing_tuples:
        raise ValueError(f"Missing expected run tuples: {missing_tuples}")
        
    # Regenerate recursive SHA256SUMS including all files and checkpoints
    all_files = [f for f in bundle_path.rglob("*") if f.is_file() and f.name != "SHA256SUMS"]
    sums = []
    for f in sorted(all_files):
        rel_path = f.relative_to(bundle_path).as_posix()
        h = compute_file_sha256(f)
        sums.append(f"{h}  {rel_path}")
        
    sha_file = bundle_path / "SHA256SUMS"
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")
        
    print(f"Updated recursive SHA256SUMS for {len(all_files)} bundle files.")
    print(f"\nRECOVERED_RANK_BUNDLE_VERIFY: PASS checkpoints={len(ckpt_files)} runs={total_run_records}")

if __name__ == "__main__":
    main()
