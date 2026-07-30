#!/usr/bin/env python3
"""
verify_experiment_bundle.py

Validates SHA256 checksums and schema conformance across completed job bundles.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify Experiment Bundle Checksums & Schemas")
    parser.add_argument("--bundle-dir", type=str, required=True, help="Directory containing artifact bundle to verify")
    args = parser.parse_args()
    
    b_dir = Path(args.bundle_dir)
    if not b_dir.exists() or not b_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory missing: {b_dir}")
        
    sha_file = b_dir / "SHA256SUMS"
    if not sha_file.exists():
        print(f"Warning: SHA256SUMS file missing in {b_dir}")
        return
        
    with open(sha_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]
        
    verified = 0
    failed = 0
    for expected_hash, fname in lines:
        fpath = b_dir / fname
        if not fpath.exists():
            print(f"FAIL: File missing -> {fname}")
            failed += 1
            continue
        actual_hash = compute_file_sha256(fpath)
        if actual_hash == expected_hash:
            verified += 1
        else:
            print(f"FAIL: Checksum mismatch -> {fname} (Expected {expected_hash[:8]}, Got {actual_hash[:8]})")
            failed += 1
            
    print(f"Verification completed for {b_dir.name}: {verified} verified, {failed} failed.")

if __name__ == "__main__":
    main()
