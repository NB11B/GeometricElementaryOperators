#!/usr/bin/env python3
"""
verify_experiment_bundle.py

Work Package 6: Validates durability checksums, completion status, and required schema fields across job bundles.
Exits nonzero if any verification condition fails.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

REQUIRED_JOB_FILES = [
    "config.json",
    "metrics.jsonl",
    "final_metrics.json",
    "selected_checkpoint.pt",
    "environment.json",
    "status.json",
    "SHA256SUMS"
]

def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify Experiment Bundle Integrity")
    parser.add_argument("--bundle-dir", type=str, required=True, help="Directory containing job bundle")
    args = parser.parse_args()
    
    b_dir = Path(args.bundle_dir)
    if not b_dir.exists() or not b_dir.is_dir():
        print(f"FAIL: Bundle directory missing: {b_dir}")
        sys.exit(1)
        
    sha_file = b_dir / "SHA256SUMS"
    if not sha_file.exists():
        print(f"FAIL: SHA256SUMS file missing in {b_dir}")
        sys.exit(1)
        
    st_file = b_dir / "status.json"
    if not st_file.exists():
        print(f"FAIL: status.json missing in {b_dir}")
        sys.exit(1)
        
    with open(st_file, "r", encoding="utf-8") as f:
        st = json.load(f)
    if st.get("status") != "COMPLETED":
        print(f"FAIL: Job status is {st.get('status')}, expected COMPLETED")
        sys.exit(1)
        
    # Check all required files
    for fname in REQUIRED_JOB_FILES:
        fpath = b_dir / fname
        if not fpath.exists():
            print(f"FAIL: Required file missing: {fname}")
            sys.exit(1)
            
    # Check SHA256SUMS entries
    with open(sha_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]
        
    for expected_hash, fname in lines:
        fpath = b_dir / fname
        if not fpath.exists():
            print(f"FAIL: SHA256SUMS listed file missing: {fname}")
            sys.exit(1)
        actual_hash = compute_file_sha256(fpath)
        if actual_hash != expected_hash:
            print(f"FAIL: Checksum mismatch for {fname}")
            sys.exit(1)
            
    print(f"DURABILITY_BUNDLE_VERIFY: PASS bundle={b_dir.name}")
    sys.exit(0)

if __name__ == "__main__":
    main()
