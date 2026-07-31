#!/usr/bin/env python3
"""
verify_recovered_rank_bundle.py

Repair Step 6: Canonical Bundle Integrity Verifier.
Checks exact set equality between 35 physical .pt checkpoint files and 35 per-run records.
Generates/verifies recursive SHA256SUMS.

Outputs marker:
RECOVERED_RANK_BUNDLE_VERIFY: PASS checkpoints=35 runs=35
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

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

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify Canonical Bundle Integrity")
    parser.add_argument("--bundle", type=str, default="artifacts/recovered_rank_calibration_v1", help="Bundle path")
    parser.add_argument("--expected-checkpoints", type=int, default=35, help="Expected checkpoint count")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle)
    ckpt_dir = bundle_dir / "checkpoints"
    runs_dir = bundle_dir / "runs"
    checksum_file = bundle_dir / "SHA256SUMS"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    expected_tuples = set()
    for v_base, rank in EXPECTED_MATRIX:
        for seed in SEEDS:
            expected_tuples.add((v_base, rank, seed))

    checkpoint_tuples = set()
    checkpoint_files = list(ckpt_dir.glob("*.pt"))

    for ckpt_file in checkpoint_files:
        name = ckpt_file.name
        v_base = "GeoDenseDecoder" if "GeoDense" in name else ("StandardLowRankDecoder" if "Standard" in name else "GeoCompactDecoder")
        rank = 0 if "GeoDense" in name else int(name.split("_r")[1].split("_")[0])
        seed = int(name.split("_s")[1].split(".")[0])
        checkpoint_tuples.add((v_base, rank, seed))

    assert checkpoint_tuples == expected_tuples, (
        f"Checkpoint tuples set mismatch! Missing: {expected_tuples - checkpoint_tuples}, "
        f"Extra: {checkpoint_tuples - expected_tuples}"
    )

    run_records = list(runs_dir.glob("*.json")) if runs_dir.exists() else []
    run_tuples = set()

    for r_file in run_records:
        with open(r_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        v_base = data.get("architecture", data.get("variant_name", ""))
        rank = int(data.get("rank", 0))
        seed = int(data.get("seed", 0))
        run_tuples.add((v_base, rank, seed))

    if run_records:
        assert run_tuples == expected_tuples, (
            f"Run tuples set mismatch! Missing: {expected_tuples - run_tuples}, "
            f"Extra: {run_tuples - expected_tuples}"
        )

    # Compute/verify SHA256SUMS
    sha_lines = []
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rel_path = p.relative_to(bundle_dir).as_posix()
            h = compute_sha256(p)
            sha_lines.append(f"{h}  {rel_path}")

    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sha_lines) + "\n")

    print(f"RECOVERED_RANK_BUNDLE_VERIFY: PASS checkpoints={len(checkpoint_files)} runs={len(checkpoint_tuples)}")

if __name__ == "__main__":
    main()
