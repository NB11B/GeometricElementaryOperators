#!/usr/bin/env python3
"""
consolidate_35_checkpoints.py

Consolidates all 35 physical checkpoint files into artifacts/recovered_rank_calibration_v1/checkpoints/.
Maps each (architecture, rank, seed) tuple to its physical checkpoint file from GEOSDP artifacts.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
import torch

RECOVERED_CKPT_DIR = Path("artifacts/recovered_rank_calibration_v1/checkpoints")
RECOVERED_CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    Path(r"C:\GEO-Workspace\GEOSDP\artifacts\rank_calibration_v1\checkpoints"),
    Path(r"C:\GEO-Workspace\GEOSDP\artifacts\checkpoints"),
    Path(r"C:\GEO-Workspace\GEOSDP\artifacts\development_calibration\checkpoints")
]

MATRIX = [
    ("GeoDenseDecoder", 0, "GeoDenseDecoder"),
    ("StandardLowRankDecoder", 4, "StandardLowRankDecoder_r4"),
    ("StandardLowRankDecoder", 8, "StandardLowRankDecoder_r8"),
    ("StandardLowRankDecoder", 16, "StandardLowRankDecoder_r16"),
    ("GeoCompactDecoder", 4, "GeoCompactDecoder_r4"),
    ("GeoCompactDecoder", 8, "GeoCompactDecoder_r8"),
    ("GeoCompactDecoder", 16, "GeoCompactDecoder_r16"),
]

SEEDS = [42, 43, 44, 45, 46]

def main():
    print("Consolidating 35 physical checkpoint files into canonical bundle...")
    found_count = 0
    
    for v_base, rank, v_key in MATRIX:
        for seed in SEEDS:
            dest_name = f"{v_key}_s{seed}.pt"
            dest_path = RECOVERED_CKPT_DIR / dest_name
            
            if dest_path.exists():
                found_count += 1
                continue
                
            possible_patterns = [
                f"{v_key}_s{seed}.pt",
                f"{v_base}_s{seed}.pt",
                f"{v_base}_seed{seed}_best.pt",
                f"*{v_base}*seed{seed}*.pt",
                f"*{v_key}*s{seed}*.pt"
            ]
            
            matched_src = None
            for s_dir in SEARCH_DIRS:
                if not s_dir.exists():
                    continue
                for pat in possible_patterns:
                    matches = list(s_dir.glob(pat))
                    if matches:
                        matched_src = matches[0]
                        break
                if matched_src:
                    break
                    
            if matched_src:
                shutil.copy2(matched_src, dest_path)
                print(f"Copied {matched_src.name} -> {dest_name}")
                found_count += 1
            else:
                print(f"Warning: Could not locate source for {dest_name}")
                
    print(f"Consolidation completed: {found_count}/35 physical checkpoints present in {RECOVERED_CKPT_DIR}")

if __name__ == "__main__":
    main()
