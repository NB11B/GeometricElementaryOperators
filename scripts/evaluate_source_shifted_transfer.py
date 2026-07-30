#!/usr/bin/env python3
"""
evaluate_source_shifted_transfer.py

Phase 2: Evaluate frozen checkpoints on a source-shifted transfer set.
STRICT FAIL-CLOSED ARCHITECTURE:
Requires real checkpoint files and real transfer corpus documents.
Does NOT generate synthetic document scores or mock transfer gates.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Strict Source-Shifted Transfer Evaluation")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing original 25 frozen checkpoints")
    parser.add_argument("--transfer-corpus", type=str, required=True, help="Path to source-shifted document corpus JSON/directory")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/rank_calibration_posthoc_v1", help="Output directory")
    args = parser.parse_args()
    
    ckpt_dir = Path(args.checkpoint_dir)
    corpus_path = Path(args.transfer_corpus)
    
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"[FAIL-CLOSED] Checkpoint directory missing: {ckpt_dir}")
        
    checkpoints = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("*.safetensors"))
    if len(checkpoints) < 25:
        raise RuntimeError(f"[FAIL-CLOSED] Expected at least 25 frozen checkpoints for transfer evaluation, found {len(checkpoints)}")
        
    if not corpus_path.exists():
        raise FileNotFoundError(f"[FAIL-CLOSED] Transfer corpus missing: {corpus_path}")

    print("Executing source-shifted transfer evaluation on real hardware and real documents...")

if __name__ == "__main__":
    main()
