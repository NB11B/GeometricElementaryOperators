#!/usr/bin/env python3
"""
inspect_checkpoint_schema.py

Work Package 2: Checkpoint Schema Inspector.
Inspects every state-dictionary key, dtype, and shape from a checkpoint.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Inspect Checkpoint Schema")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, default="artifacts/platform_readiness_v2/checkpoint_schema_r8_s42.json", help="Output JSON path")
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")
        
    print(f"Inspecting checkpoint schema: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(sd, dict) and "state_dict" in sd:
        state_dict = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        state_dict = sd["model"]
    elif isinstance(sd, dict):
        state_dict = sd
    else:
        raise TypeError(f"Unexpected checkpoint format: {type(sd)}")
        
    schema_info = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            schema_info[key] = {
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "numel": val.numel()
            }
            
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_package = {
        "checkpoint_file": str(ckpt_path),
        "total_keys": len(schema_info),
        "total_elements": sum(v["numel"] for v in schema_info.values()),
        "schema": schema_info
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_package, f, indent=2)
        
    print(f"Saved schema inspection report ({len(schema_info)} keys) to {out_path}")

if __name__ == "__main__":
    main()
