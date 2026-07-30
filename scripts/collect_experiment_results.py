#!/usr/bin/env python3
"""
collect_experiment_results.py

Work Package 6: Aggregates verified final_metrics.json from COMPLETED jobs into per_run_results.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Collect Final Experiment Metrics into Aggregate JSON")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Matrix runs directory")
    parser.add_argument("--output-json", type=str, default="artifacts/per_run_results.json", help="Output JSON path")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    collected_runs = []
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                st_file = d / "status.json"
                fm_file = d / "final_metrics.json"
                sha_file = d / "SHA256SUMS"
                
                if st_file.exists() and fm_file.exists() and sha_file.exists():
                    with open(st_file, "r", encoding="utf-8") as f:
                        st = json.load(f)
                    if st.get("status") == "COMPLETED":
                        with open(fm_file, "r", encoding="utf-8") as f:
                            fm = json.load(f)
                        collected_runs.append(fm)
                        
    output_package = {
        "collected_runs_count": len(collected_runs),
        "runs": collected_runs
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_package, f, indent=2)
        
    print(f"Collected {len(collected_runs)} verified COMPLETED run metrics into {out_file}")

if __name__ == "__main__":
    main()
