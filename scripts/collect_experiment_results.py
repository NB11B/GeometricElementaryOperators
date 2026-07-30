#!/usr/bin/env python3
"""
collect_experiment_results.py

Aggregates atomic job outputs into per_run_results.json without waiting for all jobs.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Collect Experiment Results into Aggregate JSON")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing matrix runs")
    parser.add_argument("--output-json", type=str, default="artifacts/per_run_results.json", help="Path for aggregate output JSON")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    runs_data = []
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            if d.is_dir():
                st_file = d / "status.json"
                if st_file.exists():
                    with open(st_file, "r") as f:
                        runs_data.append(json.load(f))
                        
    output = {
        "collected_runs": len(runs_data),
        "runs": runs_data
    }
    
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Collected {len(runs_data)} completed run records to {out_file}")

if __name__ == "__main__":
    main()
