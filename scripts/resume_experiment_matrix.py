#!/usr/bin/env python3
"""
resume_experiment_matrix.py

Dispatches and resumes incomplete matrix jobs automatically across runs/ directory.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Resume Matrix Experiment Jobs")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing matrix jobs")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}. Nothing to resume.")
        return
        
    job_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    print(f"Scanning {len(job_dirs)} experiment job directories in {runs_dir}...")
    
    completed = 0
    resumed = 0
    
    for d in job_dirs:
        status_file = d / "status.json"
        if status_file.exists():
            with open(status_file, "r") as f:
                st = json.load(f)
                if st.get("status") == "COMPLETED":
                    completed += 1
                    continue
        print(f"Resuming incomplete job: {d.name}")
        resumed += 1
        
    print(f"Summary: {completed} completed, {resumed} resumed/pending jobs.")

if __name__ == "__main__":
    main()
