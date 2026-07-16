#!/usr/bin/env python3
"""Run the fixed numerical envelope and emit CSV, JSON, and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6A09E667)
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("samples must be positive")
    if not args.executable.exists():
        parser.error(f"missing executable: {args.executable}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "numerical_error.csv"
    completed = subprocess.run(
        [
            str(args.executable),
            "--samples", str(args.samples),
            "--seed", str(args.seed),
            "--csv", str(raw_csv),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    (args.out_dir / "numerical_error.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"numerical envelope failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("numerical envelope produced no result rows")

    typed_rows: list[dict[str, object]] = []
    mismatch_total = 0
    overflow_total = 0
    completed_total = 0
    for row in rows:
        typed = {
            "operation": row["operation"],
            "backend": row["backend"],
            "precision": args.precision,
            "requested": int(row["requested"]),
            "completed": int(row["completed"]),
            "overflows": int(row["overflows"]),
            "max_absolute": float(row["max_absolute"]),
            "max_relative": float(row["max_relative"]),
            "max_angular_radians": float(row["max_angular"]),
            "max_projective_scale_error": float(row["max_projective_scale"]),
            "mismatches": int(row["mismatches"]),
        }
        mismatch_total += int(typed["mismatches"])
        overflow_total += int(typed["overflows"])
        completed_total += int(typed["completed"])
        typed_rows.append(typed)

    metadata = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "precision": args.precision,
        "samples_per_operation": args.samples,
        "seed": args.seed,
        "completed_total": completed_total,
        "mismatch_total": mismatch_total,
        "overflow_total": overflow_total,
    }
    payload = {"metadata": metadata, "results": typed_rows}
    (args.out_dir / "numerical_error.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    markdown = [
        "# Numerical error envelope",
        "",
        f"Generated: `{metadata['generated_utc']}`",
        "",
        f"Precision: `{args.precision}`  ",
        f"Requested samples per operation: `{args.samples}`  ",
        f"Seed: `{args.seed}`",
        "",
        "| Operation | Backend | Requested | Completed | Overflows | Max absolute | Max relative | Max angular (rad) | Max projective-scale error | Mismatches |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in typed_rows:
        markdown.append(
            f"| {row['operation']} | {row['backend']} | {row['requested']} | "
            f"{row['completed']} | {row['overflows']} | "
            f"{row['max_absolute']:.3e} | {row['max_relative']:.3e} | "
            f"{row['max_angular_radians']:.3e} | "
            f"{row['max_projective_scale_error']:.3e} | {row['mismatches']} |"
        )
    markdown.extend([
        "",
        "Componentwise, angular, and projective-scale errors are computed on deterministic fixtures. Checked fixed-point overflows are excluded from error statistics and reported separately.",
        "",
    ])
    (args.out_dir / "numerical_error.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
