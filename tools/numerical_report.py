#!/usr/bin/env python3
"""Run the fixed numerical envelope and emit CSV, JSON, and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
import time
from pathlib import Path

CONFIG_RE = re.compile(
    r"^samples=(?P<samples>[0-9]+) seed=(?P<seed>[0-9]+) "
    r"q_fraction_bits=(?P<fraction_bits>[0-9]+)$"
)


def parse_configuration(stdout: str) -> tuple[int, int, int]:
    matches = [
        CONFIG_RE.match(line.strip())
        for line in stdout.splitlines()
    ]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise RuntimeError(
            "numerical envelope must emit exactly one configuration line; "
            f"found {len(matches)}\n{stdout}"
        )
    match = matches[0]
    assert match is not None
    samples = int(match.group("samples"))
    seed = int(match.group("seed"))
    fraction_bits = int(match.group("fraction_bits"))
    if fraction_bits < 1 or fraction_bits > 30:
        raise RuntimeError(
            f"numerical envelope reported invalid Q fraction bits: {fraction_bits}"
        )
    return samples, seed, fraction_bits


def finite_nonnegative(value: str, field: str, operation: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(
            f"invalid {field} for operation {operation}: {value}"
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        default=0x6A09E667,
    )
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.seed < 0 or args.seed > 0xFFFFFFFF:
        parser.error("seed must be in the unsigned 32-bit range")
    if not args.executable.is_file():
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

    reported_samples, reported_seed, fraction_bits = parse_configuration(
        completed.stdout
    )
    if reported_samples != args.samples or reported_seed != args.seed:
        raise RuntimeError(
            "numerical envelope configuration does not match requested values: "
            f"requested=({args.samples}, {args.seed}) "
            f"reported=({reported_samples}, {reported_seed})"
        )
    if not raw_csv.is_file():
        raise RuntimeError("numerical envelope did not create its CSV output")

    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("numerical envelope produced no result rows")

    typed_rows: list[dict[str, object]] = []
    seen_operations: set[str] = set()
    mismatch_total = 0
    overflow_total = 0
    completed_total = 0
    for row in rows:
        operation = row["operation"]
        if operation in seen_operations:
            raise RuntimeError(f"duplicate numerical operation row: {operation}")
        seen_operations.add(operation)
        requested = int(row["requested"])
        completed_count = int(row["completed"])
        overflows = int(row["overflows"])
        mismatches = int(row["mismatches"])
        if requested != args.samples:
            raise RuntimeError(
                f"operation {operation} reported {requested} requested samples; "
                f"expected {args.samples}"
            )
        if completed_count < 0 or overflows < 0 or mismatches < 0:
            raise RuntimeError(f"operation {operation} reported a negative count")
        if completed_count + overflows != requested:
            raise RuntimeError(
                f"operation {operation} did not account for every sample: "
                f"completed={completed_count} overflows={overflows} "
                f"requested={requested}"
            )
        typed = {
            "operation": operation,
            "backend": row["backend"],
            "precision": args.precision,
            "q_fraction_bits": fraction_bits,
            "requested": requested,
            "completed": completed_count,
            "overflows": overflows,
            "max_absolute": finite_nonnegative(
                row["max_absolute"], "max_absolute", operation
            ),
            "max_relative": finite_nonnegative(
                row["max_relative"], "max_relative", operation
            ),
            "max_angular_radians": finite_nonnegative(
                row["max_angular"], "max_angular", operation
            ),
            "max_projective_scale_error": finite_nonnegative(
                row["max_projective_scale"],
                "max_projective_scale",
                operation,
            ),
            "mismatches": mismatches,
        }
        mismatch_total += mismatches
        overflow_total += overflows
        completed_total += completed_count
        typed_rows.append(typed)

    metadata = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "precision": args.precision,
        "q_fraction_bits": fraction_bits,
        "samples_per_operation": args.samples,
        "seed": args.seed,
        "operation_count": len(typed_rows),
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
        f"Q fraction bits: `{fraction_bits}`  ",
        f"Requested samples per operation: `{args.samples}`  ",
        f"Seed: `{args.seed}`",
        "",
        "| Operation | Backend | Precision | Q frac | Requested | Completed | Overflows | Max absolute | Max relative | Max angular (rad) | Max projective-scale error | Mismatches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in typed_rows:
        markdown.append(
            f"| {row['operation']} | {row['backend']} | {row['precision']} | "
            f"{row['q_fraction_bits']} | {row['requested']} | "
            f"{row['completed']} | {row['overflows']} | "
            f"{row['max_absolute']:.3e} | {row['max_relative']:.3e} | "
            f"{row['max_angular_radians']:.3e} | "
            f"{row['max_projective_scale_error']:.3e} | "
            f"{row['mismatches']} |"
        )
    markdown.extend([
        "",
        "Componentwise, angular, and projective-scale errors are computed on "
        "deterministic fixtures. Checked fixed-point overflows are excluded "
        "from error statistics and reported separately.",
        "",
    ])
    (args.out_dir / "numerical_error.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    if mismatch_total != 0:
        raise RuntimeError(
            f"numerical envelope recorded {mismatch_total} mismatches"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
