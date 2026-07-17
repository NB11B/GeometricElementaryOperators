#!/usr/bin/env python3
"""Run and strictly validate the fixed GEB-36 typed numerical envelope."""
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

BACKEND = "fixed_geb36"
MANIFEST = (
    ("zero", "cl20"), ("one", "cl20"), ("minus_one", "cl20"),
    ("e1", "cl20"), ("e2", "cl20"), ("pseudoscalar", "cl20"),
    ("negation", "cl20"), ("reversion", "cl20"),
    ("grade_involution", "cl20"), ("clifford_conjugation", "cl20"),
    ("scalar_projection", "cl20"), ("vector_projection", "cl20"),
    ("bivector_projection", "cl20"), ("addition", "cl20"),
    ("subtraction", "cl20"), ("geometric_product", "cl20"),
    ("reverse_product", "cl20"), ("vector_dot", "scalar"),
    ("vector_wedge", "cl20"), ("commutator", "cl20"),
    ("anticommutator", "cl20"), ("vector_norm_squared", "scalar"),
    ("distance_squared", "scalar"),
    ("projection_numerator", "projective"),
    ("rejection_numerator", "projective"),
    ("reflection_numerator", "projective"), ("dual", "cl20"),
    ("even_projection", "cl20"), ("odd_projection", "cl20"),
    ("rotor_action", "cl20"), ("rotor_composition", "cl20"),
    ("rotor_norm_squared", "scalar"), ("dilation", "cl20"),
    ("translation_unipotent", "unipotent"),
    ("vector_inverse_projective", "projective"),
    ("angle_cosine_numerator", "scalar"),
)
EXPECTED = dict(MANIFEST)
FIELDS = (
    "operation", "backend", "expected_kind", "requested", "completed",
    "overflows", "status_failures", "kind_failures", "max_absolute",
    "max_relative", "max_angular", "max_projective_scale", "mismatches",
)
CONFIG_RE = re.compile(
    r"^samples=(?P<samples>[0-9]+) seed=(?P<seed>[0-9]+) "
    r"precision=(?P<precision>float|double) "
    r"q_fraction_bits=(?P<fraction_bits>[0-9]+)$"
)


def parse_configuration(stdout: str) -> tuple[int, int, str, int]:
    matches = [CONFIG_RE.match(line.strip()) for line in stdout.splitlines()]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise RuntimeError(
            "numerical envelope must emit exactly one configuration line; "
            f"found {len(matches)}\n{stdout}"
        )
    match = matches[0]
    assert match is not None
    return (
        int(match.group("samples")),
        int(match.group("seed")),
        match.group("precision"),
        int(match.group("fraction_bits")),
    )


def validate_configuration(
    stdout: str, samples: int, seed: int, precision: str, fraction_bits: int
) -> tuple[int, int, str, int]:
    reported = parse_configuration(stdout)
    requested = (samples, seed, precision, fraction_bits)
    if reported != requested:
        raise RuntimeError(
            "numerical envelope configuration does not match requested values: "
            f"requested={requested} reported={reported}"
        )
    return reported


def finite_nonnegative(value: str, field: str, operation: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(f"invalid {field} for operation {operation}: {value}")
    return parsed


def nonnegative_count(value: str, field: str, operation: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(
            f"invalid {field} for operation {operation}: {value}"
        ) from error
    if parsed < 0:
        raise RuntimeError(f"negative {field} for operation {operation}")
    return parsed


def validate_rows(
    rows: list[dict[str, str]], samples: int, precision: str, fraction_bits: int
) -> list[dict[str, object]]:
    seen: set[str] = set()
    typed_rows: list[dict[str, object]] = []
    for row in rows:
        missing = [field for field in FIELDS if field not in row or row[field] is None]
        if missing:
            raise RuntimeError(f"numerical row is missing fields: {missing}")
        operation = row["operation"]
        if operation not in EXPECTED:
            raise RuntimeError(f"unexpected numerical operation row: {operation}")
        if operation in seen:
            raise RuntimeError(f"duplicate numerical operation row: {operation}")
        seen.add(operation)
        if row["backend"] != BACKEND:
            raise RuntimeError(
                f"operation {operation} reported backend {row['backend']}; "
                f"expected {BACKEND}"
            )
        if row["expected_kind"] != EXPECTED[operation]:
            raise RuntimeError(
                f"operation {operation} reported kind {row['expected_kind']}; "
                f"expected {EXPECTED[operation]}"
            )

        requested = nonnegative_count(row["requested"], "requested", operation)
        completed = nonnegative_count(row["completed"], "completed", operation)
        overflows = nonnegative_count(row["overflows"], "overflows", operation)
        status_failures = nonnegative_count(
            row["status_failures"], "status_failures", operation
        )
        kind_failures = nonnegative_count(
            row["kind_failures"], "kind_failures", operation
        )
        mismatches = nonnegative_count(row["mismatches"], "mismatches", operation)
        if requested != samples:
            raise RuntimeError(
                f"operation {operation} requested={requested}; expected {samples}"
            )
        accounted = completed + overflows + status_failures + kind_failures
        if accounted != requested:
            raise RuntimeError(
                f"operation {operation} accounting={accounted}; expected {requested}"
            )
        if completed != requested:
            raise RuntimeError(
                f"operation {operation} completed only {completed}/{requested}; "
                "bounded fixtures require every sample to complete"
            )
        if overflows or status_failures or kind_failures or mismatches:
            raise RuntimeError(
                f"operation {operation} recorded failure counts: "
                f"overflows={overflows} status={status_failures} "
                f"kind={kind_failures} mismatches={mismatches}"
            )
        typed_rows.append({
            "operation": operation,
            "backend": BACKEND,
            "expected_kind": row["expected_kind"],
            "precision": precision,
            "q_fraction_bits": fraction_bits,
            "requested": requested,
            "completed": completed,
            "overflows": overflows,
            "status_failures": status_failures,
            "kind_failures": kind_failures,
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
                row["max_projective_scale"], "max_projective_scale", operation
            ),
            "mismatches": mismatches,
        })
    missing_operations = [name for name, _kind in MANIFEST if name not in seen]
    if missing_operations:
        raise RuntimeError(f"missing numerical operations: {missing_operations}")
    if len(typed_rows) != len(MANIFEST):
        raise RuntimeError(
            f"expected {len(MANIFEST)} numerical rows; found {len(typed_rows)}"
        )
    return typed_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6A09E667)
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    parser.add_argument("--q-fraction-bits", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.seed < 0 or args.seed > 0xFFFFFFFF:
        parser.error("seed must be in the unsigned 32-bit range")
    if args.q_fraction_bits < 1 or args.q_fraction_bits > 30:
        parser.error("q-fraction-bits must be in the range 1..30")
    if not args.executable.is_file():
        parser.error(f"missing executable: {args.executable}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "numerical_error.csv"
    completed = subprocess.run(
        [str(args.executable), "--samples", str(args.samples), "--seed",
         str(args.seed), "--csv", str(raw_csv)],
        text=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    (args.out_dir / "numerical_error.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"numerical envelope failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    _reported_samples, _reported_seed, reported_precision, reported_q = (
        validate_configuration(
            completed.stdout, args.samples, args.seed, args.precision,
            args.q_fraction_bits
        )
    )
    if not raw_csv.is_file():
        raise RuntimeError("numerical envelope did not create its CSV output")
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError(
                f"numerical CSV fields {reader.fieldnames}; expected {list(FIELDS)}"
            )
        rows = list(reader)
    typed_rows = validate_rows(rows, args.samples, reported_precision, reported_q)

    metadata = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "precision": reported_precision,
        "q_fraction_bits": reported_q,
        "samples_per_operation": args.samples,
        "seed": args.seed,
        "operation_count": len(typed_rows),
        "completed_total": sum(int(row["completed"]) for row in typed_rows),
        "manifest": [
            {"operation": operation, "backend": BACKEND, "expected_kind": kind}
            for operation, kind in MANIFEST
        ],
        "oracle_scope": "typed floating-reference envelope over quantized fixtures",
    }
    payload = {"metadata": metadata, "results": typed_rows}
    (args.out_dir / "numerical_error.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    markdown = [
        "# Fixed GEB-36 numerical envelope", "",
        f"Generated: `{metadata['generated_utc']}`", "",
        f"Precision: `{reported_precision}`  ",
        f"Q fraction bits: `{reported_q}`  ",
        f"Requested and completed samples per operation: `{args.samples}`  ",
        f"Seed: `{args.seed}`", "",
        "This is a typed floating-reference envelope over decoded, quantized "
        "fixtures; it is not an independently implemented oracle.", "",
        "| Operation | Kind | Completed | Max absolute | Max relative | "
        "Max angular (rad) | Max projective-scale error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in typed_rows:
        markdown.append(
            f"| {row['operation']} | {row['expected_kind']} | {row['completed']} | "
            f"{row['max_absolute']:.3e} | {row['max_relative']:.3e} | "
            f"{row['max_angular_radians']:.3e} | "
            f"{row['max_projective_scale_error']:.3e} |"
        )
    markdown.extend(["", "All 36 manifest entries must complete every bounded sample "
                     "with the expected result kind and zero mismatches.", ""])
    (args.out_dir / "numerical_error.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
