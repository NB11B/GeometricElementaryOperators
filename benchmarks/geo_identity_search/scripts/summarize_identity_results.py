#!/usr/bin/env python3
"""Summarize and validate the exact geometric identity search CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


MARKER = "CSV_IDENTITY"
EXPECTED_NAMES = {
    "vector_square_is_scalar": "identity",
    "reverse_reverses_product_order": "identity",
    "vector_wedge_is_antisymmetric": "identity",
    "vector_product_is_not_commutative": "counterexample",
    "reverse_preserves_product_order_false": "counterexample",
}
COLUMNS = (
    "identity",
    "expected",
    "dimension",
    "signature",
    "prime",
    "variables",
    "nodes",
    "assignments",
    "cpu_checks",
    "kernel_us",
    "assignments_per_second",
    "found_counterexample",
    "witness_assignment",
    "witness_blade",
    "witness_lhs",
    "witness_rhs",
    "result",
)


@dataclass(frozen=True)
class Row:
    identity: str
    expected: str
    dimension: int
    signature: str
    prime: int
    variables: int
    nodes: int
    assignments: int
    cpu_checks: int
    kernel_us: float
    assignments_per_second: float
    found_counterexample: bool
    witness_assignment: int | None
    witness_blade: int | None
    witness_lhs: int | None
    witness_rhs: int | None
    result: str


def parse_bool(value: str, context: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{context} must be true or false, found {value!r}")


def parse_optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def read_rows(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines: list[str] = []
    for line in text.splitlines():
        marker = line.find(MARKER + ",")
        if marker >= 0:
            lines.append(line[marker + len(MARKER) + 1 :])

    if not lines:
        raise ValueError(f"no {MARKER} records found in {path}")

    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(
            "header mismatch: expected "
            f"{','.join(COLUMNS)}, found {','.join(reader.fieldnames or [])}"
        )

    rows: list[Row] = []
    for raw in reader:
        kernel_us = float(raw["kernel_us"])
        assignments_per_second = float(raw["assignments_per_second"])
        if not math.isfinite(kernel_us) or kernel_us <= 0:
            raise ValueError(f"{raw['identity']}: invalid kernel_us={kernel_us}")
        if not math.isfinite(assignments_per_second) or assignments_per_second <= 0:
            raise ValueError(
                f"{raw['identity']}: invalid assignments_per_second="
                f"{assignments_per_second}"
            )
        rows.append(
            Row(
                identity=raw["identity"],
                expected=raw["expected"],
                dimension=int(raw["dimension"]),
                signature=raw["signature"],
                prime=int(raw["prime"]),
                variables=int(raw["variables"]),
                nodes=int(raw["nodes"]),
                assignments=int(raw["assignments"]),
                cpu_checks=int(raw["cpu_checks"]),
                kernel_us=kernel_us,
                assignments_per_second=assignments_per_second,
                found_counterexample=parse_bool(
                    raw["found_counterexample"],
                    f"{raw['identity']}.found_counterexample",
                ),
                witness_assignment=parse_optional_int(raw["witness_assignment"]),
                witness_blade=parse_optional_int(raw["witness_blade"]),
                witness_lhs=parse_optional_int(raw["witness_lhs"]),
                witness_rhs=parse_optional_int(raw["witness_rhs"]),
                result=raw["result"],
            )
        )
    return rows


def validate(rows: list[Row], *, allow_dynamic_corpus: bool) -> list[str]:
    failures: list[str] = []
    names = [row.identity for row in rows]
    if len(names) != len(set(names)):
        failures.append("duplicate identity rows are present")

    if not allow_dynamic_corpus:
        missing = sorted(set(EXPECTED_NAMES) - set(names))
        extra = sorted(set(names) - set(EXPECTED_NAMES))
        if missing:
            failures.append(f"missing identities: {', '.join(missing)}")
        if extra:
            failures.append(f"unexpected identities: {', '.join(extra)}")

    assignments = {row.assignments for row in rows}
    if len(assignments) != 1:
        failures.append(f"inconsistent assignment counts: {sorted(assignments)}")

    for row in rows:
        expected = row.expected if allow_dynamic_corpus else EXPECTED_NAMES.get(row.identity)
        if expected is None:
            continue
        if expected not in {"identity", "counterexample"}:
            failures.append(
                f"{row.identity}: invalid expected label {expected!r}"
            )
            continue
        if not allow_dynamic_corpus and row.expected != expected:
            failures.append(
                f"{row.identity}: expected label {row.expected!r}, "
                f"corpus requires {expected!r}"
            )
        if row.result != "pass":
            failures.append(f"{row.identity}: result={row.result!r}")
        if row.dimension < 1 or row.dimension > 6:
            failures.append(f"{row.identity}: invalid dimension {row.dimension}")
        if row.prime < 3 or row.prime % 2 == 0:
            failures.append(f"{row.identity}: invalid prime {row.prime}")
        if row.variables <= 0 or row.nodes <= 0:
            failures.append(f"{row.identity}: variables/nodes must be positive")
        if row.cpu_checks > row.assignments:
            failures.append(f"{row.identity}: cpu_checks exceeds assignments")

        witness_values = (
            row.witness_assignment,
            row.witness_blade,
            row.witness_lhs,
            row.witness_rhs,
        )
        if expected == "identity":
            if row.found_counterexample:
                failures.append(f"{row.identity}: true identity produced a witness")
            if any(value is not None for value in witness_values):
                failures.append(
                    f"{row.identity}: identity row contains unexpected witness fields"
                )
        else:
            if not row.found_counterexample:
                failures.append(
                    f"{row.identity}: mutated false statement produced no witness"
                )
            if any(value is None for value in witness_values):
                failures.append(
                    f"{row.identity}: counterexample row has incomplete witness fields"
                )
            elif row.witness_assignment is not None and (
                row.witness_assignment < 0
                or row.witness_assignment >= row.assignments
            ):
                failures.append(
                    f"{row.identity}: witness assignment is outside searched range"
                )

    return failures


def write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def markdown(rows: list[Row]) -> str:
    ordered = sorted(rows, key=lambda row: row.identity)
    lines = [
        "| identity | expected | dim | signature | nodes | assignments | kernel us | assignments/s | witness | result |",
        "|---|---|---:|---|---:|---:|---:|---:|---|:---:|",
    ]
    for row in ordered:
        if row.found_counterexample:
            witness = (
                f"assignment {row.witness_assignment}, blade {row.witness_blade}, "
                f"{row.witness_lhs} != {row.witness_rhs}"
            )
        else:
            witness = "none"
        lines.append(
            f"| {row.identity} | {row.expected} | {row.dimension} | "
            f"`{row.signature}` | {row.nodes} | {row.assignments} | "
            f"{row.kernel_us:.6f} | {row.assignments_per_second:.3f} | "
            f"{witness} | {row.result} |"
        )

    total_assignments = sum(row.assignments for row in rows)
    total_kernel_seconds = sum(row.kernel_us for row in rows) * 1.0e-6
    aggregate_rate = (
        total_assignments / total_kernel_seconds
        if total_kernel_seconds > 0
        else 0.0
    )
    lines.extend(
        [
            "",
            f"- identities evaluated: {len(rows)}",
            f"- total exact modular assignments: {total_assignments}",
            f"- summed kernel time: {total_kernel_seconds:.6f} s",
            f"- aggregate assignment rate: {aggregate_rate:.3f} assignments/s",
            "",
            "A surviving identity is not proven over characteristic zero. "
            "A reported witness is an exact finite-field counterexample to the "
            "specified modular statement.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize exact geometric identity search output"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--allow-dynamic-corpus",
        action="store_true",
        help=(
            "validate arbitrary generated identity names using each row's declared "
            "identity/counterexample expectation instead of the fixed v1 corpus"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_rows(args.input)
        failures = validate(rows, allow_dynamic_corpus=args.allow_dynamic_corpus)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"VALIDATION: FAIL\n- {exc}", file=sys.stderr)
        return 1

    report = markdown(rows)
    print(report, end="")

    if args.summary_csv:
        write_csv(args.summary_csv, rows)
    if args.markdown_out:
        args.markdown_out.write_text(report, encoding="utf-8", newline="\n")

    if failures:
        print("VALIDATION: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
