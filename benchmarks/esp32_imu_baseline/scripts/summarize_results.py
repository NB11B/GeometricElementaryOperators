#!/usr/bin/env python3
"""Summarize and validate ESP32-C6 IMU replay benchmark output."""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

EXPECTED_IMPLEMENTATIONS = (
    "A0_geo_float_generic",
    "B0_geo_fixed_q16_generic",
    "A1_geo_float_fused",
    "B1_geo_fixed_q16_fused",
    "C_conventional_quaternion",
    "D_quantized_tinyml",
)

CSV_COLUMNS = (
    "implementation",
    "run",
    "state_bytes",
    "mean_us",
    "stddev_us",
    "min_us",
    "p50_us",
    "p95_us",
    "p99_us",
    "max_us",
    "deadline_misses",
    "mean_error_deg",
    "p95_error_deg",
    "max_error_deg",
    "nan_count",
    "output_hash",
    "min_free_heap",
    "largest_free_block",
)

NUMERIC_FIELDS = tuple(
    field for field in CSV_COLUMNS if field not in {"implementation", "output_hash"}
)


@dataclass(frozen=True)
class Summary:
    implementation: str
    runs: int
    state_bytes: int
    mean_us: float
    run_mean_stddev_us: float
    min_us: int
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: int
    deadline_misses: int
    mean_error_deg: float
    p95_error_deg: float
    max_error_deg: float
    nan_count: int
    stable_hash: bool
    unique_hashes: int
    min_free_heap: int
    largest_free_block: int


def read_text_auto(path: Path) -> str:
    """Read UTF-8 or Windows PowerShell UTF-16 output without data loss."""
    raw = path.read_bytes()
    if not raw:
        return ""

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    sample = raw[:1024]
    if b"\x00" in sample:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return raw.decode("utf-16", errors="replace")

    return raw.decode("utf-8", errors="replace")


def extract_csv_lines(path: Path) -> list[str]:
    """Extract only complete benchmark CSV records from a log or CSV file."""
    records: list[str] = []
    expected_fields = len(CSV_COLUMNS) + 1  # leading CSV marker

    for raw_line in read_text_auto(path).splitlines():
        marker = raw_line.find("CSV,")
        if marker < 0:
            continue

        candidate = raw_line[marker:]
        try:
            values = next(csv.reader([candidate]))
        except csv.Error:
            continue

        if len(values) == expected_fields and values[0] == "CSV":
            records.append(candidate)

    if not records:
        raise ValueError(f"no complete CSV records found in {path}")
    return records


def parse_rows(lines: Iterable[str]) -> list[dict[str, str]]:
    normalized = [line[4:] if line.startswith("CSV,") else line for line in lines]
    reader = csv.DictReader(io.StringIO("\n".join(normalized)))
    rows = list(reader)

    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        raise ValueError(
            "CSV header mismatch: expected "
            f"{','.join(CSV_COLUMNS)}, found {','.join(reader.fieldnames or [])}"
        )

    for row in rows:
        for field in CSV_COLUMNS:
            if row.get(field) in (None, ""):
                raise ValueError(f"missing field {field!r} in row {row}")
        for field in NUMERIC_FIELDS:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field!r} in row {row}")

    return rows


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def as_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in row {row}")
    return value


def summarize_group(name: str, rows: list[dict[str, str]]) -> Summary:
    state_sizes = {as_int(row, "state_bytes") for row in rows}
    if len(state_sizes) != 1:
        raise ValueError(f"{name}: inconsistent state_bytes values: {state_sizes}")

    hashes = {row["output_hash"].lower() for row in rows}
    mean_values = [as_float(row, "mean_us") for row in rows]

    return Summary(
        implementation=name,
        runs=len(rows),
        state_bytes=next(iter(state_sizes)),
        mean_us=statistics.fmean(mean_values),
        run_mean_stddev_us=statistics.pstdev(mean_values) if len(rows) > 1 else 0.0,
        min_us=min(as_int(row, "min_us") for row in rows),
        p50_us=statistics.median(as_float(row, "p50_us") for row in rows),
        p95_us=statistics.median(as_float(row, "p95_us") for row in rows),
        p99_us=statistics.median(as_float(row, "p99_us") for row in rows),
        max_us=max(as_int(row, "max_us") for row in rows),
        deadline_misses=sum(as_int(row, "deadline_misses") for row in rows),
        mean_error_deg=statistics.fmean(
            as_float(row, "mean_error_deg") for row in rows
        ),
        p95_error_deg=statistics.fmean(
            as_float(row, "p95_error_deg") for row in rows
        ),
        max_error_deg=max(as_float(row, "max_error_deg") for row in rows),
        nan_count=sum(as_int(row, "nan_count") for row in rows),
        stable_hash=len(hashes) == 1,
        unique_hashes=len(hashes),
        min_free_heap=min(as_int(row, "min_free_heap") for row in rows),
        largest_free_block=min(as_int(row, "largest_free_block") for row in rows),
    )


def build_summaries(rows: list[dict[str, str]]) -> dict[str, Summary]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["implementation"]].append(row)
    return {name: summarize_group(name, group) for name, group in grouped.items()}


def validate_rows(
    rows: list[dict[str, str]],
    summaries: dict[str, Summary],
    expected_runs: int,
    require_all: bool,
) -> list[str]:
    failures: list[str] = []

    if require_all:
        missing = [name for name in EXPECTED_IMPLEMENTATIONS if name not in summaries]
        extra = [name for name in summaries if name not in EXPECTED_IMPLEMENTATIONS]
        if missing:
            failures.append(f"missing implementations: {', '.join(missing)}")
        if extra:
            failures.append(f"unexpected implementations: {', '.join(extra)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["implementation"]].append(row)

    expected_run_ids = set(range(expected_runs))
    for name, summary in summaries.items():
        run_ids = {as_int(row, "run") for row in grouped[name]}
        if summary.runs != expected_runs:
            failures.append(f"{name}: expected {expected_runs} rows, found {summary.runs}")
        if run_ids != expected_run_ids:
            failures.append(
                f"{name}: run IDs are incomplete or duplicated; found {sorted(run_ids)}"
            )
        if summary.deadline_misses != 0:
            failures.append(f"{name}: deadline_misses={summary.deadline_misses}")
        if summary.nan_count != 0:
            failures.append(f"{name}: nan_count={summary.nan_count}")
        if not summary.stable_hash:
            failures.append(
                f"{name}: output hash changed across runs "
                f"({summary.unique_hashes} unique hashes)"
            )

    return failures


def speedup(reference_us: float, candidate_us: float) -> float:
    return reference_us / candidate_us


def reduction(reference_us: float, candidate_us: float) -> float:
    return (1.0 - candidate_us / reference_us) * 100.0


def markdown_report(summaries: dict[str, Summary]) -> str:
    ordered = [name for name in EXPECTED_IMPLEMENTATIONS if name in summaries]
    ordered.extend(sorted(name for name in summaries if name not in ordered))

    lines = [
        "| implementation | runs | state B | mean us | run-mean sd | p50 | p95 | p99 | max | mean error deg | p95 error | max error | misses | NaNs | stable hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for name in ordered:
        summary = summaries[name]
        lines.append(
            f"| {name} | {summary.runs} | {summary.state_bytes} | "
            f"{summary.mean_us:.6f} | {summary.run_mean_stddev_us:.6f} | "
            f"{summary.p50_us:.1f} | {summary.p95_us:.1f} | "
            f"{summary.p99_us:.1f} | {summary.max_us} | "
            f"{summary.mean_error_deg:.6f} | {summary.p95_error_deg:.6f} | "
            f"{summary.max_error_deg:.6f} | {summary.deadline_misses} | "
            f"{summary.nan_count} | {'yes' if summary.stable_hash else 'no'} |"
        )

    comparisons: list[str] = []
    a0 = summaries.get("A0_geo_float_generic")
    b0 = summaries.get("B0_geo_fixed_q16_generic")
    a1 = summaries.get("A1_geo_float_fused")
    b1 = summaries.get("B1_geo_fixed_q16_fused")
    conventional = summaries.get("C_conventional_quaternion")
    tinyml = summaries.get("D_quantized_tinyml")

    if a0 and a1:
        comparisons.append(
            f"- A1 versus A0: {speedup(a0.mean_us, a1.mean_us):.3f}x throughput; "
            f"{reduction(a0.mean_us, a1.mean_us):.2f}% lower mean latency."
        )
    if b0 and b1:
        comparisons.append(
            f"- B1 versus B0: {speedup(b0.mean_us, b1.mean_us):.3f}x throughput; "
            f"{reduction(b0.mean_us, b1.mean_us):.2f}% lower mean latency."
        )
    if conventional and a1:
        comparisons.append(
            f"- A1 versus C: {speedup(conventional.mean_us, a1.mean_us):.3f}x "
            "relative throughput; values above 1 mean A1 is faster."
        )
    if conventional and b1:
        comparisons.append(
            f"- B1 versus C: {speedup(conventional.mean_us, b1.mean_us):.3f}x "
            "relative throughput; values above 1 mean B1 is faster."
        )
    if tinyml and a1:
        comparisons.append(
            f"- A1 versus D: {speedup(tinyml.mean_us, a1.mean_us):.3f}x throughput."
        )
    if tinyml and b1:
        comparisons.append(
            f"- B1 versus D: {speedup(tinyml.mean_us, b1.mean_us):.3f}x throughput."
        )

    if comparisons:
        lines.extend(["", "## Comparisons", "", *comparisons])

    return "\n".join(lines) + "\n"


def write_summary_csv(path: Path, summaries: dict[str, Summary]) -> None:
    ordered = [name for name in EXPECTED_IMPLEMENTATIONS if name in summaries]
    ordered.extend(sorted(name for name in summaries if name not in ordered))
    fieldnames = list(Summary.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in ordered:
            writer.writerow(asdict(summaries[name]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ESP32-C6 GEO IMU replay CSV or monitor log"
    )
    parser.add_argument("input", type=Path, help="monitor log or extracted CSV")
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=30,
        help="required run count per implementation (default: 30)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="do not require all six expected implementations",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="write the Markdown summary to this file",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="write the aggregate CSV to this file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        lines = extract_csv_lines(args.input)
        rows = parse_rows(lines)
        summaries = build_summaries(rows)
        failures = validate_rows(
            rows,
            summaries,
            expected_runs=args.expected_runs,
            require_all=not args.allow_partial,
        )
        report = markdown_report(summaries)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(report, end="")

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report, encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(args.csv_out, summaries)

    if failures:
        print("VALIDATION: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
