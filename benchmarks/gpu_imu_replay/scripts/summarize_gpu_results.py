#!/usr/bin/env python3
"""Summarize and validate CUDA IMU replay benchmark CSV output."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

EXPECTED_IMPLEMENTATIONS = (
    "A1_geo_float_generated_gpu",
    "C_conventional_quaternion_gpu",
)

EXPECTED_MODE = "resident_replay"
MARKER = "CSV_GPU"
SAMPLE_COUNT = 12000

CSV_COLUMNS = (
    "implementation",
    "mode",
    "run",
    "batch",
    "state_bytes",
    "kernel_us",
    "ns_per_sample_trajectory",
    "trajectories_per_second",
    "samples_per_second",
    "mean_error_deg",
    "p95_error_deg",
    "max_error_deg",
    "nan_count",
    "trace_hash",
    "batch_hash",
    "device_bytes",
)


@dataclass(frozen=True)
class Summary:
    batch: int
    implementation: str
    runs: int
    state_bytes: int
    mean_kernel_us: float
    run_mean_stddev_us: float
    min_kernel_us: float
    max_kernel_us: float
    mean_ns_per_sample_trajectory: float
    mean_trajectories_per_second: float
    mean_samples_per_second: float
    mean_error_deg: float
    p95_error_deg: float
    max_error_deg: float
    nan_count: int
    stable_trace_hash: bool
    trace_hash: str
    stable_batch_hash: bool
    batch_hash: str
    device_bytes: int


@dataclass(frozen=True)
class Parity:
    batch: int
    runs_per_path: int
    a1_mean_us: float
    c_mean_us: float
    a1_vs_c_percent: float
    trace_hash_match: bool
    accuracy_match: bool
    zero_nans: bool
    stable_hashes: bool


def read_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    normalized: list[str] = []
    for line in text.splitlines():
        marker = line.find(MARKER + ",")
        if marker < 0:
            continue
        normalized.append(line[marker + len(MARKER) + 1 :])

    if not normalized:
        raise ValueError(f"no {MARKER} records found in {path}")

    reader = csv.DictReader(normalized)
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        raise ValueError(
            "CSV header mismatch: expected "
            f"{','.join(CSV_COLUMNS)}, found {','.join(reader.fieldnames or [])}"
        )

    rows = list(reader)
    if not rows:
        raise ValueError("CSV contains a header but no data rows")
    return rows


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def as_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in row {row}")
    return value


def summarize(rows: list[dict[str, str]]) -> tuple[list[Summary], list[Parity], list[str]]:
    failures: list[str] = []
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row["mode"] != EXPECTED_MODE:
            failures.append(
                f"{row['implementation']} batch {row['batch']}: unexpected mode {row['mode']}"
            )
        grouped[(as_int(row, "batch"), row["implementation"])].append(row)

    batches = sorted({batch for batch, _ in grouped})
    summaries: list[Summary] = []

    for batch in batches:
        implementations = {name for b, name in grouped if b == batch}
        missing = [name for name in EXPECTED_IMPLEMENTATIONS if name not in implementations]
        extra = sorted(implementations - set(EXPECTED_IMPLEMENTATIONS))
        if missing:
            failures.append(f"batch {batch}: missing implementations: {', '.join(missing)}")
        if extra:
            failures.append(f"batch {batch}: unexpected implementations: {', '.join(extra)}")

        for implementation in EXPECTED_IMPLEMENTATIONS:
            group = grouped.get((batch, implementation), [])
            if not group:
                continue

            run_ids = [as_int(row, "run") for row in group]
            expected_ids = list(range(len(group)))
            if sorted(run_ids) != expected_ids:
                failures.append(
                    f"{implementation} batch {batch}: run IDs are incomplete or duplicated: {sorted(run_ids)}"
                )

            states = {as_int(row, "state_bytes") for row in group}
            device_sizes = {as_int(row, "device_bytes") for row in group}
            trace_hashes = {row["trace_hash"].lower() for row in group}
            batch_hashes = {row["batch_hash"].lower() for row in group}
            nan_count = sum(as_int(row, "nan_count") for row in group)

            if len(states) != 1:
                failures.append(
                    f"{implementation} batch {batch}: inconsistent state sizes: {sorted(states)}"
                )
            if len(device_sizes) != 1:
                failures.append(
                    f"{implementation} batch {batch}: inconsistent device byte counts"
                )
            if len(trace_hashes) != 1:
                failures.append(
                    f"{implementation} batch {batch}: trace hash changed across runs"
                )
            if len(batch_hashes) != 1:
                failures.append(
                    f"{implementation} batch {batch}: batch hash changed across runs"
                )
            if nan_count != 0:
                failures.append(
                    f"{implementation} batch {batch}: nan_count={nan_count}"
                )

            kernel_values = [as_float(row, "kernel_us") for row in group]
            summaries.append(
                Summary(
                    batch=batch,
                    implementation=implementation,
                    runs=len(group),
                    state_bytes=next(iter(states)),
                    mean_kernel_us=statistics.fmean(kernel_values),
                    run_mean_stddev_us=(
                        statistics.pstdev(kernel_values) if len(kernel_values) > 1 else 0.0
                    ),
                    min_kernel_us=min(kernel_values),
                    max_kernel_us=max(kernel_values),
                    mean_ns_per_sample_trajectory=statistics.fmean(
                        as_float(row, "ns_per_sample_trajectory") for row in group
                    ),
                    mean_trajectories_per_second=statistics.fmean(
                        as_float(row, "trajectories_per_second") for row in group
                    ),
                    mean_samples_per_second=statistics.fmean(
                        as_float(row, "samples_per_second") for row in group
                    ),
                    mean_error_deg=statistics.fmean(
                        as_float(row, "mean_error_deg") for row in group
                    ),
                    p95_error_deg=statistics.fmean(
                        as_float(row, "p95_error_deg") for row in group
                    ),
                    max_error_deg=max(as_float(row, "max_error_deg") for row in group),
                    nan_count=nan_count,
                    stable_trace_hash=len(trace_hashes) == 1,
                    trace_hash=next(iter(trace_hashes)),
                    stable_batch_hash=len(batch_hashes) == 1,
                    batch_hash=next(iter(batch_hashes)),
                    device_bytes=next(iter(device_sizes)),
                )
            )

    by_key = {(summary.batch, summary.implementation): summary for summary in summaries}
    parity_rows: list[Parity] = []
    for batch in batches:
        a1 = by_key.get((batch, EXPECTED_IMPLEMENTATIONS[0]))
        conventional = by_key.get((batch, EXPECTED_IMPLEMENTATIONS[1]))
        if a1 is None or conventional is None:
            continue

        accuracy_match = (
            math.isclose(a1.mean_error_deg, conventional.mean_error_deg, abs_tol=1.0e-9)
            and math.isclose(a1.p95_error_deg, conventional.p95_error_deg, abs_tol=1.0e-9)
            and math.isclose(a1.max_error_deg, conventional.max_error_deg, abs_tol=1.0e-9)
        )
        trace_hash_match = a1.trace_hash == conventional.trace_hash
        stable_hashes = (
            a1.stable_trace_hash
            and conventional.stable_trace_hash
            and a1.stable_batch_hash
            and conventional.stable_batch_hash
        )
        zero_nans = a1.nan_count == 0 and conventional.nan_count == 0

        if a1.runs != conventional.runs:
            failures.append(f"batch {batch}: A1 and C run counts differ")
        if not trace_hash_match:
            failures.append(f"batch {batch}: A1 and C trace hashes differ")
        if not accuracy_match:
            failures.append(f"batch {batch}: A1 and C accuracy metrics differ")
        if not zero_nans:
            failures.append(f"batch {batch}: A1 or C produced NaNs")
        if not stable_hashes:
            failures.append(f"batch {batch}: A1 or C hashes were unstable")

        parity_rows.append(
            Parity(
                batch=batch,
                runs_per_path=a1.runs,
                a1_mean_us=a1.mean_kernel_us,
                c_mean_us=conventional.mean_kernel_us,
                a1_vs_c_percent=100.0 * (a1.mean_kernel_us / conventional.mean_kernel_us - 1.0),
                trace_hash_match=trace_hash_match,
                accuracy_match=accuracy_match,
                zero_nans=zero_nans,
                stable_hashes=stable_hashes,
            )
        )

    return summaries, parity_rows, failures


def write_csv(path: Path, records: list[object]) -> None:
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def markdown_report(summaries: list[Summary], parity_rows: list[Parity]) -> str:
    lines = [
        "| batch | implementation | runs | mean kernel us | run sd us | ns/sample-trajectory | trajectories/s | samples/s | mean error deg | trace hash | batch hash |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.batch} | {summary.implementation} | {summary.runs} | "
            f"{summary.mean_kernel_us:.6f} | {summary.run_mean_stddev_us:.6f} | "
            f"{summary.mean_ns_per_sample_trajectory:.9f} | "
            f"{summary.mean_trajectories_per_second:.3f} | "
            f"{summary.mean_samples_per_second:.3f} | "
            f"{summary.mean_error_deg:.6f} | {summary.trace_hash} | {summary.batch_hash} |"
        )

    lines.extend([
        "",
        "## A1 versus C parity",
        "",
        "| batch | runs/path | A1 mean us | C mean us | A1 vs C % | trace hash match | accuracy match | zero NaNs | stable hashes |",
        "|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ])
    for parity in parity_rows:
        lines.append(
            f"| {parity.batch} | {parity.runs_per_path} | {parity.a1_mean_us:.6f} | "
            f"{parity.c_mean_us:.6f} | {parity.a1_vs_c_percent:.6f} | "
            f"{'yes' if parity.trace_hash_match else 'no'} | "
            f"{'yes' if parity.accuracy_match else 'no'} | "
            f"{'yes' if parity.zero_nans else 'no'} | "
            f"{'yes' if parity.stable_hashes else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and validate CUDA IMU replay CSV output"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--parity-csv", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_rows(args.input)
        summaries, parity_rows, failures = summarize(rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"VALIDATION: FAIL\n- {exc}", file=sys.stderr)
        return 1

    report = markdown_report(summaries, parity_rows)
    print(report, end="")

    if args.summary_csv:
        write_csv(args.summary_csv, summaries)
    if args.parity_csv:
        write_csv(args.parity_csv, parity_rows)
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
