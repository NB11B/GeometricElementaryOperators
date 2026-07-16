#!/usr/bin/env python3
"""Run benchmark executables repeatedly and emit CSV, JSON, and Markdown reports."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

COMPARE_RE = re.compile(r"^(?P<name>.{1,64}?)\s+(?P<value>[0-9]+(?:\.[0-9]+)?) ns/op$")
HOST_RE = re.compile(r"^(?P<name>.{1,64}?)\s+iterations=.*?ns/op=(?P<value>[0-9]+(?:\.[0-9]+)?)$")
MATRIX_RE = re.compile(
    r"^GEO_BENCH_MATRIX\|operation=(?P<operation>[a-z0-9_]+)\|path=(?P<path>[a-z0-9_]+)$"
)


def run_one(executable: Path) -> tuple[dict[str, float], list[dict[str, str]]]:
    completed = subprocess.run(
        [str(executable)], check=True, text=True, capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    values: dict[str, float] = {}
    matrix: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        timing_match = COMPARE_RE.match(line) or HOST_RE.match(line)
        if timing_match:
            values[timing_match.group("name").strip()] = float(timing_match.group("value"))
            continue
        matrix_match = MATRIX_RE.match(line)
        if matrix_match:
            matrix.append({
                "operation": matrix_match.group("operation"),
                "path": matrix_match.group("path"),
            })
    if not values:
        raise RuntimeError(f"no benchmark rows parsed from {executable}\n{completed.stdout}")
    if not matrix:
        raise RuntimeError(f"no operation/path matrix parsed from {executable}\n{completed.stdout}")
    if len({(row['operation'], row['path']) for row in matrix}) != len(matrix):
        raise RuntimeError(f"duplicate operation/path matrix row from {executable}")
    return values, matrix


def summarize(samples_by_name: dict[str, list[float]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for name in sorted(samples_by_name):
        values = samples_by_name[name]
        median = statistics.median(values)
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append({
            "benchmark": name,
            "samples": len(values),
            "min_ns": min(values),
            "median_ns": median,
            "mean_ns": mean,
            "max_ns": max(values),
            "stdev_ns": stdev,
            "relative_spread_percent": 0.0 if median == 0 else 100.0 * (max(values) - min(values)) / median,
        })
    return rows


def markdown_report(metadata: dict, matrix: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Geometric Elementary Operators benchmark report",
        "",
        f"Generated: `{metadata['generated_utc']}`",
        "",
        "## Environment",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Processor: `{metadata['processor']}`",
        f"- Python: `{metadata['python']}`",
        f"- Repetitions per executable: `{metadata['repetitions']}`",
        "",
        "## Exact operation/path matrix",
        "",
        "| Executable | Operation | Path |",
        "|---|---|---|",
    ]
    for row in matrix:
        lines.append(f"| {row['executable']} | {row['operation']} | {row['path']} |")
    lines.extend([
        "",
        "## Results",
        "",
        "| Benchmark | Samples | Min ns | Median ns | Mean ns | Max ns | Stddev ns | Spread % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['samples']} | {row['min_ns']:.3f} | "
            f"{row['median_ns']:.3f} | {row['mean_ns']:.3f} | {row['max_ns']:.3f} | "
            f"{row['stdev_ns']:.3f} | {row['relative_spread_percent']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "The median is the primary comparison statistic. Results from shared CI runners are retained as regression evidence, not as definitive hardware-performance claims. Publishable comparisons should use a pinned machine, fixed power policy, release builds, and multiple independent process launches.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executables", nargs="+", type=Path)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repetitions < 3:
        parser.error("repetitions must be at least 3")
    for executable in args.executables:
        if not executable.exists():
            parser.error(f"missing executable: {executable}")

    samples_by_name: dict[str, list[float]] = defaultdict(list)
    operation_path_matrix: list[dict[str, str]] = []
    for executable in args.executables:
        prefix = executable.stem
        expected_matrix: list[dict[str, str]] | None = None
        for _ in range(args.repetitions):
            values, matrix = run_one(executable)
            if expected_matrix is None:
                expected_matrix = matrix
            elif matrix != expected_matrix:
                raise RuntimeError(f"operation/path matrix changed between runs for {executable}")
            for name, value in values.items():
                samples_by_name[f"{prefix}: {name}"].append(value)
        assert expected_matrix is not None
        operation_path_matrix.extend(
            {"executable": prefix, "operation": row["operation"], "path": row["path"]}
            for row in expected_matrix
        )

    rows = summarize(samples_by_name)
    metadata = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "repetitions": args.repetitions,
        "executables": [str(path) for path in args.executables],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "operation_path_matrix": operation_path_matrix,
        "results": rows,
    }
    (args.out_dir / "benchmark_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / "benchmark_report.md").write_text(
        markdown_report(metadata, operation_path_matrix, rows), encoding="utf-8"
    )
    with (args.out_dir / "benchmark_report.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["benchmark"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
