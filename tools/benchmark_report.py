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
from pathlib import Path
from typing import Iterable

COMPARE_RE = re.compile(r"^(?P<name>.{1,64}?)\s+(?P<value>[0-9]+(?:\.[0-9]+)?) ns/op$")
HOST_RE = re.compile(r"^(?P<name>.{1,64}?)\s+iterations=.*?ns/op=(?P<value>[0-9]+(?:\.[0-9]+)?)$")


def run_one(executable: Path) -> dict[str, float]:
    completed = subprocess.run(
        [str(executable)], check=True, text=True, capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    values: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        match = COMPARE_RE.match(line) or HOST_RE.match(line)
        if match:
            values[match.group("name").strip()] = float(match.group("value"))
    if not values:
        raise RuntimeError(f"no benchmark rows parsed from {executable}\n{completed.stdout}")
    return values


def summarize(samples: list[dict[str, float]]) -> list[dict[str, float | str | int]]:
    names = sorted(set.intersection(*(set(sample) for sample in samples)))
    rows: list[dict[str, float | str | int]] = []
    for name in names:
        values = [sample[name] for sample in samples]
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


def markdown_report(metadata: dict, rows: list[dict]) -> str:
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
        f"- Repetitions: `{metadata['repetitions']}`",
        "",
        "## Results",
        "",
        "| Benchmark | Samples | Min ns | Median ns | Mean ns | Max ns | Stddev ns | Spread % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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

    all_samples: list[dict[str, float]] = []
    for executable in args.executables:
        for _ in range(args.repetitions):
            all_samples.append(run_one(executable))

    rows = summarize(all_samples)
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
    payload = {"metadata": metadata, "results": rows}
    (args.out_dir / "benchmark_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / "benchmark_report.md").write_text(markdown_report(metadata, rows), encoding="utf-8")
    with (args.out_dir / "benchmark_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["benchmark"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
