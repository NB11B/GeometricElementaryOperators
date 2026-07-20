#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(raw_files: list[Path], output: Path, raw_json: Path) -> None:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    checksums: dict[tuple[str, str, str, str], list[float]] = {}
    for path in raw_files:
        for row in load(path):
            key = (row["backend"], row["mode"], row["dimension"], row["batch"])
            groups.setdefault(key, []).append(float(row["samples_per_second"]))
            checksums.setdefault(key, []).append(float(row["checksum"]))

    rows: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for key in sorted(groups):
        values = groups[key]
        checksum_values = checksums[key]
        backend, mode, dimension, batch = key
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        rows.append({
            "backend": backend,
            "mode": mode,
            "dimension": dimension,
            "batch": batch,
            "samples_per_second": f"{median:.17g}",
            "checksum": f"{statistics.median(checksum_values):.17g}",
            "trial_count": len(values),
            "minimum_samples_per_second": f"{min(values):.17g}",
            "maximum_samples_per_second": f"{max(values):.17g}",
            "mean_samples_per_second": f"{statistics.fmean(values):.17g}",
            "stddev_samples_per_second": f"{statistics.pstdev(values):.17g}",
            "median_absolute_deviation": f"{mad:.17g}",
        })
        raw.append({
            "backend": backend,
            "mode": mode,
            "dimension": int(dimension),
            "batch": int(batch),
            "samples_per_second_trials": values,
            "checksum_trials": checksum_values,
        })

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_json.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def affinity_prefix(cpu: int) -> tuple[list[str], str]:
    taskset = shutil.which("taskset")
    if os.name != "nt" and taskset is not None:
        return [taskset, "-c", str(cpu)], "taskset"
    if os.name == "nt":
        return [], "unsupported_on_windows"
    return [], "taskset_unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--pytorch-script", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    if args.trials < 3:
        raise SystemExit("trials must be at least 3")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    prefix, affinity_mode = affinity_prefix(args.cpu)
    (args.out_dir / "affinity.txt").write_text(
        "\n".join([
            f"platform={sys.platform}",
            f"affinity_mode={affinity_mode}",
            f"requested_cpu={args.cpu}",
            f"affinity_applied={str(bool(prefix)).lower()}",
            "",
        ]),
        encoding="utf-8",
    )

    baseline_raw: list[Path] = []
    optimized_raw: list[Path] = []
    pytorch_raw: list[Path] = []
    for trial in range(args.trials):
        baseline_path = args.out_dir / f"geo-baseline-trial-{trial:02d}.csv"
        optimized_path = args.out_dir / f"geo-optimized-trial-{trial:02d}.csv"
        pytorch_path = args.out_dir / f"pytorch-trial-{trial:02d}.csv"
        run(prefix + [str(args.baseline), str(baseline_path)], env)
        run(prefix + [str(args.optimized), str(optimized_path)], env)
        run(prefix + [sys.executable, str(args.pytorch_script), "--out", str(pytorch_path)], env)
        baseline_raw.append(baseline_path)
        optimized_raw.append(optimized_path)
        pytorch_raw.append(pytorch_path)

    aggregate(
        baseline_raw,
        args.out_dir / "geo-baseline.csv",
        args.out_dir / "geo-baseline-raw-trials.json",
    )
    aggregate(
        optimized_raw,
        args.out_dir / "geo-optimized.csv",
        args.out_dir / "geo-optimized-raw-trials.json",
    )
    aggregate(
        pytorch_raw,
        args.out_dir / "pytorch.csv",
        args.out_dir / "pytorch-raw-trials.json",
    )
    print(
        "GEO_CPU_BENCHMARK_PROTOCOL: PASS "
        f"trials={args.trials} affinity_mode={affinity_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
