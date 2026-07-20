#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
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

    taskset = subprocess.run(["sh", "-c", "command -v taskset"], capture_output=True, text=True).returncode == 0
    prefix = ["taskset", "-c", str(args.cpu)] if taskset else []
    (args.out_dir / "affinity.txt").write_text(
        f"taskset_available={str(taskset).lower()}\nrequested_cpu={args.cpu}\n",
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
        run(prefix + ["python", str(args.pytorch_script), "--out", str(pytorch_path)], env)
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
    print(f"GEO_CPU_BENCHMARK_PROTOCOL: PASS trials={args.trials} affinity={taskset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
