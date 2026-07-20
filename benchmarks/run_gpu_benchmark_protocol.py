#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(paths: list[Path], output_csv: Path, output_json: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    checksums: dict[tuple[str, str, str, str, str], list[float]] = {}
    for path in paths:
        for row in load(path):
            key = (
                row["backend"],
                row["mode"],
                row["timing_class"],
                row["dimension"],
                row["batch"],
            )
            groups.setdefault(key, []).append(float(row["samples_per_second"]))
            checksums.setdefault(key, []).append(float(row["checksum"]))

    rows: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for key in sorted(groups):
        values = groups[key]
        checksum_values = checksums[key]
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        backend, mode, timing_class, dimension, batch = key
        rows.append({
            "backend": backend,
            "mode": mode,
            "timing_class": timing_class,
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
            "timing_class": timing_class,
            "dimension": int(dimension),
            "batch": int(batch),
            "samples_per_second_trials": values,
            "checksum_trials": checksum_values,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo-exe", required=True, type=Path)
    parser.add_argument("--pytorch-script", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    if args.trials < 3:
        raise SystemExit("trials must be at least 3")
    if not args.geo_exe.exists():
        raise SystemExit(f"missing GEO executable: {args.geo_exe}")
    if not args.pytorch_script.exists():
        raise SystemExit(f"missing PyTorch script: {args.pytorch_script}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geo_paths: list[Path] = []
    eager_paths: list[Path] = []
    compile_paths: list[Path] = []

    for trial in range(args.trials):
        geo = args.out_dir / f"geo-cuda-trial-{trial:02d}.csv"
        eager = args.out_dir / f"pytorch-eager-cuda-trial-{trial:02d}.csv"
        run([str(args.geo_exe), str(geo), str(args.iterations)])
        run([
            sys.executable,
            str(args.pytorch_script),
            "--backend", "eager",
            "--iterations", str(args.iterations),
            "--out", str(eager),
        ])
        geo_paths.append(geo)
        eager_paths.append(eager)

        if not args.skip_compile:
            compiled = args.out_dir / f"pytorch-compile-cuda-trial-{trial:02d}.csv"
            run([
                sys.executable,
                str(args.pytorch_script),
                "--backend", "compile",
                "--iterations", str(args.iterations),
                "--out", str(compiled),
            ])
            compile_paths.append(compiled)

    aggregate(geo_paths, args.out_dir / "geo-cuda.csv", args.out_dir / "geo-cuda-raw-trials.json")
    aggregate(eager_paths, args.out_dir / "pytorch-eager-cuda.csv", args.out_dir / "pytorch-eager-cuda-raw-trials.json")
    if compile_paths:
        aggregate(
            compile_paths,
            args.out_dir / "pytorch-compile-cuda.csv",
            args.out_dir / "pytorch-compile-cuda-raw-trials.json",
        )

    print(
        "GEO_GPU_BENCHMARK_PROTOCOL: PASS "
        f"trials={args.trials} iterations={args.iterations} "
        f"torch_compile={'false' if args.skip_compile else 'true'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
