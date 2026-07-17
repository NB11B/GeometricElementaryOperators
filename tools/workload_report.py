#!/usr/bin/env python3
"""Run CPU and optional CUDA workloads and emit percentile/error reports."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CPU_RESULT_RE = re.compile(
    r"^RESULT operation=(?P<operation>\S+) backend=(?P<backend>\S+) "
    r"ns_per_op=(?P<ns>[0-9eE+\-.]+) "
    r"max_absolute=(?P<max_abs>[0-9eE+\-.]+) "
    r"max_relative=(?P<max_rel>[0-9eE+\-.]+) "
    r"mismatches=(?P<mismatches>[0-9]+)$"
)
CUDA_DEVICE_RE = re.compile(
    r"^device: (?P<name>.+) \(compute (?P<major>[0-9]+)\."
    r"(?P<minor>[0-9]+), (?P<sms>[0-9]+) SMs, runtime "
    r"(?P<runtime>[0-9]+), driver (?P<driver>[0-9]+)\)$"
)


@dataclass
class Samples:
    timings_ns: list[float] = field(default_factory=list)
    max_absolute: float = 0.0
    max_relative: float = 0.0
    mismatches: int = 0

    def add(
        self,
        timing_ns: float,
        max_absolute: float,
        max_relative: float,
        mismatches: int,
    ) -> None:
        if not math.isfinite(timing_ns) or timing_ns < 0.0:
            raise ValueError(f"invalid timing sample: {timing_ns}")
        if not math.isfinite(max_absolute) or max_absolute < 0.0:
            raise ValueError(f"invalid absolute error: {max_absolute}")
        if not math.isfinite(max_relative) or max_relative < 0.0:
            raise ValueError(f"invalid relative error: {max_relative}")
        if mismatches < 0:
            raise ValueError(f"invalid mismatch count: {mismatches}")
        self.timings_ns.append(timing_ns)
        self.max_absolute = max(self.max_absolute, max_absolute)
        self.max_relative = max(self.max_relative, max_relative)
        self.mismatches += mismatches


@dataclass(frozen=True)
class CudaDeviceInfo:
    name: str
    compute_major: int
    compute_minor: int
    multiprocessor_count: int
    runtime_version: int
    driver_version: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "compute_major": self.compute_major,
            "compute_minor": self.compute_minor,
            "multiprocessor_count": self.multiprocessor_count,
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
        }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def run_command(
    command: list[str],
    *,
    allow_skip: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if allow_skip and completed.returncode == 77:
        return completed
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def collect_cpu(
    executable: Path,
    repetitions: int,
    iterations: int,
    warmup: int,
    seed: int,
    precision: str,
    samples: dict[tuple[str, str, str, int], Samples],
) -> None:
    expected_rows: set[tuple[str, str]] | None = None
    for _ in range(repetitions):
        completed = run_command([
            str(executable),
            "--iterations", str(iterations),
            "--warmup", str(warmup),
            "--seed", str(seed),
        ])
        rows: set[tuple[str, str]] = set()
        for line in completed.stdout.splitlines():
            match = CPU_RESULT_RE.match(line.strip())
            if match is None:
                continue
            operation = match.group("operation")
            backend = match.group("backend")
            row_key = (operation, backend)
            if row_key in rows:
                raise RuntimeError(
                    f"duplicate CPU result row {operation}/{backend} from {executable}"
                )
            rows.add(row_key)
            samples[(operation, backend, precision, 1)].add(
                float(match.group("ns")),
                float(match.group("max_abs")),
                float(match.group("max_rel")),
                int(match.group("mismatches")),
            )
        if not rows:
            raise RuntimeError(
                f"no workload result rows parsed from {executable}\n"
                f"{completed.stdout}"
            )
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise RuntimeError(
                "CPU workload operation/backend matrix changed between repetitions: "
                f"expected={sorted(expected_rows)} actual={sorted(rows)}"
            )


def parse_cuda_device(stdout: str) -> CudaDeviceInfo:
    matches = [
        CUDA_DEVICE_RE.match(line.strip())
        for line in stdout.splitlines()
    ]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise RuntimeError(
            "CUDA harness must emit exactly one device metadata line; "
            f"found {len(matches)}\n{stdout}"
        )
    match = matches[0]
    assert match is not None
    return CudaDeviceInfo(
        name=match.group("name"),
        compute_major=int(match.group("major")),
        compute_minor=int(match.group("minor")),
        multiprocessor_count=int(match.group("sms")),
        runtime_version=int(match.group("runtime")),
        driver_version=int(match.group("driver")),
    )


def merge_samples(
    destination: dict[tuple[str, str, str, int], Samples],
    source: dict[tuple[str, str, str, int], Samples],
) -> None:
    for key, incoming in source.items():
        existing = destination[key]
        existing.timings_ns.extend(incoming.timings_ns)
        existing.max_absolute = max(existing.max_absolute, incoming.max_absolute)
        existing.max_relative = max(existing.max_relative, incoming.max_relative)
        existing.mismatches += incoming.mismatches


def collect_cuda(
    executable: Path,
    repetitions: int,
    batches: list[int],
    iterations: int,
    warmup: int,
    expected_precision: str,
    samples: dict[tuple[str, str, str, int], Samples],
) -> tuple[bool, str, CudaDeviceInfo | None]:
    pending: dict[tuple[str, str, str, int], Samples] = defaultdict(Samples)
    device_info: CudaDeviceInfo | None = None
    expected_operations: set[str] | None = None

    with tempfile.TemporaryDirectory(prefix="geo-cuda-report-") as temporary:
        directory = Path(temporary)
        for repetition in range(repetitions):
            for batch in batches:
                csv_path = directory / f"cuda-{repetition}-{batch}.csv"
                completed = run_command([
                    str(executable),
                    "--batch", str(batch),
                    "--iterations", str(iterations),
                    "--warmup", str(warmup),
                    "--operation", "all",
                    "--csv", str(csv_path),
                ], allow_skip=True)
                if completed.returncode == 77:
                    reason = (
                        completed.stderr.strip()
                        or completed.stdout.strip()
                        or "CUDA device unavailable"
                    )
                    return False, reason, None

                observed_device = parse_cuda_device(completed.stdout)
                if device_info is None:
                    device_info = observed_device
                elif observed_device != device_info:
                    raise RuntimeError(
                        "CUDA device/runtime metadata changed during report collection: "
                        f"expected={device_info} actual={observed_device}"
                    )

                if not csv_path.is_file():
                    raise RuntimeError(f"CUDA harness did not create {csv_path}")
                operations: set[str] = set()
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows:
                    raise RuntimeError(
                        f"CUDA harness produced no CSV rows for batch {batch}"
                    )
                for row in rows:
                    operation = row["operation"]
                    if operation in operations:
                        raise RuntimeError(
                            f"duplicate CUDA operation row {operation} for batch {batch}"
                        )
                    operations.add(operation)
                    precision = row["precision"]
                    if precision != expected_precision:
                        raise RuntimeError(
                            "CUDA precision does not match the configured report: "
                            f"expected={expected_precision} actual={precision}"
                        )
                    if int(row["batch"]) != batch:
                        raise RuntimeError(
                            "CUDA CSV batch does not match the requested batch: "
                            f"requested={batch} actual={row['batch']}"
                        )
                    max_absolute = float(row["max_absolute_error"])
                    max_relative = float(row["max_relative_error"])
                    mismatches = int(row["mismatches"])
                    kernel_ns = (
                        float(row["kernel_ms_per_batch"]) * 1.0e6 / batch
                    )
                    total_ns = (
                        float(row["total_ms_per_batch"]) * 1.0e6 / batch
                    )
                    pending[(operation, "cuda_kernel", precision, batch)].add(
                        kernel_ns,
                        max_absolute,
                        max_relative,
                        mismatches,
                    )
                    pending[(operation, "cuda_end_to_end", precision, batch)].add(
                        total_ns,
                        max_absolute,
                        max_relative,
                        mismatches,
                    )
                if expected_operations is None:
                    expected_operations = operations
                elif operations != expected_operations:
                    raise RuntimeError(
                        "CUDA operation matrix changed between runs: "
                        f"expected={sorted(expected_operations)} "
                        f"actual={sorted(operations)}"
                    )

    if device_info is None:
        raise RuntimeError("CUDA report collection completed without device metadata")
    merge_samples(samples, pending)
    return True, "", device_info


def summarize(
    samples: dict[tuple[str, str, str, int], Samples],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(samples):
        operation, backend, precision, batch = key
        values = samples[key]
        timings = values.timings_ns
        rows.append({
            "operation": operation,
            "backend": backend,
            "precision": precision,
            "batch": batch,
            "samples": len(timings),
            "min_ns": min(timings),
            "p50_ns": percentile(timings, 0.50),
            "p95_ns": percentile(timings, 0.95),
            "p99_ns": percentile(timings, 0.99),
            "mean_ns": statistics.fmean(timings),
            "max_ns": max(timings),
            "stdev_ns": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "max_absolute_error": values.max_absolute,
            "max_relative_error": values.max_relative,
            "mismatches": values.mismatches,
        })
    return rows


def markdown_report(
    metadata: dict[str, object],
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Geometric Elementary Operators workload report",
        "",
        f"Generated: `{metadata['generated_utc']}`",
        "",
        "## Configuration",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Processor: `{metadata['processor']}`",
        f"- CPU precision: `{metadata['cpu_precision']}`",
        f"- CPU repetitions: `{metadata['cpu_repetitions']}`",
        f"- CPU iterations per repetition: `{metadata['cpu_iterations']}`",
        f"- CPU warmup iterations: `{metadata['cpu_warmup']}`",
        f"- Fixture seed: `{metadata['seed']}`",
        f"- CUDA included: `{metadata['cuda_included']}`",
    ]
    cuda_device = metadata.get("cuda_device")
    if isinstance(cuda_device, dict):
        lines.extend([
            f"- CUDA device: `{cuda_device['name']}`",
            "- CUDA compute capability: "
            f"`{cuda_device['compute_major']}.{cuda_device['compute_minor']}`",
            f"- CUDA multiprocessors: `{cuda_device['multiprocessor_count']}`",
            f"- CUDA runtime version: `{cuda_device['runtime_version']}`",
            f"- CUDA driver version: `{cuda_device['driver_version']}`",
        ])
    if metadata.get("cuda_reason"):
        lines.append(f"- CUDA status: `{metadata['cuda_reason']}`")
    lines.extend([
        "",
        "## Results",
        "",
        "| Operation | Backend | Precision | Batch | Samples | Min ns | P50 ns | P95 ns | P99 ns | Max abs error | Max rel error | Mismatches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['operation']} | {row['backend']} | {row['precision']} | "
            f"{row['batch']} | {row['samples']} | {row['min_ns']:.3f} | "
            f"{row['p50_ns']:.3f} | {row['p95_ns']:.3f} | "
            f"{row['p99_ns']:.3f} | "
            f"{row['max_absolute_error']:.3e} | "
            f"{row['max_relative_error']:.3e} | {row['mismatches']} |"
        )
    lines.extend([
        "",
        "Every timing row is coupled to a correctness comparison on the same "
        "deterministic fixtures. CUDA kernel-only and transfer-inclusive rows "
        "are reported separately. Shared-runner results are regression evidence, "
        "not definitive hardware claims.",
        "",
    ])
    return "\n".join(lines)


def parse_batches(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            raise argparse.ArgumentTypeError("CUDA batch entries must not be empty")
        try:
            value = int(stripped)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid CUDA batch: {stripped}"
            ) from error
        if value <= 0:
            raise argparse.ArgumentTypeError("CUDA batches must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one CUDA batch is required")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-executable", type=Path, required=True)
    parser.add_argument("--cuda-executable", type=Path)
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    parser.add_argument("--cpu-repetitions", type=int, default=11)
    parser.add_argument("--cpu-iterations", type=int, default=100000)
    parser.add_argument("--cpu-warmup", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        default=0x243F6A88,
    )
    parser.add_argument("--cuda-repetitions", type=int, default=5)
    parser.add_argument(
        "--cuda-batches",
        type=parse_batches,
        default=parse_batches("1,32,256,1024,16384,262144,1048576"),
    )
    parser.add_argument("--cuda-iterations", type=int, default=100)
    parser.add_argument("--cuda-warmup", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cpu_repetitions < 3 or args.cuda_repetitions < 1:
        parser.error(
            "CPU repetitions must be at least 3 and CUDA repetitions at least 1"
        )
    if args.cpu_iterations <= 0 or args.cpu_warmup < 0:
        parser.error("invalid CPU iteration configuration")
    if args.cuda_iterations <= 0 or args.cuda_warmup < 0:
        parser.error("invalid CUDA iteration configuration")
    if args.seed < 0 or args.seed > 0xFFFFFFFF:
        parser.error("seed must be in the unsigned 32-bit range")
    if not args.cpu_executable.is_file():
        parser.error(f"missing CPU executable: {args.cpu_executable}")
    if args.cuda_executable is not None and not args.cuda_executable.is_file():
        parser.error(f"missing CUDA executable: {args.cuda_executable}")

    samples: dict[tuple[str, str, str, int], Samples] = defaultdict(Samples)
    collect_cpu(
        args.cpu_executable,
        args.cpu_repetitions,
        args.cpu_iterations,
        args.cpu_warmup,
        args.seed,
        args.precision,
        samples,
    )

    cuda_included = False
    cuda_reason = "not built"
    cuda_device: CudaDeviceInfo | None = None
    if args.cuda_executable is not None:
        cuda_included, cuda_reason, cuda_device = collect_cuda(
            args.cuda_executable,
            args.cuda_repetitions,
            args.cuda_batches,
            args.cuda_iterations,
            args.cuda_warmup,
            args.precision,
            samples,
        )
        if cuda_included:
            cuda_reason = ""

    rows = summarize(samples)
    mismatch_total = sum(int(row["mismatches"]) for row in rows)
    metadata: dict[str, object] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_precision": args.precision,
        "cpu_repetitions": args.cpu_repetitions,
        "cpu_iterations": args.cpu_iterations,
        "cpu_warmup": args.cpu_warmup,
        "seed": args.seed,
        "cuda_included": cuda_included,
        "cuda_reason": cuda_reason,
        "cuda_device": cuda_device.as_dict() if cuda_device is not None else None,
        "cuda_repetitions": args.cuda_repetitions,
        "cuda_batches": args.cuda_batches,
        "cuda_iterations": args.cuda_iterations,
        "cuda_warmup": args.cuda_warmup,
        "mismatch_total": mismatch_total,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": rows}
    (args.out_dir / "workload_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "workload_report.md").write_text(
        markdown_report(metadata, rows),
        encoding="utf-8",
    )
    with (args.out_dir / "workload_report.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = list(rows[0].keys()) if rows else ["operation"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if mismatch_total != 0:
        raise RuntimeError(f"workload suite recorded {mismatch_total} mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
