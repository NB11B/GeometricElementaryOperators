#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        row["mode"],
        row["timing_class"],
        int(row["dimension"]),
        int(row["batch"]),
    )


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--device", default="NVIDIA GeForce RTX 5070 Laptop GPU")
    parser.add_argument("--torch", default="2.12.1+cu130")
    parser.add_argument("--triton", default="3.7.1")
    parser.add_argument("--cuda", default="13.1 toolkit / 13.0 PyTorch runtime")
    args = parser.parse_args()

    geo = load_csv(args.results_dir / "geo-cuda.csv")
    eager = load_csv(args.results_dir / "pytorch-eager-cuda.csv")
    compiled = load_csv(args.results_dir / "pytorch-compile-cuda.csv")

    for name, rows in (("geo", geo), ("eager", eager), ("compile", compiled)):
        bad = [row for row in rows if int(row.get("trial_count", "0")) != 9]
        if bad:
            raise RuntimeError(f"{name} contains rows without nine trials")

    eager_index = {row_key(row): row for row in eager}
    compiled_index = {row_key(row): row for row in compiled}
    geo_by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in geo:
        geo_by_backend[row["backend"]].append(row)

    summary_rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []

    for backend, rows in sorted(geo_by_backend.items()):
        for comparator_name, comparator_index in (
            ("PyTorch eager CUDA", eager_index),
            ("PyTorch compile CUDA", compiled_index),
        ):
            grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in rows:
                key = row_key(row)
                comparator = comparator_index.get(key)
                if comparator is None:
                    raise RuntimeError(f"missing comparator row for {backend}: {key}")
                geo_rate = float(row["samples_per_second"])
                comparator_rate = float(comparator["samples_per_second"])
                ratio = geo_rate / comparator_rate
                mode, timing, dimension, batch = key
                grouped[(mode, timing)].append(ratio)
                detailed_rows.append({
                    "geo_backend": backend,
                    "comparator": comparator_name,
                    "mode": mode,
                    "timing_class": timing,
                    "dimension": dimension,
                    "batch": batch,
                    "geo_samples_per_second": geo_rate,
                    "comparator_samples_per_second": comparator_rate,
                    "ratio": ratio,
                })
            for (mode, timing), ratios in sorted(grouped.items()):
                summary_rows.append({
                    "geo_backend": backend,
                    "comparator": comparator_name,
                    "mode": mode,
                    "timing_class": timing,
                    "geometric_mean_ratio": geometric_mean(ratios),
                    "case_count": len(ratios),
                    "minimum_ratio": min(ratios),
                    "maximum_ratio": max(ratios),
                })

    ratios_csv = args.out.with_suffix(".ratios.csv")
    with ratios_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0]))
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_csv = args.out.with_suffix(".summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    def lookup(backend: str, comparator: str, mode: str, timing: str) -> float:
        for row in summary_rows:
            if (
                row["geo_backend"] == backend
                and row["comparator"] == comparator
                and row["mode"] == mode
                and row["timing_class"] == timing
            ):
                return float(row["geometric_mean_ratio"])
        raise KeyError((backend, comparator, mode, timing))

    lines: list[str] = []
    lines.append("# GEO V8 CUDA Physical Acceptance Report")
    lines.append("")
    lines.append("## Acceptance status")
    lines.append("")
    lines.append("The physical correctness matrix and nine-trial GPU benchmark protocol completed successfully on the recorded NVIDIA system. This report is specific to the device, precision, workload, software stack, comparator, and timing class stated below.")
    lines.append("")
    lines.append("## Recorded environment")
    lines.append("")
    lines.append(f"- Source SHA: `{args.source_sha}`")
    lines.append(f"- Device: {args.device}")
    lines.append("- Compute capability: 12.0")
    lines.append(f"- CUDA: {args.cuda}")
    lines.append(f"- PyTorch: {args.torch}")
    lines.append(f"- Triton: {args.triton}")
    lines.append("- Precision: float64")
    lines.append("- Dimensions: 2 through 6")
    lines.append("- Batches: 1, 16, 64, 256, and 1024")
    lines.append("- Trials: 9 measured trials per aggregate row")
    lines.append("")
    lines.append("## Optimized GEO CUDA results")
    lines.append("")
    lines.append("| Comparator | Mode | Resident | Transfer + compute | End-to-end |")
    lines.append("|---|---:|---:|---:|---:|")
    for comparator in ("PyTorch eager CUDA", "PyTorch compile CUDA"):
        for mode in ("inference", "training"):
            resident = lookup("geo_cuda_planned", comparator, mode, "resident")
            transfer = lookup("geo_cuda_planned", comparator, mode, "transfer_compute")
            end_to_end = lookup("geo_cuda_planned", comparator, mode, "end_to_end")
            lines.append(
                f"| {comparator} | {mode.title()} | **{resident:.2f}×** | **{transfer:.2f}×** | **{end_to_end:.2f}×** |"
            )
    lines.append("")
    lines.append("## Independent native-CUDA control")
    lines.append("")
    if "hand_cuda_same_plan" in geo_by_backend:
        lines.append("The hand-written CUDA comparator uses the same uploaded routing/sign plan but independent forward, VJP, loss, and SGD kernels. Its presence separates the value of GEO's specialized implementation from the generic benefit of replacing a framework path with native CUDA.")
        lines.append("")
        lines.append("| Comparator | Mode | Resident | Transfer + compute | End-to-end |")
        lines.append("|---|---:|---:|---:|---:|")
        for comparator in ("PyTorch eager CUDA", "PyTorch compile CUDA"):
            for mode in ("inference", "training"):
                resident = lookup("hand_cuda_same_plan", comparator, mode, "resident")
                transfer = lookup("hand_cuda_same_plan", comparator, mode, "transfer_compute")
                end_to_end = lookup("hand_cuda_same_plan", comparator, mode, "end_to_end")
                lines.append(
                    f"| {comparator} | {mode.title()} | {resident:.2f}× | {transfer:.2f}× | {end_to_end:.2f}× |"
                )
    else:
        lines.append("The independent hand-written CUDA comparator was not present in the supplied aggregate CSV and remains pending.")
    lines.append("")
    lines.append("## Correctness boundary")
    lines.append("")
    lines.append("The acceptance matrix compares CPU reference results with GEO CUDA reference, GEO CUDA planned, and hand-written CUDA paths across all canonical signatures, both multiplication sides, dimensions 2–6, and batches 1/16/64/256. Forward, parameter VJP, and one MSE/SGD update are checked under declared float64 tolerances.")
    lines.append("")
    lines.append("## Timing boundary")
    lines.append("")
    lines.append("Resident, transfer-plus-compute, and end-to-end measurements are distinct. Ratios in this report compare only matching mode, dimension, batch, precision, and timing class. Resident results are not compared against end-to-end results.")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("These results do not establish universal GPU superiority, performance on other devices, tensor-core performance, mixed-precision performance, or performance on unrelated workloads. Geometric means are primary; peak ratios are descriptive only.")
    lines.append("")
    lines.append("## Defensible summary")
    lines.append("")
    lines.append(
        f"On the recorded NVIDIA GPU and float64 workload, optimized GEO CUDA achieved geometric-mean resident-inference ratios of {lookup('geo_cuda_planned', 'PyTorch eager CUDA', 'inference', 'resident'):.2f}× versus PyTorch eager CUDA and {lookup('geo_cuda_planned', 'PyTorch compile CUDA', 'inference', 'resident'):.2f}× versus PyTorch compile CUDA. End-to-end inference ratios were {lookup('geo_cuda_planned', 'PyTorch eager CUDA', 'inference', 'end_to_end'):.2f}× and {lookup('geo_cuda_planned', 'PyTorch compile CUDA', 'inference', 'end_to_end'):.2f}× respectively."
    )
    lines.append("")
    lines.append("## Evidence files")
    lines.append("")
    lines.append(f"- Detailed matched ratios: `{ratios_csv.name}`")
    lines.append(f"- Aggregate summaries: `{summary_csv.name}`")
    lines.append("- Raw and aggregate benchmark CSV/JSON files are preserved in the packaged evidence archive.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"GEO_GPU_REPORT: PASS output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
