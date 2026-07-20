#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int, int], float]:
    result: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["mode"], int(row["dimension"]), int(row["batch"]))] = float(row["samples_per_second"])
    return result


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--pytorch", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    optimized = load(args.optimized)
    pytorch = load(args.pytorch)
    rows: list[dict[str, object]] = []
    for key in sorted(set(baseline) & set(optimized) & set(pytorch)):
        mode, dimension, batch = key
        base = baseline[key]
        opt = optimized[key]
        torch = pytorch[key]
        rows.append({
            "mode": mode,
            "dimension": dimension,
            "batch": batch,
            "baseline_geo_samples_per_second": base,
            "optimized_geo_samples_per_second": opt,
            "pytorch_samples_per_second": torch,
            "optimized_over_baseline": opt / base,
            "optimized_over_pytorch": opt / torch,
        })

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    inf_opt = [float(row["optimized_over_pytorch"]) for row in rows if row["mode"] == "inference"]
    train_opt = [float(row["optimized_over_pytorch"]) for row in rows if row["mode"] == "train_step"]
    inf_gain = [float(row["optimized_over_baseline"]) for row in rows if row["mode"] == "inference"]
    train_gain = [float(row["optimized_over_baseline"]) for row in rows if row["mode"] == "train_step"]
    n6_train = [row for row in rows if row["mode"] == "train_step" and row["dimension"] == 6]

    lines = [
        "# GEO V8 planned batch optimization benchmark",
        "",
        "Matched float64 CPU workloads, one thread. The optimized path precomputes blade routing/signs and executes batch-major native forward/VJP/SGD kernels.",
        "",
        f"- Optimized GEO / baseline GEO inference geometric mean: **{geomean(inf_gain):.2f}x**",
        f"- Optimized GEO / baseline GEO training geometric mean: **{geomean(train_gain):.2f}x**",
        f"- Optimized GEO / PyTorch inference geometric mean: **{geomean(inf_opt):.2f}x**",
        f"- Optimized GEO / PyTorch training geometric mean: **{geomean(train_opt):.2f}x**",
        "",
        "| Mode | n | Batch | Baseline GEO/s | Optimized GEO/s | PyTorch/s | Opt/Base | Opt/PyTorch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['dimension']} | {row['batch']} | "
            f"{float(row['baseline_geo_samples_per_second']):.1f} | "
            f"{float(row['optimized_geo_samples_per_second']):.1f} | "
            f"{float(row['pytorch_samples_per_second']):.1f} | "
            f"{float(row['optimized_over_baseline']):.2f}x | "
            f"{float(row['optimized_over_pytorch']):.2f}x |"
        )
    lines += ["", "## Dimension-6 training crossover"]
    for row in n6_train:
        lines.append(
            f"- batch {row['batch']}: optimized GEO / PyTorch = "
            f"**{float(row['optimized_over_pytorch']):.2f}x**"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "This tests one specialized geometric-product/MSE/SGD path. It does not establish universal speedup for arbitrary V8 graphs, GPUs, compiled PyTorch, or vendor-tuned kernels.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "GEO_V8_BATCH_OPT_BENCHMARK: PASS "
        f"inference_opt_over_base={geomean(inf_gain):.3f} "
        f"train_opt_over_base={geomean(train_gain):.3f} "
        f"inference_opt_over_torch={geomean(inf_opt):.3f} "
        f"train_opt_over_torch={geomean(train_opt):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
