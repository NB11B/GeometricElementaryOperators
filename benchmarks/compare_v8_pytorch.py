#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int, int], float]:
    result = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[(row["mode"], int(row["dimension"]), int(row["batch"]))] = float(row["samples_per_second"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", required=True, type=Path)
    parser.add_argument("--pytorch", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    geo = load(args.geo)
    torch = load(args.pytorch)
    rows = []
    for key in sorted(geo):
        if key not in torch:
            continue
        mode, dimension, batch = key
        speedup = geo[key] / torch[key]
        rows.append({
            "mode": mode,
            "dimension": dimension,
            "batch": batch,
            "geo_samples_per_second": geo[key],
            "pytorch_samples_per_second": torch[key],
            "geo_over_pytorch": speedup,
        })
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    inf = [r["geo_over_pytorch"] for r in rows if r["mode"] == "inference"]
    train = [r["geo_over_pytorch"] for r in rows if r["mode"] == "train_step"]
    md = [
        "# GEO V8 vs PyTorch eager CPU benchmark",
        "",
        "Matched float64 geometric-product workloads, one CPU thread, identical dimensions and batch sizes.",
        "",
        f"- Geometric mean inference speedup: **{geomean(inf):.2f}x**",
        f"- Geometric mean training speedup: **{geomean(train):.2f}x**",
        "",
        "| Mode | n | Batch | GEO samples/s | PyTorch samples/s | GEO / PyTorch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['mode']} | {r['dimension']} | {r['batch']} | {r['geo_samples_per_second']:.1f} | {r['pytorch_samples_per_second']:.1f} | {r['geo_over_pytorch']:.2f}x |")
    md += [
        "",
        "## Interpretation boundary",
        "",
        "This measures small-to-medium CPU workloads where framework overhead and GEO's compact native execution are relevant. It does not establish superiority for large batched tensor workloads, GPU execution, or highly optimized compiled PyTorch graphs.",
    ]
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"GEO_V8_BENCHMARK: PASS inference_geomean={geomean(inf):.3f} train_geomean={geomean(train):.3f}")
    return 0


def geomean(values: list[float]) -> float:
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


if __name__ == "__main__":
    raise SystemExit(main())
