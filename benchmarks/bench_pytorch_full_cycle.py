#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch


def gp_tables(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = 1 << n
    a_idx = []
    b_idx = []
    out_idx = []
    signs = []
    for a in range(count):
        for b in range(count):
            sign = 1
            for i in range(n):
                if (a >> i) & 1 and (b & ((1 << i) - 1)).bit_count() & 1:
                    sign = -sign
            a_idx.append(a)
            b_idx.append(b)
            out_idx.append(a ^ b)
            signs.append(sign)
    return (
        torch.tensor(a_idx, dtype=torch.long),
        torch.tensor(b_idx, dtype=torch.long),
        torch.tensor(out_idx, dtype=torch.long),
        torch.tensor(signs, dtype=torch.float64),
    )


def make_input(n: int, seed: int) -> torch.Tensor:
    count = 1 << n
    vals = []
    for i in range(count):
        x = (seed + 0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1)
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
        x ^= x >> 31
        vals.append(((x % 2001) - 1000) / 1000.0)
    return torch.tensor(vals, dtype=torch.float64)


def gp(x: torch.Tensor, w: torch.Tensor, table: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a_idx, b_idx, out_idx, signs = table
    products = x[..., a_idx] * w[b_idx] * signs
    out_shape = (*x.shape[:-1], x.shape[-1])
    out = torch.zeros(out_shape, dtype=x.dtype)
    expanded = out_idx.expand(*x.shape[:-1], out_idx.numel())
    return out.scatter_add(-1, expanded, products)


def bench_case(n: int, batch: int, iterations: int) -> list[dict[str, object]]:
    table = gp_tables(n)
    w_truth = make_input(n, 3)
    rows = []

    for i in range(20):
        x = torch.stack([make_input(n, 100 + i * batch + j) for j in range(batch)])
        gp(x, w_truth, table)

    start = time.perf_counter()
    checksum = 0.0
    with torch.no_grad():
        for i in range(iterations):
            x = torch.stack([make_input(n, 101 + i * batch + j) for j in range(batch)])
            y = gp(x, w_truth, table)
            checksum += float((0.5 * y.square().sum()).item())
    elapsed = time.perf_counter() - start
    rows.append({
        "backend": "pytorch_eager",
        "mode": "inference",
        "dimension": n,
        "batch": batch,
        "samples_per_second": iterations * batch / elapsed,
        "checksum": checksum,
    })

    w = torch.nn.Parameter(w_truth.clone())
    opt = torch.optim.SGD([w], lr=1e-6)
    start = time.perf_counter()
    checksum = 0.0
    for i in range(iterations):
        x = torch.stack([make_input(n, 1001 + i * batch + j) for j in range(batch)])
        with torch.no_grad():
            target = gp(x, w_truth, table)
        opt.zero_grad(set_to_none=True)
        pred = gp(x, w, table)
        loss = 0.5 * (pred - target).square().sum() / batch
        loss.backward()
        opt.step()
        checksum += float(loss.item())
    elapsed = time.perf_counter() - start
    rows.append({
        "backend": "pytorch_eager",
        "mode": "train_step",
        "dimension": n,
        "batch": batch,
        "samples_per_second": iterations * batch / elapsed,
        "checksum": checksum,
    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    all_rows: list[dict[str, object]] = []
    for n in range(2, 7):
        for batch in (1, 16, 64):
            iterations = 50 if n >= 6 else 200
            all_rows.extend(bench_case(n, batch, iterations))
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "mode", "dimension", "batch", "samples_per_second", "checksum"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"PYTORCH_VERSION={torch.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
