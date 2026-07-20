#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch


def gp_sign(left: int, right: int, signature: list[int], dimension: int) -> int:
    sign = 1
    for axis in range(dimension):
        if (left >> axis) & 1:
            lower = right & ((1 << axis) - 1)
            if lower.bit_count() & 1:
                sign = -sign
            if (right >> axis) & 1:
                sign *= signature[axis]
    return sign


def coefficient_tensor(dimension: int, device: torch.device) -> torch.Tensor:
    blades = 1 << dimension
    signature = [1] * dimension
    coeff = torch.zeros((blades, blades, blades), dtype=torch.float64, device=device)
    for left in range(blades):
        for right in range(blades):
            coeff[left ^ right, left, right] = float(gp_sign(left, right, signature, dimension))
    return coeff


def deterministic(count: int, salt: int, device: torch.device | None = None) -> torch.Tensor:
    values = [(((i * 23 + salt * 11) % 41) - 20) / 31.0 for i in range(count)]
    return torch.tensor(values, dtype=torch.float64, device=device)


def forward(inputs: torch.Tensor, parameter: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,oip,p->bo", inputs, coeff, parameter)


def training_step(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    parameter: torch.Tensor,
    coeff: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parameter.grad is not None:
        parameter.grad = None
    outputs = forward(inputs, parameter, coeff)
    loss = 0.5 * torch.sum((outputs - targets) ** 2) / inputs.shape[0]
    loss.backward()
    with torch.no_grad():
        parameter -= learning_rate * parameter.grad
    return parameter, loss


def resident_seconds(fn, iterations: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / 1000.0


def wall_seconds(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_case(
    dimension: int,
    batch: int,
    mode: str,
    timing_class: str,
    backend: str,
    iterations: int,
    device: torch.device,
) -> tuple[float, float]:
    blades = 1 << dimension
    host_inputs = deterministic(batch * blades, dimension).reshape(batch, blades)
    host_targets = deterministic(batch * blades, dimension + 2).reshape(batch, blades)
    host_parameter = deterministic(blades, dimension + 5)
    coeff = coefficient_tensor(dimension, device)

    def prepare():
        inputs = host_inputs.to(device)
        targets = host_targets.to(device)
        parameter = host_parameter.to(device).clone().requires_grad_(mode == "training")
        return inputs, targets, parameter

    inputs, targets, parameter = prepare()
    eager_forward = forward
    eager_training = training_step
    if backend == "compile":
        eager_forward = torch.compile(forward, backend="inductor", mode="max-autotune")
        eager_training = torch.compile(training_step, backend="inductor", mode="max-autotune")

    def resident_fn():
        nonlocal parameter
        if mode == "inference":
            return eager_forward(inputs, parameter, coeff)
        parameter, loss = eager_training(inputs, targets, parameter, coeff, 0.0001)
        return loss

    if backend == "compile":
        resident_fn()
        torch.cuda.synchronize()

    if timing_class == "resident":
        seconds = resident_seconds(resident_fn, iterations)
        result = resident_fn()
    elif timing_class == "transfer_compute":
        def transfer_fn():
            nonlocal inputs, targets, parameter
            inputs, targets, parameter = prepare()
            return resident_fn()
        seconds = wall_seconds(transfer_fn, iterations)
        result = transfer_fn()
    else:
        def end_to_end_fn():
            local_inputs, local_targets, local_parameter = prepare()
            if mode == "inference":
                output = eager_forward(local_inputs, local_parameter, coeff)
            else:
                output, _ = eager_training(local_inputs, local_targets, local_parameter, coeff, 0.0001)
            return output.cpu()
        seconds = wall_seconds(end_to_end_fn, iterations)
        result = end_to_end_fn()

    if isinstance(result, tuple):
        result = result[0]
    checksum = float(result.detach().double().reshape(-1).dot(
        torch.arange(1, result.numel() + 1, dtype=torch.float64, device=result.device)
    ).item())
    return seconds, checksum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend", choices=("eager", "compile"), default="eager")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.iterations <= 0:
        raise SystemExit("iterations must be positive")

    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for dimension in range(2, 7):
        for batch in (1, 16, 64, 256, 1024):
            for mode in ("inference", "training"):
                for timing_class in ("resident", "transfer_compute", "end_to_end"):
                    seconds, checksum = run_case(
                        dimension, batch, mode, timing_class, args.backend, args.iterations, device
                    )
                    rows.append({
                        "backend": f"pytorch_{args.backend}_cuda",
                        "mode": mode,
                        "timing_class": timing_class,
                        "dimension": dimension,
                        "batch": batch,
                        "samples_per_second": f"{batch * args.iterations / seconds:.17g}",
                        "checksum": f"{checksum:.17g}",
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("PYTORCH_VERSION=" + torch.__version__)
    print("PYTORCH_CUDA=" + str(torch.version.cuda))
    print("PYTORCH_CUDA_BENCHMARK: PASS backend=" + args.backend + " output=" + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
