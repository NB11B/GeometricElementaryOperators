#!/usr/bin/env python3
"""Generate deterministic CUDA device helpers from the IMU sparse schedule IR."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR_PATH = ROOT / "tools" / "generate_imu_sparse_schedule.py"


def load_base_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_imu_sparse_schedule",
        BASE_GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {BASE_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_generator()


def emit_cuda_header(schedule, source_path: str) -> str:
    guard = f"GEO_CUDA_GENERATED_{schedule.name.upper()}_CUH"
    lines: list[str] = [
        "/*",
        " * Generated file. Do not edit by hand.",
        f" * Source: {source_path}",
        f" * Schedule: {schedule.name}",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        "#if !defined(__CUDACC__)",
        '#error "This generated header requires a CUDA compiler"',
        "#endif",
        "",
        "#define GEO_IMU_CUDA_INLINE __device__ __forceinline__",
        "",
    ]

    for function in schedule.functions:
        float_parameters = [f"float {name}" for name in function.inputs]
        float_parameters.extend(
            f"float *{output.name}" for output in function.outputs
        )
        lines.append(
            f"GEO_IMU_CUDA_INLINE void geo_gpu_generated_float_{function.name}("
        )
        for index, parameter in enumerate(float_parameters):
            suffix = "," if index + 1 < len(float_parameters) else ""
            lines.append(f"    {parameter}{suffix}")
        lines.extend([")", "{"])
        for output in function.outputs:
            expression = BASE._scaled_float_expression(output)
            lines.append(f"    *{output.name} = {expression};")
        lines.extend(["}", ""])

        q32_parameters = [f"int32_t {name}" for name in function.inputs]
        q32_parameters.extend(
            f"int64_t *{output.name}" for output in function.outputs
        )
        lines.append(
            f"GEO_IMU_CUDA_INLINE void geo_gpu_generated_q32_{function.name}("
        )
        for index, parameter in enumerate(q32_parameters):
            suffix = "," if index + 1 < len(q32_parameters) else ""
            lines.append(f"    {parameter}{suffix}")
        lines.extend([")", "{"])
        for output in function.outputs:
            expression = BASE._scaled_q32_expression(output)
            lines.append(f"    *{output.name} = {expression};")
        lines.extend(["}", ""])

    lines.extend(
        [
            "#undef GEO_IMU_CUDA_INLINE",
            "",
            f"#endif /* {guard} */",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CUDA float and Q32 helpers for the IMU benchmark"
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in CUDA header differs from generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        schedule = BASE.load_schedule(args.schedule)
        try:
            source_path = args.schedule.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            source_path = args.schedule.as_posix()
        generated = emit_cuda_header(schedule, source_path)
    except (BASE.ScheduleError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: unable to read {args.output}: {exc}", file=sys.stderr)
            return 2
        if existing != generated:
            print(
                f"ERROR: {args.output} is stale; regenerate it from {args.schedule}",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: {args.output} matches {args.schedule}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
