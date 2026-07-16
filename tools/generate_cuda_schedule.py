#!/usr/bin/env python3
"""Generate batched CUDA kernels and launchers from checked schedule JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_OPCODES = {
    "cl20_add",
    "cl20_product",
    "reverse",
    "vector_dot",
    "vector_wedge",
}


def validate_schedule(data: dict[str, Any]) -> dict[str, Any]:
    name = data.get("name")
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise ValueError("schedule name must be a C/CUDA identifier")
    input_registers = data.get("input_registers")
    if not isinstance(input_registers, list) or not input_registers:
        raise ValueError(f"{name}: input_registers must be non-empty")
    if any(not isinstance(value, int) or value < 0 for value in input_registers):
        raise ValueError(f"{name}: invalid input register")
    if len(set(input_registers)) != len(input_registers):
        raise ValueError(f"{name}: duplicate input register")

    available = set(input_registers)
    instructions = data.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{name}: instructions must be non-empty")
    normalized: list[dict[str, int | str]] = []
    for index, raw in enumerate(instructions):
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: instruction {index} is not an object")
        opcode = raw.get("opcode")
        destination = raw.get("destination")
        if opcode not in SUPPORTED_OPCODES:
            raise ValueError(f"{name}: unsupported opcode {opcode!r}")
        if not isinstance(destination, int) or destination < 0 or destination in available:
            raise ValueError(f"{name}: destination must be a fresh register")
        if opcode == "reverse":
            source = raw.get("source", raw.get("left"))
            if not isinstance(source, int) or source not in available:
                raise ValueError(f"{name}: unavailable reverse source")
            normalized.append({
                "opcode": opcode,
                "destination": destination,
                "source": source,
            })
        else:
            left = raw.get("left")
            right = raw.get("right")
            if not isinstance(left, int) or left not in available:
                raise ValueError(f"{name}: unavailable left register")
            if not isinstance(right, int) or right not in available:
                raise ValueError(f"{name}: unavailable right register")
            normalized.append({
                "opcode": opcode,
                "destination": destination,
                "left": left,
                "right": right,
            })
        available.add(destination)

    root = data.get("root_register")
    if not isinstance(root, int) or root not in available:
        raise ValueError(f"{name}: invalid root register")
    return {
        "name": name,
        "input_registers": input_registers,
        "instructions": normalized,
        "root_register": root,
    }


def header_text(schedule: dict[str, Any]) -> str:
    name = schedule["name"]
    guard = f"GEO_CUDA_SCHEDULE_{name.upper()}_H"
    input_parameters = ",\n    ".join(
        f"const geo_cl20_t *input_r{register}"
        for register in schedule["input_registers"]
    )
    if input_parameters:
        input_parameters += ",\n    "
    return f'''#ifndef {guard}
#define {guard}

#include <stddef.h>

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {{
#endif

/* Device pointers are required for all input and output arrays. */
int geo_cuda_schedule_{name}_launch(
    {input_parameters}geo_cl20_t *output,
    size_t count,
    unsigned int block_size,
    void *stream
);

#ifdef __cplusplus
}}
#endif

#endif
'''


def instruction_lines(schedule: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for register in schedule["input_registers"]:
        lines.append(f"    const geo_cl20_t r{register} = input_r{register}[index];")
    for instruction in schedule["instructions"]:
        opcode = instruction["opcode"]
        destination = int(instruction["destination"])
        if opcode == "reverse":
            source = int(instruction["source"])
            expression = f"geo_schedule_reverse(r{source})"
        else:
            left = int(instruction["left"])
            right = int(instruction["right"])
            helper = {
                "cl20_add": "geo_schedule_add",
                "cl20_product": "geo_schedule_product",
                "vector_dot": "geo_schedule_dot",
                "vector_wedge": "geo_schedule_wedge",
            }[str(opcode)]
            expression = f"{helper}(r{left}, r{right})"
        lines.append(f"    const geo_cl20_t r{destination} = {expression};")
    lines.append(f"    output[index] = r{schedule['root_register']};")
    return lines


def source_text(schedule: dict[str, Any]) -> str:
    name = schedule["name"]
    input_parameters = ",\n    ".join(
        f"const geo_cl20_t *input_r{register}"
        for register in schedule["input_registers"]
    )
    if input_parameters:
        input_parameters += ",\n    "
    kernel_arguments = ", ".join(
        f"input_r{register}" for register in schedule["input_registers"]
    )
    if kernel_arguments:
        kernel_arguments += ", "
    pointer_checks = " || ".join(
        f"input_r{register} == nullptr" for register in schedule["input_registers"]
    )
    if pointer_checks:
        pointer_checks += " || "

    return f'''#include "geo_cuda_schedule_{name}.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>

static __device__ __forceinline__ geo_cl20_t geo_schedule_make(
    geo_real_t scalar,
    geo_real_t e1,
    geo_real_t e2,
    geo_real_t e12
) {{
    const geo_cl20_t value = {{scalar, e1, e2, e12}};
    return value;
}}

static __device__ __forceinline__ geo_cl20_t geo_schedule_add(
    geo_cl20_t left,
    geo_cl20_t right
) {{
    return geo_schedule_make(
        left.scalar + right.scalar,
        left.e1 + right.e1,
        left.e2 + right.e2,
        left.e12 + right.e12
    );
}}

static __device__ __forceinline__ geo_cl20_t geo_schedule_reverse(
    geo_cl20_t value
) {{
    return geo_schedule_make(value.scalar, value.e1, value.e2, -value.e12);
}}

static __device__ __forceinline__ geo_cl20_t geo_schedule_product(
    geo_cl20_t a,
    geo_cl20_t b
) {{
    return geo_schedule_make(
        a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 - a.e12 * b.e12,
        a.scalar * b.e1 + a.e1 * b.scalar - a.e2 * b.e12 + a.e12 * b.e2,
        a.scalar * b.e2 + a.e2 * b.scalar + a.e1 * b.e12 - a.e12 * b.e1,
        a.scalar * b.e12 + a.e12 * b.scalar + a.e1 * b.e2 - a.e2 * b.e1
    );
}}

static __device__ __forceinline__ geo_cl20_t geo_schedule_dot(
    geo_cl20_t left,
    geo_cl20_t right
) {{
    return geo_schedule_make(
        left.e1 * right.e1 + left.e2 * right.e2,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0
    );
}}

static __device__ __forceinline__ geo_cl20_t geo_schedule_wedge(
    geo_cl20_t left,
    geo_cl20_t right
) {{
    return geo_schedule_make(
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0,
        left.e1 * right.e2 - left.e2 * right.e1
    );
}}

static __global__ void geo_cuda_schedule_{name}_kernel(
    {input_parameters}geo_cl20_t *output,
    size_t count
) {{
    const size_t index = (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;
    if (index >= count) return;
{chr(10).join(instruction_lines(schedule))}
}}

extern "C" int geo_cuda_schedule_{name}_launch(
    {input_parameters}geo_cl20_t *output,
    size_t count,
    unsigned int block_size,
    void *stream
) {{
    size_t block_count;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (count == 0u) return (int)cudaSuccess;
    if ({pointer_checks}output == nullptr || block_size == 0u || block_size > 1024u) {{
        return (int)cudaErrorInvalidValue;
    }}
    if (count > SIZE_MAX - (size_t)(block_size - 1u)) {{
        return (int)cudaErrorInvalidValue;
    }}
    block_count = (count + (size_t)block_size - 1u) / (size_t)block_size;
    if (block_count > (size_t)UINT_MAX) return (int)cudaErrorInvalidConfiguration;

    geo_cuda_schedule_{name}_kernel<<<
        (unsigned int)block_count,
        block_size,
        0u,
        cuda_stream
    >>>({kernel_arguments}output, count);
    return (int)cudaGetLastError();
}}
'''


def self_test() -> None:
    schedule = validate_schedule({
        "name": "self_test",
        "input_registers": [0, 1],
        "instructions": [
            {"opcode": "cl20_product", "destination": 2, "left": 0, "right": 1},
            {"opcode": "reverse", "destination": 3, "source": 2},
        ],
        "root_register": 3,
    })
    source = source_text(schedule)
    assert "geo_cuda_schedule_self_test_kernel" in source
    assert "geo_schedule_product" in source
    assert "geo_schedule_reverse" in source
    assert "SIZE_MAX" in source


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for schedule_path in args.schedule_json:
        schedule = validate_schedule(
            json.loads(schedule_path.read_text(encoding="utf-8"))
        )
        name = schedule["name"]
        header = args.out_dir / f"geo_cuda_schedule_{name}.h"
        source = args.out_dir / f"geo_cuda_schedule_{name}.cu"
        header.write_text(header_text(schedule), encoding="utf-8")
        source.write_text(source_text(schedule), encoding="utf-8")
        manifest.append({
            "name": name,
            "source_schedule": str(schedule_path),
            "header": header.name,
            "source": source.name,
            "input_registers": schedule["input_registers"],
            "root_register": schedule["root_register"],
            "instruction_count": len(schedule["instructions"]),
        })

    (args.out_dir / "cuda_schedule_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
