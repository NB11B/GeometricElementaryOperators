#!/usr/bin/env python3
"""Emit and benchmark signed-permutation lowering for fixed basis blades."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import geo_identity_v4_2_engine as engine
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_engine as engine


def dense_gp_fixed(values: Sequence[int], dimension: int, signature: Sequence[int], blade: int, coefficient: int, side: str) -> list[int]:
    count = 1 << dimension
    fixed = [0] * count
    fixed[blade] = coefficient
    output = [0] * count
    for left_blade in range(count):
        left_coefficient = fixed[left_blade] if side == "left" else values[left_blade]
        if left_coefficient == 0:
            continue
        for right_blade in range(count):
            right_coefficient = values[right_blade] if side == "left" else fixed[right_blade]
            if right_coefficient == 0:
                continue
            output[left_blade ^ right_blade] += engine.gp_sign(left_blade, right_blade, signature) * left_coefficient * right_coefficient
    return output


def permutation_apply(values: Sequence[int], plan: list[dict[str, int]]) -> list[int]:
    output = [0] * len(values)
    for row in plan:
        output[row["target"]] = row["factor"] * values[row["source"]]
    return output


def benchmark(values: list[int], dimension: int, signature: list[int], blade: int, coefficient: int, side: str, iterations: int) -> dict[str, Any]:
    plan = engine.fixed_blade_permutation(dimension, signature, blade, coefficient, side)
    generic = dense_gp_fixed(values, dimension, signature, blade, coefficient, side)
    specialized = permutation_apply(values, plan)
    if generic != specialized:
        raise ValueError(f"lowering mismatch for signature={signature} side={side} coefficient={coefficient}")
    start = time.perf_counter()
    for _ in range(iterations):
        dense_gp_fixed(values, dimension, signature, blade, coefficient, side)
    generic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iterations):
        permutation_apply(values, plan)
    specialized_seconds = time.perf_counter() - start
    return {
        "side": side,
        "blade": blade,
        "coefficient": coefficient,
        "iterations": iterations,
        "generic_seconds": generic_seconds,
        "specialized_seconds": specialized_seconds,
        "speedup": generic_seconds / specialized_seconds if specialized_seconds else None,
        "generic_dense_pair_slots": (1 << dimension) ** 2,
        "specialized_assignments": 1 << dimension,
        "plan": plan,
    }


def cpp_array(name: str, plan: list[dict[str, int]]) -> list[str]:
    targets = ", ".join(str(row["target"]) for row in plan)
    factors = ", ".join(str(row["factor"]) for row in plan)
    return [
        f"inline constexpr int {name}_target[{len(plan)}] = {{{targets}}};",
        f"inline constexpr int {name}_factor[{len(plan)}] = {{{factors}}};",
        f"template <class T> inline void {name}(const T *input, T *output) {{",
        f"    for (int source = 0; source < {len(plan)}; ++source) {{",
        f"        output[{name}_target[source]] = static_cast<T>({name}_factor[source]) * input[source];",
        "    }",
        "}",
        "",
    ]


def emit_header(reports: list[dict[str, Any]], dimension: int) -> str:
    lines = [
        "#ifndef GEO_FIXED_BLADE_LOWERING_V4_2_HPP",
        "#define GEO_FIXED_BLADE_LOWERING_V4_2_HPP",
        "",
        "namespace geo_fixed_blade_v4_2 {",
        "",
        f"inline constexpr int DIMENSION = {dimension};",
        f"inline constexpr int BLADE_COUNT = {1 << dimension};",
        "",
    ]
    for report in reports:
        prefix = report["signature_name"] + "_" + report["kind"] + "_" + report["side"]
        lines.extend(cpp_array(prefix, report["plan"]))
    lines.extend(["}  // namespace geo_fixed_blade_v4_2", "", "#endif", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--header-out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    try:
        if args.iterations < 1:
            raise ValueError("iterations must be positive")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        dimension = int(config["dimension"])
        blade = (1 << dimension) - 1
        values = [((index * 17 + 5) % 19) - 9 for index in range(1 << dimension)]
        reports: list[dict[str, Any]] = []
        for signature_row in config["signatures"]:
            signature_name = str(signature_row["name"])
            signature = [int(value) for value in signature_row["metric"]]
            square = engine.pseudoscalar_square(signature)
            for kind, coefficient in (("I", 1), ("Iinv", square)):
                for side in ("left", "right"):
                    row = benchmark(values, dimension, signature, blade, coefficient, side, args.iterations)
                    row.update({"signature_name": signature_name, "signature": signature, "pseudoscalar_square": square, "kind": kind})
                    reports.append(row)
        report = {
            "schema_version": 1,
            "engine": "geometric_identity_v4_2_fixed_blade_lowering",
            "dimension": dimension,
            "blade": blade,
            "report_count": len(reports),
            "reports": reports,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.header_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.header_out.write_text(emit_header(reports, dimension), encoding="utf-8", newline="\n")
        lines = [
            "# V4.2 fixed-blade signed-permutation lowering",
            "",
            f"- dimension: {dimension}",
            f"- fixed pseudoscalar blade: {blade}",
            f"- lowering rows: {len(reports)}",
            f"- benchmark iterations per row: {args.iterations}",
            "",
            "| signature | constant | side | generic slots | specialized assignments | measured speedup |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in reports:
            speedup = row["speedup"] if row["speedup"] is not None else 0.0
            lines.append(f"| `{row['signature_name']}` | {row['kind']} | {row['side']} | {row['generic_dense_pair_slots']} | {row['specialized_assignments']} | {speedup:.3f}x |")
        lines.extend([
            "",
            "The specialized implementation is a compile-time signed permutation: each source blade maps to exactly one target blade with a metric-dependent factor. The Python timing is an implementation sanity benchmark, not a hardware performance claim.",
            "",
        ])
        args.markdown_out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"V4_2_LOWERING: PASS reports={report['report_count']} blade={report['blade']}")
    print(f"header: {args.header_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
