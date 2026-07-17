#!/usr/bin/env python3
"""Generate fixed-point RTL validation artifacts for all supported Q formats."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.generate_rtl import cl20_module
from tools.generate_rtl_schedule import (
    COMPONENTS,
    Multivector,
    VectorCase,
    c_harness,
    execute_schedule,
    nominal_inputs,
    schedule_module,
    schedule_testbench,
    sv_signed,
    validate_schedule,
)

SUPPORTED_FRACTION_BITS = (1, 8, 16, 24, 30)
SCHEDULE_PATHS = (
    "rtl/examples/addition_schedule.json",
    "rtl/examples/geometric_product_schedule.json",
    "rtl/examples/vector_dot_schedule.json",
    "rtl/examples/vector_wedge_schedule.json",
    "rtl/examples/rotor_action_schedule.json",
)


def representable_overflow_inputs(
    schedule: dict[str, Any], width: int, fraction_bits: int
) -> dict[int, Multivector]:
    maximum = (1 << (width - 1)) - 1
    one = 1 << fraction_bits
    inputs = {index: Multivector() for index in schedule["input_registers"]}
    ordered = schedule["input_registers"]
    first_opcode = schedule["instructions"][0]["opcode"]

    if first_opcode == "cl20_add":
        inputs[ordered[0]] = Multivector(scalar=maximum)
        inputs[ordered[1]] = Multivector(scalar=1)
    elif first_opcode == "vector_dot":
        inputs[ordered[0]] = Multivector(e1=maximum)
        inputs[ordered[1]] = Multivector(e1=maximum)
    elif first_opcode == "vector_wedge":
        inputs[ordered[0]] = Multivector(e1=maximum)
        inputs[ordered[1]] = Multivector(e2=maximum)
    else:
        inputs[ordered[0]] = Multivector(scalar=maximum)
        inputs[ordered[1]] = Multivector(scalar=maximum)
        for register in ordered[2:]:
            inputs[register] = Multivector(scalar=one)

    return inputs


def build_representable_cases(
    schedule: dict[str, Any], width: int, fraction_bits: int
) -> list[VectorCase]:
    cases: list[VectorCase] = []
    for index, inputs in enumerate(nominal_inputs(schedule, fraction_bits)):
        expected, overflow = execute_schedule(schedule, inputs, width, fraction_bits)
        if overflow:
            raise AssertionError(
                f"{schedule['name']}: nominal case {index} overflowed in Q{fraction_bits}"
            )
        cases.append(VectorCase(f"nominal_{index}", inputs, expected, False))

    inputs = representable_overflow_inputs(schedule, width, fraction_bits)
    expected, overflow = execute_schedule(schedule, inputs, width, fraction_bits)
    if not overflow:
        raise AssertionError(
            f"{schedule['name']}: representable overflow fixture did not overflow in Q{fraction_bits}"
        )
    cases.append(VectorCase("overflow", inputs, expected, True))
    return cases


def product_testbench(width: int, fraction_bits: int, module: str) -> str:
    one = 1 << fraction_bits
    maximum = (1 << (width - 1)) - 1
    minimum = -(1 << (width - 1))
    return f'''`timescale 1ns/1ps
module tb_{module};
localparam int WIDTH = {width};
localparam int FRAC = {fraction_bits};
localparam logic signed [WIDTH-1:0] ONE = {sv_signed(one, width)};
localparam logic signed [WIDTH-1:0] MAX_FIXED = {sv_signed(maximum, width)};
localparam logic signed [WIDTH-1:0] MIN_FIXED = {sv_signed(minimum, width)};
logic signed [WIDTH-1:0] a_s, a_e1, a_e2, a_e12;
logic signed [WIDTH-1:0] b_s, b_e1, b_e2, b_e12;
logic signed [WIDTH-1:0] y_s, y_e1, y_e2, y_e12;
logic overflow;

{module} #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (.*);

task clear_inputs;
begin
    a_s='0; a_e1='0; a_e2='0; a_e12='0;
    b_s='0; b_e1='0; b_e2='0; b_e12='0;
end
endtask

initial begin
    clear_inputs(); a_e1=ONE; b_e2=ONE; #1;
    if (overflow || y_e12 !== ONE || y_s !== 0 || y_e1 !== 0 || y_e2 !== 0)
        $fatal(1, "nominal e1*e2");

    clear_inputs(); a_s=MAX_FIXED; b_s=MAX_FIXED; #1;
    if (!overflow) $fatal(1, "rounded multiplication overflow");
    if (y_s !== 0 || y_e1 !== 0 || y_e2 !== 0 || y_e12 !== 0)
        $fatal(1, "multiply-overflow output invalidation");

    clear_inputs(); a_s=ONE; b_s=MAX_FIXED; a_e1=ONE; b_e1=MAX_FIXED; #1;
    if (!overflow) $fatal(1, "final wide-sum overflow");
    if (y_s !== 0 || y_e1 !== 0 || y_e2 !== 0 || y_e12 !== 0)
        $fatal(1, "sum-overflow output invalidation");

    clear_inputs();
    a_s=-ONE; a_e1=-ONE; a_e2=-ONE; a_e12=-ONE;
    b_s=-MAX_FIXED; b_e1=-MAX_FIXED; b_e2=0; b_e12=-MAX_FIXED; #1;
    if (overflow) $fatal(1, "wide cancellation must remain valid");
    if (y_s !== MAX_FIXED || y_e1 !== MAX_FIXED || y_e2 !== MAX_FIXED || y_e12 !== MAX_FIXED)
        $fatal(1, "wide cancellation result");

    clear_inputs(); a_e12=MIN_FIXED; b_s=ONE; #1;
    if (overflow || y_e12 !== MIN_FIXED) $fatal(1, "representable minimum propagation");

    $display("PASS {module} Q{fraction_bits}");
    $finish;
end
endmodule
'''


def validate_case_inputs(cases: list[VectorCase], width: int) -> None:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    for case in cases:
        for value in case.inputs.values():
            for component in value.values():
                if component < minimum or component > maximum:
                    raise AssertionError(
                        f"{case.name}: input component {component} is outside signed {width}-bit range"
                    )


def generate(out_dir: Path, width: int, fraction_bits: int) -> None:
    product_module = "geo_cl20_product_q"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{product_module}.sv").write_text(
        cl20_module(width, fraction_bits, product_module), encoding="utf-8"
    )
    (out_dir / f"tb_{product_module}.sv").write_text(
        product_testbench(width, fraction_bits, product_module), encoding="utf-8"
    )

    manifest: dict[str, Any] = {
        "width": width,
        "fraction_bits": fraction_bits,
        "product_module": product_module,
        "schedules": [],
    }

    for relative_path in SCHEDULE_PATHS:
        path = REPOSITORY_ROOT / relative_path
        schedule = validate_schedule(json.loads(path.read_text(encoding="utf-8")))
        cases = build_representable_cases(schedule, width, fraction_bits)
        validate_case_inputs(cases, width)
        module = f"geo_schedule_{schedule['name']}"
        (out_dir / f"{module}.sv").write_text(
            schedule_module(schedule, width, fraction_bits, product_module), encoding="utf-8"
        )
        (out_dir / f"tb_{module}.sv").write_text(
            schedule_testbench(schedule, cases, width, fraction_bits), encoding="utf-8"
        )
        (out_dir / f"test_{module}.c").write_text(
            c_harness(schedule, cases), encoding="utf-8"
        )
        manifest["schedules"].append(
            {
                "name": schedule["name"],
                "module": module,
                "source": relative_path,
                "cases": [
                    {
                        "name": case.name,
                        "overflow": case.overflow,
                        "expected": case.expected.values(),
                    }
                    for case in cases
                ],
            }
        )

    (out_dir / "rtl_multi_q_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def self_test() -> None:
    for fraction_bits in SUPPORTED_FRACTION_BITS:
        schedule = validate_schedule(
            json.loads(
                (REPOSITORY_ROOT / "rtl/examples/geometric_product_schedule.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cases = build_representable_cases(schedule, 32, fraction_bits)
        validate_case_inputs(cases, 32)
        if not cases[-1].overflow:
            raise AssertionError("final multi-Q case must be an overflow fixture")
        text = product_testbench(32, fraction_bits, "geo_cl20_product_q")
        if f"Q{fraction_bits}" not in text:
            raise AssertionError("product testbench did not record the Q format")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frac", type=int, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.width < 4 or args.frac not in SUPPORTED_FRACTION_BITS:
        parser.error("unsupported fixed-point width/fraction")
    if args.frac >= args.width - 1:
        parser.error("fraction bits must leave a signed integer range")
    if args.self_test:
        self_test()
    generate(args.out_dir, args.width, args.frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
