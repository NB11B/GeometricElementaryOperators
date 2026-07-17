#!/usr/bin/env python3
"""Generate executable fixed-point C and RTL datapaths from schedule JSON."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_OPCODES = {
    "cl20_add",
    "cl20_product",
    "reverse",
    "vector_dot",
    "vector_wedge",
}
COMPONENTS = ("s", "e1", "e2", "e12")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_identifier(value: object) -> bool:
    return isinstance(value, str) and IDENTIFIER_RE.fullmatch(value) is not None


@dataclass(frozen=True)
class Multivector:
    scalar: int = 0
    e1: int = 0
    e2: int = 0
    e12: int = 0

    def values(self) -> tuple[int, int, int, int]:
        return (self.scalar, self.e1, self.e2, self.e12)


@dataclass(frozen=True)
class VectorCase:
    name: str
    inputs: dict[int, Multivector]
    expected: Multivector
    overflow: bool


def checked(value: int, width: int) -> int:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    if value < minimum or value > maximum:
        raise OverflowError(value)
    return value


def round_divide(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    absolute_numerator = abs(numerator)
    absolute_denominator = abs(denominator)
    quotient, remainder = divmod(absolute_numerator, absolute_denominator)
    if remainder >= (absolute_denominator + 1) // 2:
        quotient += 1
    return sign * quotient


def fixed_multiply(a: int, b: int, width: int, fraction_bits: int) -> int:
    return checked(round_divide(a * b, 1 << fraction_bits), width)


def fixed_add(a: Multivector, b: Multivector, width: int) -> Multivector:
    return Multivector(*(checked(x + y, width) for x, y in zip(a.values(), b.values())))


def fixed_reverse(value: Multivector, width: int) -> Multivector:
    return Multivector(value.scalar, value.e1, value.e2, checked(-value.e12, width))


def fixed_product(a: Multivector, b: Multivector, width: int, fraction_bits: int) -> Multivector:
    av = a.values()
    bv = b.values()
    products = {
        (i, j): fixed_multiply(av[i], bv[j], width, fraction_bits)
        for i in range(4)
        for j in range(4)
    }
    return Multivector(
        checked(products[0, 0] + products[1, 1] + products[2, 2] - products[3, 3], width),
        checked(products[0, 1] + products[1, 0] - products[2, 3] + products[3, 2], width),
        checked(products[0, 2] + products[2, 0] + products[1, 3] - products[3, 1], width),
        checked(products[0, 3] + products[3, 0] + products[1, 2] - products[2, 1], width),
    )


def fixed_dot(a: Multivector, b: Multivector, width: int, fraction_bits: int) -> Multivector:
    x = fixed_multiply(a.e1, b.e1, width, fraction_bits)
    y = fixed_multiply(a.e2, b.e2, width, fraction_bits)
    return Multivector(scalar=checked(x + y, width))


def fixed_wedge(a: Multivector, b: Multivector, width: int, fraction_bits: int) -> Multivector:
    xy = fixed_multiply(a.e1, b.e2, width, fraction_bits)
    yx = fixed_multiply(a.e2, b.e1, width, fraction_bits)
    return Multivector(e12=checked(xy - yx, width))


def validate_schedule(data: dict[str, Any]) -> dict[str, Any]:
    name = data.get("name")
    if not is_identifier(name):
        raise ValueError("schedule name must be a SystemVerilog/C identifier")
    input_registers = data.get("input_registers")
    if not isinstance(input_registers, list) or not input_registers:
        raise ValueError(f"{name}: input_registers must be a non-empty array")
    if any(not isinstance(value, int) or value < 0 for value in input_registers):
        raise ValueError(f"{name}: invalid input register")
    if len(set(input_registers)) != len(input_registers):
        raise ValueError(f"{name}: duplicate input register")

    instructions = data.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{name}: instructions must be a non-empty array")
    available = set(input_registers)
    normalized: list[dict[str, int | str]] = []
    for index, raw in enumerate(instructions):
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: instruction {index} is not an object")
        opcode = raw.get("opcode")
        destination = raw.get("destination")
        if opcode not in SUPPORTED_OPCODES:
            raise ValueError(f"{name}: unsupported opcode {opcode!r}")
        if not isinstance(destination, int) or destination < 0 or destination in available:
            raise ValueError(f"{name}: destination {destination!r} is not fresh")
        if opcode == "reverse":
            source = raw.get("source", raw.get("left"))
            if not isinstance(source, int) or source not in available:
                raise ValueError(f"{name}: instruction {index} reads unavailable source")
            normalized.append({"opcode": opcode, "destination": destination, "source": source})
        else:
            left = raw.get("left")
            right = raw.get("right")
            if not isinstance(left, int) or left not in available:
                raise ValueError(f"{name}: instruction {index} reads unavailable left register")
            if not isinstance(right, int) or right not in available:
                raise ValueError(f"{name}: instruction {index} reads unavailable right register")
            normalized.append({
                "opcode": opcode,
                "destination": destination,
                "left": left,
                "right": right,
            })
        available.add(destination)

    root = data.get("root_register")
    if not isinstance(root, int) or root not in available:
        raise ValueError(f"{name}: invalid root_register")
    return {
        "name": name,
        "input_registers": input_registers,
        "instructions": normalized,
        "root_register": root,
        "register_count": max(available) + 1,
        "registers": sorted(available),
    }


def execute_schedule(
    schedule: dict[str, Any],
    inputs: dict[int, Multivector],
    width: int,
    fraction_bits: int,
) -> tuple[Multivector, bool]:
    registers = {index: inputs[index] for index in schedule["input_registers"]}
    try:
        for instruction in schedule["instructions"]:
            opcode = instruction["opcode"]
            destination = int(instruction["destination"])
            if opcode == "reverse":
                registers[destination] = fixed_reverse(
                    registers[int(instruction["source"])], width
                )
                continue
            left = registers[int(instruction["left"])]
            right = registers[int(instruction["right"])]
            if opcode == "cl20_add":
                result = fixed_add(left, right, width)
            elif opcode == "cl20_product":
                result = fixed_product(left, right, width, fraction_bits)
            elif opcode == "vector_dot":
                result = fixed_dot(left, right, width, fraction_bits)
            elif opcode == "vector_wedge":
                result = fixed_wedge(left, right, width, fraction_bits)
            else:  # pragma: no cover - validation prevents this
                raise AssertionError(opcode)
            registers[destination] = result
    except OverflowError:
        return Multivector(), True
    return registers[int(schedule["root_register"])], False


def round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def quantize(value: float, fraction_bits: int) -> int:
    return round_away_from_zero(value * (1 << fraction_bits))


def nominal_inputs(schedule: dict[str, Any], fraction_bits: int) -> list[dict[int, Multivector]]:
    q = lambda value: quantize(value, fraction_bits)
    inputs = schedule["input_registers"]
    if schedule["name"] == "rotor_action":
        rotor = Multivector(q(0.96875), 0, 0, q(-0.25))
        reverse = Multivector(rotor.scalar, rotor.e1, rotor.e2, -rotor.e12)
        identity = Multivector(1 << fraction_bits, 0, 0, 0)
        vector = Multivector(0, q(0.5), q(0.25), 0)
        return [
            {inputs[0]: rotor, inputs[1]: vector, inputs[2]: reverse},
            {inputs[0]: identity, inputs[1]: vector, inputs[2]: identity},
        ]
    return [
        {
            inputs[0]: Multivector(q(0.25), q(0.5), q(-0.25), q(0.125)),
            inputs[1]: Multivector(q(-0.125), q(0.25), q(0.375), q(-0.0625)),
        },
        {
            inputs[0]: Multivector(0, 1 << fraction_bits, 0, 0),
            inputs[1]: Multivector(0, 0, 1 << fraction_bits, 0),
        },
    ]


def overflow_inputs(schedule: dict[str, Any], width: int, fraction_bits: int) -> dict[int, Multivector]:
    maximum = (1 << (width - 1)) - 1
    one = 1 << fraction_bits
    inputs = {index: Multivector() for index in schedule["input_registers"]}
    first_opcode = schedule["instructions"][0]["opcode"]
    ordered_inputs = schedule["input_registers"]
    if first_opcode == "cl20_add":
        inputs[ordered_inputs[0]] = Multivector(scalar=maximum)
        inputs[ordered_inputs[1]] = Multivector(scalar=1)
    elif first_opcode == "vector_dot":
        inputs[ordered_inputs[0]] = Multivector(e1=maximum)
        inputs[ordered_inputs[1]] = Multivector(e1=2 * one)
    elif first_opcode == "vector_wedge":
        inputs[ordered_inputs[0]] = Multivector(e1=maximum)
        inputs[ordered_inputs[1]] = Multivector(e2=2 * one)
    else:
        inputs[ordered_inputs[0]] = Multivector(scalar=maximum)
        inputs[ordered_inputs[1]] = Multivector(scalar=2 * one)
        for register in ordered_inputs[2:]:
            inputs[register] = Multivector(scalar=one)
    return inputs


def build_cases(schedule: dict[str, Any], width: int, fraction_bits: int) -> list[VectorCase]:
    cases: list[VectorCase] = []
    for index, inputs in enumerate(nominal_inputs(schedule, fraction_bits)):
        expected, overflow = execute_schedule(schedule, inputs, width, fraction_bits)
        cases.append(VectorCase(f"nominal_{index}", inputs, expected, overflow))
    overflow = overflow_inputs(schedule, width, fraction_bits)
    expected, overflowed = execute_schedule(schedule, overflow, width, fraction_bits)
    if not overflowed:
        raise AssertionError(f"{schedule['name']}: overflow fixture did not overflow")
    cases.append(VectorCase("overflow", overflow, expected, True))
    return cases


def sv_replication(count_expression: str, bit: str) -> str:
    return "{" + count_expression + "{" + bit + "}}}"


def sv_signed(value: int, width: int) -> str:
    if value < 0:
        return f"-{width}'sd{abs(value)}"
    return f"{width}'sd{value}"


def register_declarations(registers: list[int]) -> str:
    lines: list[str] = []
    for register in registers:
        for component in COMPONENTS:
            lines.append(f"logic signed [WIDTH-1:0] r{register}_{component};")
        lines.append(f"logic r{register}_overflow;")
    return "\n".join(lines)


def schedule_module(schedule: dict[str, Any], width: int, fraction_bits: int, product_module: str) -> str:
    module = f"geo_schedule_{schedule['name']}"
    ports: list[str] = []
    for register in schedule["input_registers"]:
        for component in COMPONENTS:
            ports.append(f"    input logic signed [WIDTH-1:0] in_r{register}_{component}")
    ports.extend([
        "    output logic signed [WIDTH-1:0] y_s",
        "    output logic signed [WIDTH-1:0] y_e1",
        "    output logic signed [WIDTH-1:0] y_e2",
        "    output logic signed [WIDTH-1:0] y_e12",
        "    output logic overflow",
    ])

    declarations: list[str] = []
    assignments: list[str] = []
    instances: list[str] = []

    for register in schedule["input_registers"]:
        for component in COMPONENTS:
            assignments.append(f"assign r{register}_{component} = in_r{register}_{component};")
        assignments.append(f"assign r{register}_overflow = 1'b0;")

    for index, instruction in enumerate(schedule["instructions"]):
        opcode = str(instruction["opcode"])
        destination = int(instruction["destination"])
        prefix = f"op{index}"
        if opcode == "reverse":
            source = int(instruction["source"])
            declarations.append(f"logic {prefix}_local_overflow;")
            assignments.extend([
                f"assign {prefix}_local_overflow = r{source}_e12 == MIN_FIXED;",
                f"assign r{destination}_overflow = r{source}_overflow || {prefix}_local_overflow;",
                f"assign r{destination}_s = r{destination}_overflow ? '0 : r{source}_s;",
                f"assign r{destination}_e1 = r{destination}_overflow ? '0 : r{source}_e1;",
                f"assign r{destination}_e2 = r{destination}_overflow ? '0 : r{source}_e2;",
                f"assign r{destination}_e12 = r{destination}_overflow ? '0 : -r{source}_e12;",
            ])
            continue

        left = int(instruction["left"])
        right = int(instruction["right"])
        if opcode == "cl20_add":
            for component in COMPONENTS:
                declarations.append(f"logic signed [WIDTH:0] {prefix}_{component}_wide;")
                assignments.append(
                    f"assign {prefix}_{component}_wide = $signed(r{left}_{component}) + $signed(r{right}_{component});"
                )
            declarations.append(f"logic {prefix}_local_overflow;")
            overflow_terms = " || ".join(
                f"({prefix}_{component}_wide[WIDTH] != {prefix}_{component}_wide[WIDTH-1])"
                for component in COMPONENTS
            )
            assignments.append(f"assign {prefix}_local_overflow = {overflow_terms};")
            assignments.append(
                f"assign r{destination}_overflow = r{left}_overflow || r{right}_overflow || {prefix}_local_overflow;"
            )
            for component in COMPONENTS:
                assignments.append(
                    f"assign r{destination}_{component} = r{destination}_overflow ? '0 : {prefix}_{component}_wide[WIDTH-1:0];"
                )
        elif opcode == "cl20_product":
            for component in COMPONENTS:
                declarations.append(f"logic signed [WIDTH-1:0] {prefix}_{component};")
            declarations.append(f"logic {prefix}_overflow;")
            instances.append(f'''{product_module} #(.WIDTH(WIDTH), .FRAC(FRAC)) {prefix}_product (
    .a_s(r{left}_s), .a_e1(r{left}_e1), .a_e2(r{left}_e2), .a_e12(r{left}_e12),
    .b_s(r{right}_s), .b_e1(r{right}_e1), .b_e2(r{right}_e2), .b_e12(r{right}_e12),
    .y_s({prefix}_s), .y_e1({prefix}_e1), .y_e2({prefix}_e2), .y_e12({prefix}_e12),
    .overflow({prefix}_overflow)
);''')
            assignments.append(
                f"assign r{destination}_overflow = r{left}_overflow || r{right}_overflow || {prefix}_overflow;"
            )
            for component in COMPONENTS:
                assignments.append(
                    f"assign r{destination}_{component} = r{destination}_overflow ? '0 : {prefix}_{component};"
                )
        elif opcode in {"vector_dot", "vector_wedge"}:
            if opcode == "vector_dot":
                pairs = (("p0", "e1", "e1"), ("p1", "e2", "e2"))
                combine = "+"
                output_component = "s"
            else:
                pairs = (("p0", "e1", "e2"), ("p1", "e2", "e1"))
                combine = "-"
                output_component = "e12"
            for product_name, left_component, right_component in pairs:
                declarations.extend([
                    f"logic signed [2*WIDTH-1:0] {prefix}_{product_name}_raw;",
                    f"logic signed [2*WIDTH:0] {prefix}_{product_name};",
                ])
                assignments.extend([
                    f"assign {prefix}_{product_name}_raw = $signed(r{left}_{left_component}) * $signed(r{right}_{right_component});",
                    f"assign {prefix}_{product_name} = round_product({prefix}_{product_name}_raw);",
                ])
            declarations.extend([
                f"logic signed [2*WIDTH+2:0] {prefix}_value;",
                f"logic {prefix}_local_overflow;",
            ])
            assignments.extend([
                f"assign {prefix}_value = {prefix}_p0 {combine} {prefix}_p1;",
                f"assign {prefix}_local_overflow = rounded_overflow({prefix}_p0) || "
                f"rounded_overflow({prefix}_p1) || accumulated_overflow({prefix}_value);",
                f"assign r{destination}_overflow = r{left}_overflow || r{right}_overflow || {prefix}_local_overflow;",
            ])
            for component in COMPONENTS:
                value = f"{prefix}_value[WIDTH-1:0]" if component == output_component else "'0"
                assignments.append(
                    f"assign r{destination}_{component} = r{destination}_overflow ? '0 : {value};"
                )
        else:  # pragma: no cover
            raise AssertionError(opcode)

    root = int(schedule["root_register"])
    assignments.extend([
        f"assign overflow = r{root}_overflow;",
        f"assign y_s = overflow ? '0 : r{root}_s;",
        f"assign y_e1 = overflow ? '0 : r{root}_e1;",
        f"assign y_e2 = overflow ? '0 : r{root}_e2;",
        f"assign y_e12 = overflow ? '0 : r{root}_e12;",
    ])

    return f'''module {module} #(
    parameter int WIDTH = {width},
    parameter int FRAC = {fraction_bits}
) (
{",\n".join(ports)}
);

localparam logic signed [WIDTH-1:0] MIN_FIXED = {{1'b1, {{(WIDTH-1){{1'b0}}}}}};

function automatic logic signed [2*WIDTH:0] round_product(
    input logic signed [2*WIDTH-1:0] product
);
    logic negative;
    logic [2*WIDTH-1:0] magnitude;
    logic [2*WIDTH-1:0] quotient;
    logic [2*WIDTH-1:0] remainder;
    logic [2*WIDTH-1:0] mask;
    logic [2*WIDTH-1:0] half;
    begin
        negative = product < 0;
        magnitude = negative ? $unsigned(-product) : $unsigned(product);
        mask = ({{(2*WIDTH){{1'b1}}}} >> (2*WIDTH-FRAC));
        half = {{(2*WIDTH){{1'b0}}}};
        half[FRAC-1] = 1'b1;
        quotient = magnitude >> FRAC;
        remainder = magnitude & mask;
        if (remainder >= half) quotient = quotient + 1'b1;
        round_product = negative ? -$signed({{1'b0, quotient}}) : $signed({{1'b0, quotient}});
    end
endfunction

function automatic logic rounded_overflow(
    input logic signed [2*WIDTH:0] value
);
    begin
        rounded_overflow = value[2*WIDTH:WIDTH] != {{(WIDTH+1){{value[WIDTH-1]}}}};
    end
endfunction

function automatic logic accumulated_overflow(
    input logic signed [2*WIDTH+2:0] value
);
    begin
        accumulated_overflow = value[2*WIDTH+2:WIDTH] != {{(WIDTH+3){{value[WIDTH-1]}}}};
    end
endfunction

{register_declarations(schedule["registers"])}

{chr(10).join(declarations)}

{chr(10).join(instances)}

{chr(10).join(assignments)}

endmodule
'''


def schedule_testbench(schedule: dict[str, Any], cases: list[VectorCase], width: int, fraction_bits: int) -> str:
    module = f"geo_schedule_{schedule['name']}"
    declarations: list[str] = []
    connections: list[str] = []
    clear_lines: list[str] = []
    for register in schedule["input_registers"]:
        for component in COMPONENTS:
            signal = f"in_r{register}_{component}"
            declarations.append(f"logic signed [WIDTH-1:0] {signal};")
            connections.append(f"    .{signal}({signal})")
            clear_lines.append(f"        {signal} = '0;")
    connections.extend([
        "    .y_s(y_s)",
        "    .y_e1(y_e1)",
        "    .y_e2(y_e2)",
        "    .y_e12(y_e12)",
        "    .overflow(overflow)",
    ])

    case_lines: list[str] = []
    for case_index, case in enumerate(cases):
        case_lines.append("    clear_inputs();")
        for register in schedule["input_registers"]:
            value = case.inputs[register]
            for component, component_value in zip(COMPONENTS, value.values()):
                case_lines.append(
                    f"    in_r{register}_{component} = {sv_signed(component_value, width)};"
                )
        case_lines.append("    #1;")
        expected_overflow = "1'b1" if case.overflow else "1'b0"
        case_lines.append(
            f"    if (overflow !== {expected_overflow}) $fatal(1, \"case {case_index} {case.name}: overflow\");"
        )
        if case.overflow:
            case_lines.append(
                f"    if (y_s !== '0 || y_e1 !== '0 || y_e2 !== '0 || y_e12 !== '0) "
                f"$fatal(1, \"case {case_index} {case.name}: overflow outputs\");"
            )
        else:
            for component, component_value in zip(COMPONENTS, case.expected.values()):
                case_lines.append(
                    f"    if (y_{component} !== {sv_signed(component_value, width)}) "
                    f"$fatal(1, \"case {case_index} {case.name}: {component}\");"
                )

    return f'''`timescale 1ns/1ps
module tb_{module};
localparam int WIDTH = {width};
localparam int FRAC = {fraction_bits};

{chr(10).join(declarations)}
logic signed [WIDTH-1:0] y_s, y_e1, y_e2, y_e12;
logic overflow;

{module} #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (
{",\n".join(connections)}
);

task clear_inputs;
begin
{chr(10).join(clear_lines)}
end
endtask

initial begin
{chr(10).join(case_lines)}
    $display("PASS {module}");
    $finish;
end
endmodule
'''


def c_mv(value: Multivector) -> str:
    return "{" + ", ".join(str(component) for component in value.values()) + "}"


def c_instruction(instruction: dict[str, Any], index: int) -> str:
    opcode = instruction["opcode"]
    destination = int(instruction["destination"])
    if opcode == "reverse":
        source = int(instruction["source"])
        return (
            f"        if (!overflow && geo_fixed_cl20_reverse_checked(r[{source}], &r[{destination}]) "
            f"!= GEO_FIXED_OK) overflow = 1;"
        )
    left = int(instruction["left"])
    right = int(instruction["right"])
    if opcode == "cl20_add":
        return (
            f"        if (!overflow && checked_add_cl20(r[{left}], r[{right}], &r[{destination}]) "
            f"!= GEO_FIXED_OK) overflow = 1;"
        )
    if opcode == "cl20_product":
        return (
            f"        if (!overflow && geo_fixed_cl20_mul(r[{left}], r[{right}], &r[{destination}]) "
            f"!= GEO_FIXED_OK) overflow = 1;"
        )
    if opcode == "vector_dot":
        return f'''        if (!overflow) {{
            geo_fixed_t scalar_{index};
            if (geo_fixed_vector_dot(r[{left}], r[{right}], &scalar_{index}) != GEO_FIXED_OK) overflow = 1;
            else r[{destination}] = (geo_fixed_cl20_t){{scalar_{index}, 0, 0, 0}};
        }}'''
    if opcode == "vector_wedge":
        return f'''        if (!overflow) {{
            geo_fixed_t scalar_{index};
            if (geo_fixed_vector_wedge(r[{left}], r[{right}], &scalar_{index}) != GEO_FIXED_OK) overflow = 1;
            else r[{destination}] = (geo_fixed_cl20_t){{0, 0, 0, scalar_{index}}};
        }}'''
    raise AssertionError(opcode)


def c_harness(schedule: dict[str, Any], cases: list[VectorCase]) -> str:
    root = int(schedule["root_register"])
    case_blocks: list[str] = []
    for case_index, case in enumerate(cases):
        assignments = [
            f"        r[{register}] = (geo_fixed_cl20_t){c_mv(case.inputs[register])};"
            for register in schedule["input_registers"]
        ]
        operations = [c_instruction(instruction, index) for index, instruction in enumerate(schedule["instructions"])]
        checks = [
            f"        if (overflow != {1 if case.overflow else 0}) fail_case({case_index}, \"overflow\");"
        ]
        if not case.overflow:
            expected = case.expected
            checks.append(
                f"        if (!same_mv(r[{root}], (geo_fixed_cl20_t){c_mv(expected)})) "
                f"fail_case({case_index}, \"root value\");"
            )
        case_blocks.append(f'''    {{
        geo_fixed_cl20_t r[{schedule["register_count"]}] = {{0}};
        int overflow = 0;
{chr(10).join(assignments)}
{chr(10).join(operations)}
{chr(10).join(checks)}
    }}''')

    return f'''#include "geo/fixed.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static geo_fixed_status_t checked_add_component(
    geo_fixed_t a,
    geo_fixed_t b,
    geo_fixed_t *output
) {{
    const int64_t value = (int64_t)a + (int64_t)b;
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}}

static geo_fixed_status_t checked_add_cl20(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {{
    geo_fixed_cl20_t temporary;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    if (checked_add_component(a.scalar, b.scalar, &temporary.scalar) != GEO_FIXED_OK) return GEO_FIXED_OVERFLOW;
    if (checked_add_component(a.e1, b.e1, &temporary.e1) != GEO_FIXED_OK) return GEO_FIXED_OVERFLOW;
    if (checked_add_component(a.e2, b.e2, &temporary.e2) != GEO_FIXED_OK) return GEO_FIXED_OVERFLOW;
    if (checked_add_component(a.e12, b.e12, &temporary.e12) != GEO_FIXED_OK) return GEO_FIXED_OVERFLOW;
    *output = temporary;
    return GEO_FIXED_OK;
}}

static int same_mv(geo_fixed_cl20_t a, geo_fixed_cl20_t b) {{
    return a.scalar == b.scalar && a.e1 == b.e1 &&
        a.e2 == b.e2 && a.e12 == b.e12;
}}

static void fail_case(int index, const char *field) {{
    fprintf(stderr, "FAIL {schedule['name']} case %d: %s\\n", index, field);
    exit(EXIT_FAILURE);
}}

int main(void) {{
{chr(10).join(case_blocks)}
    puts("PASS C schedule {schedule['name']}");
    return EXIT_SUCCESS;
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
    one = 1 << 16
    result, overflow = execute_schedule(
        schedule,
        {0: Multivector(e1=one), 1: Multivector(e2=one)},
        32,
        16,
    )
    assert not overflow and result.e12 == -one
    assert "geo_schedule_self_test" in schedule_module(schedule, 32, 16, "geo_cl20_product_q")
    assert not is_identifier("self_test\ntrailing_junk")
    assert not is_identifier("geo_cl20_product_q\ntrailing_junk")
    invalid_schedule = dict(schedule)
    invalid_schedule["name"] = "self_test\ntrailing_junk"
    try:
        validate_schedule(invalid_schedule)
    except ValueError:
        pass
    else:
        raise AssertionError("schedule identifiers must reject trailing junk")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frac", type=int, default=16)
    parser.add_argument("--product-module", default="geo_cl20_product_q")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.width < 4 or args.frac < 1 or args.frac >= args.width - 1:
        parser.error("invalid fixed-point width/fraction")
    if not is_identifier(args.product_module):
        parser.error("invalid product module name")
    if args.self_test:
        self_test()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for path in args.schedule_json:
        schedule = validate_schedule(json.loads(path.read_text(encoding="utf-8")))
        cases = build_cases(schedule, args.width, args.frac)
        module = f"geo_schedule_{schedule['name']}"
        sv_path = args.out_dir / f"{module}.sv"
        tb_path = args.out_dir / f"tb_{module}.sv"
        c_path = args.out_dir / f"test_{module}.c"
        sv_path.write_text(
            schedule_module(schedule, args.width, args.frac, args.product_module),
            encoding="utf-8",
        )
        tb_path.write_text(
            schedule_testbench(schedule, cases, args.width, args.frac),
            encoding="utf-8",
        )
        c_path.write_text(c_harness(schedule, cases), encoding="utf-8")
        manifest.append({
            "name": schedule["name"],
            "module": module,
            "source": str(path),
            "sv": sv_path.name,
            "testbench": tb_path.name,
            "c_harness": c_path.name,
            "cases": [
                {
                    "name": case.name,
                    "overflow": case.overflow,
                    "expected": case.expected.values(),
                }
                for case in cases
            ],
        })

    (args.out_dir / "schedule_equivalence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
