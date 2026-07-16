#!/usr/bin/env python3
"""Emit synthesizable SystemVerilog for fixed-point geometric kernels.

The primary target is a pipelinable Cl(2,0) product cell. The schedule emitter
also converts optimized banked-program JSON into an explicit cycle-by-cycle RTL
operation schedule suitable for downstream HLS, handwritten RTL, or formal
verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def cl20_module(width: int, frac: int, module: str) -> str:
    def mul(name: str, a: str, b: str) -> str:
        return f"logic signed [{2*width-1}:0] {name};\nassign {name} = $signed({a}) * $signed({b});"

    products = [
        ("p00", "a_s", "b_s"), ("p11", "a_e1", "b_e1"),
        ("p22", "a_e2", "b_e2"), ("p33", "a_e12", "b_e12"),
        ("p01", "a_s", "b_e1"), ("p10", "a_e1", "b_s"),
        ("p23", "a_e2", "b_e12"), ("p32", "a_e12", "b_e2"),
        ("p02", "a_s", "b_e2"), ("p20", "a_e2", "b_s"),
        ("p13", "a_e1", "b_e12"), ("p31", "a_e12", "b_e1"),
        ("p03", "a_s", "b_e12"), ("p30", "a_e12", "b_s"),
        ("p12", "a_e1", "b_e2"), ("p21", "a_e2", "b_e1"),
    ]
    declarations = "\n".join(mul(*p) for p in products)
    shifted = "\n".join(
        f"logic signed [{width}:0] q{name[1:]};\n"
        f"assign q{name[1:]} = $signed({name} >>> FRAC);"
        for name, _, _ in products
    )
    return f'''module {module} #(
    parameter int WIDTH = {width},
    parameter int FRAC = {frac}
) (
    input  logic signed [WIDTH-1:0] a_s,
    input  logic signed [WIDTH-1:0] a_e1,
    input  logic signed [WIDTH-1:0] a_e2,
    input  logic signed [WIDTH-1:0] a_e12,
    input  logic signed [WIDTH-1:0] b_s,
    input  logic signed [WIDTH-1:0] b_e1,
    input  logic signed [WIDTH-1:0] b_e2,
    input  logic signed [WIDTH-1:0] b_e12,
    output logic signed [WIDTH-1:0] y_s,
    output logic signed [WIDTH-1:0] y_e1,
    output logic signed [WIDTH-1:0] y_e2,
    output logic signed [WIDTH-1:0] y_e12
);

{declarations}

{shifted}

logic signed [WIDTH+2:0] sum_s;
logic signed [WIDTH+2:0] sum_e1;
logic signed [WIDTH+2:0] sum_e2;
logic signed [WIDTH+2:0] sum_e12;

always_comb begin
    sum_s   = q00 + q11 + q22 - q33;
    sum_e1  = q01 + q10 - q23 + q32;
    sum_e2  = q02 + q20 + q13 - q31;
    sum_e12 = q03 + q30 + q12 - q21;
    y_s   = sum_s[WIDTH-1:0];
    y_e1  = sum_e1[WIDTH-1:0];
    y_e2  = sum_e2[WIDTH-1:0];
    y_e12 = sum_e12[WIDTH-1:0];
end

endmodule
'''


def cl20_testbench(width: int, frac: int, module: str) -> str:
    one = 1 << frac
    return f'''`timescale 1ns/1ps
module tb_{module};
localparam int WIDTH = {width};
localparam int FRAC = {frac};
localparam logic signed [WIDTH-1:0] ONE = {width}'sd{one};
logic signed [WIDTH-1:0] a_s, a_e1, a_e2, a_e12;
logic signed [WIDTH-1:0] b_s, b_e1, b_e2, b_e12;
logic signed [WIDTH-1:0] y_s, y_e1, y_e2, y_e12;

{module} #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (.*);

task clear_inputs;
begin
    a_s=0; a_e1=0; a_e2=0; a_e12=0;
    b_s=0; b_e1=0; b_e2=0; b_e12=0;
end
endtask

initial begin
    clear_inputs(); a_e1=ONE; b_e2=ONE; #1;
    if (y_e12 !== ONE || y_s !== 0 || y_e1 !== 0 || y_e2 !== 0) $fatal(1, "e1*e2");
    clear_inputs(); a_e2=ONE; b_e1=ONE; #1;
    if (y_e12 !== -ONE) $fatal(1, "e2*e1");
    clear_inputs(); a_e1=ONE; b_e1=ONE; #1;
    if (y_s !== ONE) $fatal(1, "e1^2");
    clear_inputs(); a_e12=ONE; b_e12=ONE; #1;
    if (y_s !== -ONE) $fatal(1, "e12^2");
    clear_inputs(); a_s=ONE; b_s=ONE; #1;
    if (y_s !== ONE) $fatal(1, "scalar product");
    $display("PASS {module}");
    $finish;
end
endmodule
'''


def validate_schedule(data: dict[str, Any]) -> dict[str, Any]:
    instructions = data.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError("schedule requires an instructions array")
    available = set(data.get("input_registers", []))
    schedule: list[dict[str, Any]] = []
    cycle = 0
    for index, insn in enumerate(instructions):
        if not isinstance(insn, dict):
            raise ValueError(f"instruction {index} is not an object")
        opcode = insn.get("opcode")
        destination = insn.get("destination")
        operands = [value for key, value in insn.items() if key in {"left", "right", "source"}]
        if not isinstance(opcode, str) or not opcode:
            raise ValueError(f"instruction {index} has invalid opcode")
        if not isinstance(destination, int) or destination < 0:
            raise ValueError(f"instruction {index} has invalid destination")
        if any(not isinstance(op, int) or op not in available for op in operands):
            raise ValueError(f"instruction {index} reads unavailable register")
        latency = 2 if opcode in {"cl20_product", "omega_geometric"} else 1
        schedule.append({
            "cycle": cycle,
            "latency": latency,
            "opcode": opcode,
            "destination": destination,
            "operands": operands,
        })
        cycle += latency
        available.add(destination)
    return {
        "name": data.get("name", "geo_schedule"),
        "total_cycles": cycle,
        "register_count": (max(available) + 1) if available else 0,
        "operations": schedule,
    }


def schedule_sv(schedule: dict[str, Any], module: str) -> str:
    cycles = max(1, int(schedule["total_cycles"]))
    operations = schedule["operations"]
    comments = "\n".join(
        f"// cycle {op['cycle']:3d}: {op['opcode']} r{op['destination']} <- "
        + ", ".join(f"r{x}" for x in op["operands"])
        for op in operations
    )
    cases = "\n".join(
        f"            {op['cycle']}: begin valid <= 1'b1; opcode <= 8'd{index}; destination <= 8'd{op['destination']}; end"
        for index, op in enumerate(operations)
    )
    return f'''module {module} (
    input logic clk,
    input logic reset_n,
    input logic start,
    output logic busy,
    output logic done,
    output logic valid,
    output logic [7:0] opcode,
    output logic [7:0] destination
);
localparam int TOTAL_CYCLES = {cycles};
logic [$clog2(TOTAL_CYCLES+1)-1:0] cycle;

{comments}

always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        cycle <= '0; busy <= 1'b0; done <= 1'b0; valid <= 1'b0;
        opcode <= '0; destination <= '0;
    end else begin
        done <= 1'b0; valid <= 1'b0;
        if (start && !busy) begin
            busy <= 1'b1; cycle <= '0;
        end else if (busy) begin
            case (cycle)
{cases}
                default: begin end
            endcase
            if (cycle == TOTAL_CYCLES-1) begin
                busy <= 1'b0; done <= 1'b1;
            end else cycle <= cycle + 1'b1;
        end
    end
end
endmodule
'''


def self_test() -> None:
    sample = {
        "input_registers": [0, 1],
        "instructions": [
            {"opcode": "cl20_product", "destination": 2, "left": 0, "right": 1},
            {"opcode": "reverse", "destination": 3, "source": 2},
        ],
    }
    schedule = validate_schedule(sample)
    assert schedule["total_cycles"] == 3
    assert schedule["register_count"] == 4
    text = cl20_module(32, 16, "geo_cl20_product_q")
    assert "always_comb" in text and "a_e12" in text
    assert "sum_s   = q00" in text and "sum_s   = qs" not in text


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frac", type=int, default=16)
    parser.add_argument("--module", default="geo_cl20_product_q")
    parser.add_argument("--schedule-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.width < 4 or args.frac < 0 or args.frac >= args.width - 1:
        parser.error("invalid fixed-point width/fraction")
    if args.self_test:
        self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.module}.sv").write_text(
        cl20_module(args.width, args.frac, args.module), encoding="utf-8"
    )
    (args.out_dir / f"tb_{args.module}.sv").write_text(
        cl20_testbench(args.width, args.frac, args.module), encoding="utf-8"
    )
    if args.schedule_json:
        data = json.loads(args.schedule_json.read_text(encoding="utf-8"))
        schedule = validate_schedule(data)
        (args.out_dir / "rtl_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
        (args.out_dir / "geo_program_controller.sv").write_text(
            schedule_sv(schedule, "geo_program_controller"), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
