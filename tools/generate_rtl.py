#!/usr/bin/env python3
"""Emit synthesizable SystemVerilog for fixed-point geometric kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def cl20_module(width: int, frac: int, module: str) -> str:
    def mul(name: str, a: str, b: str) -> str:
        return (
            f"logic signed [2*WIDTH-1:0] {name};\n"
            f"assign {name} = $signed({a}) * $signed({b});"
        )

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
    declarations = "\n".join(mul(*product) for product in products)
    rounded = "\n".join(
        f"logic signed [2*WIDTH:0] q{name[1:]};\n"
        f"logic signed [WIDTH-1:0] n{name[1:]};\n"
        f"logic signed [WIDTH+2:0] x{name[1:]};\n"
        f"assign q{name[1:]} = round_product({name});\n"
        f"assign n{name[1:]} = q{name[1:]}[WIDTH-1:0];\n"
        f"assign x{name[1:]} = {{{{3{{n{name[1:]}[WIDTH-1]}}}}, n{name[1:]}}};"
        for name, _, _ in products
    )
    product_checks = " ||\n    ".join(
        f"rounded_overflow(q{name[1:]})" for name, _, _ in products
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
    output logic signed [WIDTH-1:0] y_e12,
    output logic overflow
);

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
    input logic signed [WIDTH+2:0] value
);
    begin
        accumulated_overflow = value[WIDTH+2:WIDTH] != {{3{{value[WIDTH-1]}}}};
    end
endfunction

{declarations}

{rounded}

logic product_overflow;
logic final_overflow;
logic signed [WIDTH+2:0] sum_s;
logic signed [WIDTH+2:0] sum_e1;
logic signed [WIDTH+2:0] sum_e2;
logic signed [WIDTH+2:0] sum_e12;

assign product_overflow =
    {product_checks};

assign sum_s   = x00 + x11 + x22 - x33;
assign sum_e1  = x01 + x10 - x23 + x32;
assign sum_e2  = x02 + x20 + x13 - x31;
assign sum_e12 = x03 + x30 + x12 - x21;
assign final_overflow = accumulated_overflow(sum_s) ||
                        accumulated_overflow(sum_e1) ||
                        accumulated_overflow(sum_e2) ||
                        accumulated_overflow(sum_e12);
assign overflow = product_overflow || final_overflow;
assign y_s   = overflow ? '0 : sum_s[WIDTH-1:0];
assign y_e1  = overflow ? '0 : sum_e1[WIDTH-1:0];
assign y_e2  = overflow ? '0 : sum_e2[WIDTH-1:0];
assign y_e12 = overflow ? '0 : sum_e12[WIDTH-1:0];

endmodule
'''


def cl20_testbench(width: int, frac: int, module: str) -> str:
    one = 1 << frac
    return f'''`timescale 1ns/1ps
module tb_{module};
localparam int WIDTH = {width};
localparam int FRAC = {frac};
localparam logic signed [WIDTH-1:0] ONE = {width}'sd{one};
localparam logic signed [WIDTH-1:0] MAX_FIXED = {{1'b0, {{(WIDTH-1){{1'b1}}}}}};
logic signed [WIDTH-1:0] a_s, a_e1, a_e2, a_e12;
logic signed [WIDTH-1:0] b_s, b_e1, b_e2, b_e12;
logic signed [WIDTH-1:0] y_s, y_e1, y_e2, y_e12;
logic overflow;

{module} #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (.*);

task clear_inputs;
begin
    a_s=0; a_e1=0; a_e2=0; a_e12=0;
    b_s=0; b_e1=0; b_e2=0; b_e12=0;
end
endtask

initial begin
    clear_inputs(); a_e1=ONE; b_e2=ONE; #1;
    if (overflow || y_e12 !== ONE || y_s !== 0 || y_e1 !== 0 || y_e2 !== 0) $fatal(1, "nominal e1*e2");
    clear_inputs(); a_e2=ONE; b_e1=ONE; #1;
    if (overflow || y_e12 !== -ONE) $fatal(1, "nominal e2*e1");
    clear_inputs(); a_e1=ONE; b_e1=ONE; #1;
    if (overflow || y_s !== ONE) $fatal(1, "nominal e1^2");
    clear_inputs(); a_e12=ONE; b_e12=ONE; #1;
    if (overflow || y_s !== -ONE) $fatal(1, "nominal e12^2");
    clear_inputs(); a_s=-1; b_s=ONE/4; #1;
    if (overflow || y_s !== 0) $fatal(1, "negative sub-half rounds toward zero");
    clear_inputs(); a_s=-1; b_s=ONE/2; #1;
    if (overflow || y_s !== -1) $fatal(1, "negative half rounds away from zero");

    clear_inputs(); a_s=MAX_FIXED; b_s=ONE <<< 1; #1;
    if (!overflow) $fatal(1, "rounded multiplication overflow must be signaled");
    if (y_s !== 0 || y_e1 !== 0 || y_e2 !== 0 || y_e12 !== 0) $fatal(1, "multiply-overflow output must be invalidated");

    clear_inputs(); a_s=ONE; b_s=MAX_FIXED; a_e1=ONE; b_e1=MAX_FIXED; #1;
    if (!overflow) $fatal(1, "final wide-sum overflow must be signaled");
    if (y_s !== 0 || y_e1 !== 0 || y_e2 !== 0 || y_e12 !== 0) $fatal(1, "sum-overflow output must be invalidated");

    clear_inputs();
    a_s=-ONE; a_e1=-ONE; a_e2=-ONE; a_e12=-ONE;
    b_s=-MAX_FIXED; b_e1=-MAX_FIXED; b_e2=0; b_e12=-MAX_FIXED; #1;
    if (overflow) $fatal(1, "wide cancellation must remain valid");
    if (y_s !== MAX_FIXED || y_e1 !== MAX_FIXED || y_e2 !== MAX_FIXED || y_e12 !== MAX_FIXED)
        $fatal(1, "wide cancellation result");

    $display("PASS {module}");
    $finish;
end
endmodule
'''


def validate_schedule(data: dict[str, Any]) -> dict[str, Any]:
    instructions = data.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("schedule requires a non-empty instructions array")
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
        if any(not isinstance(operand, int) or operand not in available for operand in operands):
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
    operations = schedule["operations"]
    operation_count = len(operations)
    comments = "\n".join(
        f"// operation {index:3d}, nominal cycle {operation['cycle']:3d}, "
        f"latency {operation['latency']}: {operation['opcode']} "
        f"r{operation['destination']} <- " + ", ".join(f"r{x}" for x in operation["operands"])
        for index, operation in enumerate(operations)
    )
    cases = "\n".join(
        f"            {index}: begin valid <= 1'b1; opcode <= 8'd{index}; "
        f"destination <= 8'd{operation['destination']}; waiting <= 1'b1; end"
        for index, operation in enumerate(operations)
    )
    return f'''module {module} (
    input logic clk,
    input logic reset_n,
    input logic start,
    input logic operation_done,
    input logic operation_overflow,
    output logic busy,
    output logic done,
    output logic valid,
    output logic result_valid,
    output logic overflow,
    output logic [7:0] opcode,
    output logic [7:0] destination
);
localparam int TOTAL_OPERATIONS = {operation_count};
localparam int INDEX_WIDTH = TOTAL_OPERATIONS <= 1 ? 1 : $clog2(TOTAL_OPERATIONS);
logic [INDEX_WIDTH-1:0] operation_index;
logic waiting;

{comments}

// Protocol:
// - reset or an accepted start clears the sticky overflow state.
// - valid is a one-cycle issue pulse; opcode and destination identify that operation.
// - dependencies are serialized: the next operation is not issued until operation_done.
// - operation_overflow is sampled only with operation_done and aborts the schedule.
// - abort and normal completion both pulse done for one cycle.
// - result_valid pulses with done only after successful completion of every operation.
// - after an abort, overflow remains asserted until reset or the next accepted start.
always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        operation_index <= '0;
        waiting <= 1'b0;
        busy <= 1'b0;
        done <= 1'b0;
        valid <= 1'b0;
        result_valid <= 1'b0;
        overflow <= 1'b0;
        opcode <= '0;
        destination <= '0;
    end else begin
        done <= 1'b0;
        valid <= 1'b0;
        result_valid <= 1'b0;

        if (!busy) begin
            if (start) begin
                operation_index <= '0;
                waiting <= 1'b0;
                busy <= 1'b1;
                overflow <= 1'b0;
                opcode <= '0;
                destination <= '0;
            end
        end else if (waiting) begin
            if (operation_done) begin
                waiting <= 1'b0;
                if (operation_overflow) begin
                    overflow <= 1'b1;
                    busy <= 1'b0;
                    done <= 1'b1;
                end else if (operation_index == TOTAL_OPERATIONS-1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    result_valid <= 1'b1;
                end else begin
                    operation_index <= operation_index + 1'b1;
                end
            end
        end else begin
            case (operation_index)
{cases}
                default: begin
                    overflow <= 1'b1;
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end
end
endmodule
'''


def controller_testbench(schedule: dict[str, Any], module: str) -> str:
    operation_count = len(schedule["operations"])
    return f'''`timescale 1ns/1ps
module tb_{module};
logic clk=0, reset_n=0, start=0;
logic operation_done=0, operation_overflow=0;
logic busy, done, valid, result_valid, overflow;
logic [7:0] opcode, destination;
integer issued_count=0;
{module} dut (.*);
always #5 clk = ~clk;
always @(posedge clk) if (valid) issued_count <= issued_count + 1;

task pulse_start;
begin
    @(negedge clk); start=1'b1;
    @(negedge clk); start=1'b0;
end
endtask

task complete_current;
    input logic fail;
begin
    wait(valid === 1'b1);
    #1;
    operation_overflow = fail;
    operation_done = 1'b1;
    @(posedge clk); #1;
    operation_done = 1'b0;
    operation_overflow = 1'b0;
end
endtask

integer index;
initial begin
    #12 reset_n=1;

    pulse_start();
    for (index=0; index<{operation_count}; index=index+1) complete_current(1'b0);
    if (!done || !result_valid || overflow || busy) $fatal(1, "successful completion protocol");
    @(posedge clk); #1;
    if (done || result_valid) $fatal(1, "completion outputs must pulse for one cycle");

    issued_count = 0;
    pulse_start();
    complete_current(1'b1);
    if (!done || result_valid || !overflow || busy) $fatal(1, "overflow abort protocol");
    @(posedge clk); #1;
    if (valid) $fatal(1, "no dependent operation may issue after overflow");
    if (!overflow) $fatal(1, "overflow must remain sticky while idle");

    pulse_start();
    @(posedge clk); #1;
    if (overflow || !busy) $fatal(1, "accepted start must clear overflow");
    for (index=0; index<{operation_count}; index=index+1) complete_current(1'b0);
    if (!done || !result_valid || overflow || busy) $fatal(1, "restart after overflow");

    if (issued_count < {operation_count}) $fatal(1, "controller did not issue the restarted schedule");
    $display("PASS {module}");
    $finish;
end
endmodule
'''


def self_test() -> None:
    sample = {"input_registers": [0, 1], "instructions": [
        {"opcode": "cl20_product", "destination": 2, "left": 0, "right": 1},
        {"opcode": "reverse", "destination": 3, "source": 2},
    ]}
    schedule = validate_schedule(sample)
    assert schedule["total_cycles"] == 3
    assert schedule["register_count"] == 4
    text = cl20_module(32, 16, "geo_cl20_product_q")
    assert "rounded_overflow" in text
    assert "final_overflow" in text
    assert "wide cancellation" in cl20_testbench(32, 16, "geo_cl20_product_q")
    controller = schedule_sv(schedule, "geo_program_controller")
    assert "operation_done" in controller
    assert "result_valid" in controller
    assert "sticky overflow" in controller


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frac", type=int, default=16)
    parser.add_argument("--module", default="geo_cl20_product_q")
    parser.add_argument("--schedule-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.width < 4 or args.frac < 1 or args.frac >= args.width - 1:
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
        (args.out_dir / "rtl_schedule.json").write_text(
            json.dumps(schedule, indent=2), encoding="utf-8"
        )
        controller = "geo_program_controller"
        (args.out_dir / f"{controller}.sv").write_text(
            schedule_sv(schedule, controller), encoding="utf-8"
        )
        (args.out_dir / f"tb_{controller}.sv").write_text(
            controller_testbench(schedule, controller), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
