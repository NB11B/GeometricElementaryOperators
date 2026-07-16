#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def compiler_command() -> list[str]:
    return shlex.split(os.environ.get("CC", "cc"))


def schedule_paths() -> list[Path]:
    return [
        ROOT / "rtl" / "examples" / "addition_schedule.json",
        ROOT / "rtl" / "examples" / "geometric_product_schedule.json",
        ROOT / "rtl" / "examples" / "vector_dot_schedule.json",
        ROOT / "rtl" / "examples" / "vector_wedge_schedule.json",
        ROOT / "rtl" / "examples" / "rotor_action_schedule.json",
    ]


def test_clpq(temp: Path) -> None:
    out = temp / "clpq"
    run(
        PYTHON,
        str(ROOT / "tools" / "generate_clpq.py"),
        "--p", "3", "--q", "0",
        "--prefix", "generated_cl30",
        "--out-dir", str(out),
        "--self-test",
    )
    manifest = json.loads((out / "generated_cl30.json").read_text(encoding="utf-8"))
    assert manifest["blade_count"] == 8
    assert manifest["signature"] == {"p": 3, "q": 0}

    test_c = out / "test_generated.c"
    test_c.write_text(
        '#include "generated_cl30.h"\n'
        'int main(void) {\n'
        '  generated_cl30_t e1=generated_cl30_basis(1u);\n'
        '  generated_cl30_t e2=generated_cl30_basis(2u);\n'
        '  generated_cl30_t e12=generated_cl30_mul(e1,e2);\n'
        '  generated_cl30_t reverse=generated_cl30_mul(e2,e1);\n'
        '  return !(e12.c[3] > 0.999 && reverse.c[3] < -0.999);\n'
        '}\n',
        encoding="utf-8",
    )
    executable = out / ("test_generated.exe" if os.name == "nt" else "test_generated")
    command = [
        *compiler_command(),
        "-std=c11",
        "-I", str(ROOT / "include"),
        "-I", str(out),
        str(out / "generated_cl30.c"),
        str(test_c),
    ]
    if os.name != "nt":
        command.append("-lm")
    command.extend(["-o", str(executable)])
    run(*command)
    run(str(executable))


def test_rtl(temp: Path) -> None:
    out = temp / "rtl"
    run(
        PYTHON,
        str(ROOT / "tools" / "generate_rtl.py"),
        "--out-dir", str(out),
        "--schedule-json", str(ROOT / "rtl" / "examples" / "rotor_action_schedule.json"),
        "--self-test",
    )
    schedule = json.loads((out / "rtl_schedule.json").read_text(encoding="utf-8"))
    assert schedule["total_cycles"] == 4
    assert len(schedule["operations"]) == 2

    sv = (out / "geo_cl20_product_q.sv").read_text(encoding="utf-8")
    assert "output logic overflow" in sv
    assert "rounded_overflow" in sv
    assert "final_overflow" in sv
    assert "assign sum_s" in sv

    product_testbench = (out / "tb_geo_cl20_product_q.sv").read_text(encoding="utf-8")
    assert "rounded multiplication overflow" in product_testbench
    assert "final wide-sum overflow" in product_testbench
    assert "wide cancellation" in product_testbench

    controller = (out / "geo_program_controller.sv").read_text(encoding="utf-8")
    assert "TOTAL_OPERATIONS = 2" in controller
    assert "operation_done" in controller
    assert "operation_overflow" in controller
    assert "result_valid" in controller
    assert "overflow remains asserted" in controller

    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog and vvp:
        product_simulation = out / "cl20_tb.vvp"
        run(
            iverilog,
            "-g2012",
            "-s", "tb_geo_cl20_product_q",
            "-o", str(product_simulation),
            str(out / "geo_cl20_product_q.sv"),
            str(out / "tb_geo_cl20_product_q.sv"),
        )
        run(vvp, str(product_simulation))

        controller_simulation = out / "controller_tb.vvp"
        run(
            iverilog,
            "-g2012",
            "-s", "tb_geo_program_controller",
            "-o", str(controller_simulation),
            str(out / "geo_program_controller.sv"),
            str(out / "tb_geo_program_controller.sv"),
        )
        run(vvp, str(controller_simulation))


def test_rtl_schedules(temp: Path) -> None:
    out = temp / "rtl-schedules"
    paths = schedule_paths()

    run(
        PYTHON,
        str(ROOT / "tools" / "generate_rtl.py"),
        "--out-dir", str(out),
        "--schedule-json", str(paths[-1]),
        "--self-test",
    )
    command = [
        PYTHON,
        str(ROOT / "tools" / "generate_rtl_schedule.py"),
        "--out-dir", str(out),
        "--self-test",
    ]
    for schedule_path in paths:
        command.extend(["--schedule-json", str(schedule_path)])
    run(*command)

    manifest = json.loads(
        (out / "schedule_equivalence_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == len(paths)
    assert {entry["name"] for entry in manifest} == {
        "addition",
        "geometric_product",
        "vector_dot",
        "vector_wedge",
        "rotor_action",
    }

    for entry in manifest:
        c_source = out / entry["c_harness"]
        executable = out / (
            f"test_{entry['name']}.exe" if os.name == "nt" else f"test_{entry['name']}"
        )
        compile_command = [
            *compiler_command(),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Wconversion",
            "-I", str(ROOT / "include"),
            str(c_source),
            str(ROOT / "src" / "fixed.c"),
        ]
        if os.name != "nt":
            compile_command.append("-lm")
        compile_command.extend(["-o", str(executable)])
        run(*compile_command)
        run(str(executable))

        module_text = (out / entry["sv"]).read_text(encoding="utf-8")
        assert "output logic overflow" in module_text
        assert "overflow ? '0" in module_text

    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog and vvp:
        for entry in manifest:
            module = entry["module"]
            simulation = out / f"{module}.vvp"
            run(
                iverilog,
                "-g2012",
                "-s", f"tb_{module}",
                "-o", str(simulation),
                str(out / "geo_cl20_product_q.sv"),
                str(out / entry["sv"]),
                str(out / entry["testbench"]),
            )
            run(vvp, str(simulation))


def test_cuda_schedules(temp: Path) -> None:
    out = temp / "cuda-schedules"
    paths = schedule_paths()
    command = [
        PYTHON,
        str(ROOT / "tools" / "generate_cuda_schedule.py"),
        "--out-dir", str(out),
        "--self-test",
    ]
    for schedule_path in paths:
        command.extend(["--schedule-json", str(schedule_path)])
    run(*command)

    manifest = json.loads(
        (out / "cuda_schedule_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == len(paths)
    assert {entry["name"] for entry in manifest} == {
        "addition",
        "geometric_product",
        "vector_dot",
        "vector_wedge",
        "rotor_action",
    }
    for entry in manifest:
        header = (out / entry["header"]).read_text(encoding="utf-8")
        source = (out / entry["source"]).read_text(encoding="utf-8")
        assert f"geo_cuda_schedule_{entry['name']}_launch" in header
        assert f"geo_cuda_schedule_{entry['name']}_kernel" in source
        assert "cudaGetLastError" in source
        assert "SIZE_MAX" in source


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="geo-generators-") as directory:
        temp = Path(directory)
        test_clpq(temp)
        test_rtl(temp)
        test_rtl_schedules(temp)
        test_cuda_schedules(temp)
    print(
        "Cl(p,q), RTL product/controller, executable RTL schedules, and CUDA schedule generator tests passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
