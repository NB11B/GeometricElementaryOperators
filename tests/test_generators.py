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


def compiler_command() -> tuple[list[str], bool]:
    raw = os.environ.get("CC", "cc")
    raw_path = Path(raw.strip('"'))
    if raw_path.exists():
        compiler = [str(raw_path)]
    else:
        compiler = shlex.split(raw, posix=os.name != "nt")
    executable = Path(compiler[0]).name.lower()
    return compiler, executable in {"cl", "cl.exe"}


def find_vsdevcmd(compiler_path: str) -> Path | None:
    path = Path(compiler_path).resolve()
    for parent in path.parents:
        candidate = parent / "Common7" / "Tools" / "VsDevCmd.bat"
        if candidate.exists():
            return candidate
    return None


def run_msvc(compiler: list[str], arguments: list[str]) -> None:
    vsdevcmd = find_vsdevcmd(compiler[0])
    if vsdevcmd is None:
        run(*compiler, *arguments)
        return
    compile_command = subprocess.list2cmdline([*compiler, *arguments])
    command = (
        f'call "{vsdevcmd}" -arch=x64 -host_arch=x64 >nul '
        f'&& {compile_command}'
    )
    run("cmd.exe", "/d", "/s", "/c", command)


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
    compiler, is_msvc = compiler_command()
    executable = out / ("test_generated.exe" if os.name == "nt" else "test_generated")
    if is_msvc:
        arguments = [
            "/nologo",
            "/std:c11",
            f"/I{ROOT / 'include'}",
            f"/I{out}",
            str(out / "generated_cl30.c"),
            str(test_c),
            f"/Fe:{executable}",
        ]
        run_msvc(compiler, arguments)
    else:
        command = [
            *compiler,
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="geo-generators-") as directory:
        temp = Path(directory)
        test_clpq(temp)
        test_rtl(temp)
    print("Cl(p,q) and RTL generator tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
