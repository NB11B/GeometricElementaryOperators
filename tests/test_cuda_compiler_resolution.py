#!/usr/bin/env python3
from __future__ import annotations

import stat
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.resolve_cuda_compiler import resolve_cuda_compiler


def make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def expect_equal(actual: Path, expected: Path, message: str) -> None:
    if actual != expected.resolve():
        raise AssertionError(f"{message}: actual={actual} expected={expected.resolve()}")


def test_cudacxx_precedes_path(root: Path) -> None:
    explicit = root / "explicit" / "nvcc"
    path_nvcc = root / "path" / "nvcc"
    make_executable(explicit)
    make_executable(path_nvcc)
    result = resolve_cuda_compiler({"CUDACXX": str(explicit), "PATH": str(path_nvcc.parent)})
    expect_equal(result, explicit, "CUDACXX must take precedence")


def test_path_resolution(root: Path) -> None:
    path_nvcc = root / "path-only" / "nvcc"
    make_executable(path_nvcc)
    result = resolve_cuda_compiler({"PATH": str(path_nvcc.parent)})
    expect_equal(result, path_nvcc, "PATH nvcc must resolve")


def test_cuda_home_resolution(root: Path) -> None:
    cuda_home = root / "cuda-home"
    nvcc = cuda_home / "bin" / "nvcc"
    make_executable(nvcc)
    result = resolve_cuda_compiler({"PATH": "", "CUDA_HOME": str(cuda_home)})
    expect_equal(result, nvcc, "CUDA_HOME/bin/nvcc must resolve")


def test_non_executable_is_rejected(root: Path) -> None:
    candidate = root / "not-executable" / "nvcc"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("not executable\n", encoding="utf-8")
    try:
        resolve_cuda_compiler({"CUDACXX": str(candidate), "PATH": ""})
    except FileNotFoundError:
        return
    raise AssertionError("non-executable CUDA compiler candidate was accepted")


def test_missing_compiler_fails() -> None:
    try:
        resolve_cuda_compiler({"PATH": ""})
    except FileNotFoundError as error:
        if "unable to locate an executable nvcc" not in str(error):
            raise AssertionError(f"unexpected missing-compiler error: {error}") from error
        return
    raise AssertionError("missing CUDA compiler did not fail")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        test_cudacxx_precedes_path(root)
        test_path_resolution(root)
        test_cuda_home_resolution(root)
        test_non_executable_is_rejected(root)
    test_missing_compiler_fails()
    print("CUDA compiler resolution tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
