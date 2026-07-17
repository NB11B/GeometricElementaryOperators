#!/usr/bin/env python3
"""Resolve a usable nvcc executable without assuming the container PATH."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping


def _candidate_paths(env: Mapping[str, str]) -> Iterable[Path]:
    seen: set[str] = set()

    def emit(value: str | None) -> Iterable[Path]:
        if not value:
            return ()
        candidate = Path(value).expanduser()
        key = str(candidate)
        if key in seen:
            return ()
        seen.add(key)
        return (candidate,)

    for variable in ("CUDACXX", "NVCC"):
        yield from emit(env.get(variable))

    path_hit = shutil.which("nvcc", path=env.get("PATH"))
    yield from emit(path_hit)

    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        root = env.get(variable)
        if root:
            yield from emit(str(Path(root) / "bin" / "nvcc"))

    for value in (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ):
        yield from emit(value)


def resolve_cuda_compiler(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    checked: list[str] = []
    for candidate in _candidate_paths(source):
        checked.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    detail = "\n  - ".join(checked) if checked else "<no candidates>"
    raise FileNotFoundError(f"unable to locate an executable nvcc; checked:\n  - {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-env",
        type=Path,
        help="append CUDACXX and CUDA bin directory to a GitHub Actions env file",
    )
    args = parser.parse_args(argv)

    try:
        compiler = resolve_cuda_compiler()
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.github_env is not None:
        with args.github_env.open("a", encoding="utf-8") as stream:
            stream.write(f"CUDACXX={compiler}\n")
            stream.write(f"CUDA_COMPILER_BIN={compiler.parent}\n")

    print(compiler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
