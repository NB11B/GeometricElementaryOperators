#!/usr/bin/env python3
"""Tests for deterministic CUDA emission of the IMU sparse schedule."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_imu_cuda_schedule.py"
SCHEDULE_PATH = (
    ROOT
    / "benchmarks"
    / "esp32_imu_baseline"
    / "schedules"
    / "imu_orientation_sparse_v1.json"
)
HEADER_PATH = (
    ROOT
    / "benchmarks"
    / "gpu_imu_replay"
    / "generated"
    / "geo_imu_generated_schedule.cuh"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_module("generate_imu_cuda_schedule", GENERATOR_PATH)


class CudaScheduleTests(unittest.TestCase):
    def test_checked_in_header_matches_generation(self) -> None:
        schedule = GENERATOR.BASE.load_schedule(SCHEDULE_PATH)
        generated = GENERATOR.emit_cuda_header(
            schedule,
            SCHEDULE_PATH.relative_to(ROOT).as_posix(),
        )
        self.assertEqual(HEADER_PATH.read_text(encoding="utf-8"), generated)

    def test_cli_check_mode(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--schedule",
                str(SCHEDULE_PATH),
                "--output",
                str(HEADER_PATH),
                "--check",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS:", completed.stdout)

    def test_cuda_header_contains_both_backends(self) -> None:
        header = HEADER_PATH.read_text(encoding="utf-8")
        self.assertIn("geo_gpu_generated_float_gravity", header)
        self.assertIn("geo_gpu_generated_q32_gravity", header)
        self.assertIn(
            "geo_gpu_generated_float_q_times_vector_quaternion",
            header,
        )
        self.assertIn(
            "geo_gpu_generated_q32_q_times_vector_quaternion",
            header,
        )
        self.assertIn("__device__ __forceinline__", header)


if __name__ == "__main__":
    unittest.main()
