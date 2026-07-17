#!/usr/bin/env python3
"""Tests for deterministic ESP32 IMU sparse schedule generation."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_imu_sparse_schedule.py"
SCHEDULE_RELATIVE = Path(
    "benchmarks/esp32_imu_baseline/schedules/imu_orientation_sparse_v1.json"
)
GENERATED_HEADER_RELATIVE = Path(
    "benchmarks/esp32_imu_baseline/main/geo_imu_generated_schedule.h"
)
SCHEDULE_PATH = ROOT / SCHEDULE_RELATIVE
GENERATED_HEADER_PATH = ROOT / GENERATED_HEADER_RELATIVE


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_imu_sparse_schedule",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_generator_module()


def direct_gravity(values: dict[str, int | float]) -> dict[str, int | float]:
    qw = values["qw"]
    qx = values["qx"]
    qy = values["qy"]
    qz = values["qz"]
    return {
        "gx": 2 * (qx * qz - qw * qy),
        "gy": 2 * (qw * qx + qy * qz),
        "gz": qw * qw - qx * qx - qy * qy + qz * qz,
    }


def direct_q_times_vector(
    values: dict[str, int | float],
) -> dict[str, int | float]:
    qw = values["qw"]
    qx = values["qx"]
    qy = values["qy"]
    qz = values["qz"]
    wx = values["wx"]
    wy = values["wy"]
    wz = values["wz"]
    return {
        "dw": -qx * wx - qy * wy - qz * wz,
        "dx": qw * wx + qy * wz - qz * wy,
        "dy": qw * wy - qx * wz + qz * wx,
        "dz": qw * wz + qx * wy - qy * wx,
    }


class SparseScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = GENERATOR.load_schedule(SCHEDULE_PATH)
        cls.functions = {
            function.name: function for function in cls.schedule.functions
        }

    def test_expected_functions_are_present(self) -> None:
        self.assertEqual(
            set(self.functions),
            {"gravity", "q_times_vector_quaternion"},
        )

    def test_checked_in_header_matches_deterministic_generation(self) -> None:
        generated = GENERATOR.emit_header(
            self.schedule,
            SCHEDULE_RELATIVE.as_posix(),
        )
        checked_in = GENERATED_HEADER_PATH.read_text(encoding="utf-8")
        self.assertEqual(checked_in, generated)

    def test_generator_cli_check_mode(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--schedule",
                SCHEDULE_RELATIVE.as_posix(),
                "--output",
                GENERATED_HEADER_RELATIVE.as_posix(),
                "--check",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS:", completed.stdout)

    def test_float_schedule_matches_direct_formulas(self) -> None:
        rng = random.Random(0x47454F)
        gravity = self.functions["gravity"]
        q_times_vector = self.functions["q_times_vector_quaternion"]

        for _ in range(1000):
            quaternion_values = {
                name: rng.uniform(-1.25, 1.25)
                for name in gravity.inputs
            }
            scheduled_gravity = GENERATOR.evaluate_function(
                gravity,
                quaternion_values,
            )
            expected_gravity = direct_gravity(quaternion_values)
            for name, expected in expected_gravity.items():
                self.assertTrue(
                    math.isclose(
                        float(scheduled_gravity[name]),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )

            vector_values = dict(quaternion_values)
            vector_values.update(
                {
                    "wx": rng.uniform(-4.0, 4.0),
                    "wy": rng.uniform(-4.0, 4.0),
                    "wz": rng.uniform(-4.0, 4.0),
                }
            )
            scheduled_vector = GENERATOR.evaluate_function(
                q_times_vector,
                vector_values,
            )
            expected_vector = direct_q_times_vector(vector_values)
            for name, expected in expected_vector.items():
                self.assertTrue(
                    math.isclose(
                        float(scheduled_vector[name]),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )

    def test_integer_accumulators_match_direct_formulas(self) -> None:
        rng = random.Random(0x513136)
        gravity = self.functions["gravity"]
        q_times_vector = self.functions["q_times_vector_quaternion"]

        for _ in range(1000):
            quaternion_values = {
                name: rng.randint(-131072, 131072)
                for name in gravity.inputs
            }
            self.assertEqual(
                GENERATOR.evaluate_function(gravity, quaternion_values),
                direct_gravity(quaternion_values),
            )

            vector_values = dict(quaternion_values)
            vector_values.update(
                {
                    "wx": rng.randint(-262144, 262144),
                    "wy": rng.randint(-262144, 262144),
                    "wz": rng.randint(-262144, 262144),
                }
            )
            self.assertEqual(
                GENERATOR.evaluate_function(q_times_vector, vector_values),
                direct_q_times_vector(vector_values),
            )

    def test_invalid_zero_coefficient_is_rejected(self) -> None:
        raw = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        raw["functions"][0]["outputs"][0]["terms"][0]["coefficient"] = 0

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.json"
            invalid_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(GENERATOR.ScheduleError):
                GENERATOR.load_schedule(invalid_path)

    def test_invalid_unknown_factor_is_rejected(self) -> None:
        raw = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        raw["functions"][1]["outputs"][0]["terms"][0]["factors"][1] = "missing"

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.json"
            invalid_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(GENERATOR.ScheduleError):
                GENERATOR.load_schedule(invalid_path)


if __name__ == "__main__":
    unittest.main()
