#!/usr/bin/env python3
"""Contract tests for strict fixed GEB-36 numerical report validation."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numerical_report  # noqa: E402


def valid_rows(samples: int = 4) -> list[dict[str, str]]:
    return [
        {
            "operation": operation,
            "backend": numerical_report.BACKEND,
            "expected_kind": kind,
            "requested": str(samples),
            "completed": str(samples),
            "overflows": "0",
            "status_failures": "0",
            "kind_failures": "0",
            "max_absolute": "0",
            "max_relative": "0",
            "max_angular": "0",
            "max_projective_scale": "0",
            "mismatches": "0",
        }
        for operation, kind in numerical_report.MANIFEST
    ]


class NumericalReportContractTests(unittest.TestCase):
    def test_exact_manifest_is_accepted(self) -> None:
        rows = numerical_report.validate_rows(valid_rows(), 4, "double", 16)
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows[34]["expected_kind"], "projective")

    def test_configuration_is_exact_and_unique(self) -> None:
        self.assertEqual(
            numerical_report.parse_configuration(
                "header\nsamples=7 seed=9 precision=double q_fraction_bits=30\n"
            ),
            (7, 9, "double", 30),
        )
        with self.assertRaises(RuntimeError):
            numerical_report.parse_configuration(
                "samples=7 seed=9 precision=double q_fraction_bits=30\n" * 2
            )
        with self.assertRaises(RuntimeError):
            numerical_report.parse_configuration(
                "samples=7 seed=9 precision=double q_fraction_bits=x"
            )
        with self.assertRaises(RuntimeError):
            numerical_report.parse_configuration(
                "samples=7 seed=9 precision=half q_fraction_bits=30"
            )

    def test_configuration_precision_mismatch_is_rejected(self) -> None:
        stdout = "samples=7 seed=9 precision=float q_fraction_bits=16\n"
        with self.assertRaisesRegex(RuntimeError, "configuration does not match"):
            numerical_report.validate_configuration(
                stdout, samples=7, seed=9, precision="double", fraction_bits=16
            )
        self.assertEqual(
            numerical_report.validate_configuration(
                stdout, samples=7, seed=9, precision="float", fraction_bits=16
            ),
            (7, 9, "float", 16),
        )

    def test_missing_extra_and_duplicate_operations_are_rejected(self) -> None:
        cases = []
        cases.append(valid_rows()[:-1])
        extra = valid_rows()
        extra[-1]["operation"] = "not_a_target"
        cases.append(extra)
        duplicate = valid_rows()
        duplicate[-1]["operation"] = duplicate[0]["operation"]
        cases.append(duplicate)
        for rows in cases:
            with self.subTest(last=rows[-1]["operation"]):
                with self.assertRaises(RuntimeError):
                    numerical_report.validate_rows(rows, 4, "double", 16)

    def test_backend_and_kind_are_exact(self) -> None:
        for field, value in (("backend", "other"), ("expected_kind", "scalar")):
            rows = valid_rows()
            rows[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    numerical_report.validate_rows(rows, 4, "double", 16)

    def test_zero_or_too_few_completed_are_rejected(self) -> None:
        for completed, overflows in (("0", "4"), ("3", "1")):
            rows = valid_rows()
            rows[0]["completed"] = completed
            rows[0]["overflows"] = overflows
            with self.subTest(completed=completed):
                with self.assertRaises(RuntimeError):
                    numerical_report.validate_rows(rows, 4, "double", 16)

    def test_bad_accounting_and_mismatch_are_rejected(self) -> None:
        for field, value in (("completed", "3"), ("mismatches", "1")):
            rows = valid_rows()
            rows[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    numerical_report.validate_rows(rows, 4, "double", 16)

    def test_nonfinite_metrics_are_rejected(self) -> None:
        for field, value in (
            ("max_absolute", "nan"),
            ("max_relative", "inf"),
            ("max_angular", "-inf"),
            ("max_projective_scale", "nan"),
        ):
            rows = valid_rows()
            rows[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    numerical_report.validate_rows(rows, 4, "double", 16)

    def test_missing_required_field_is_rejected(self) -> None:
        rows = valid_rows()
        del rows[0]["kind_failures"]
        with self.assertRaises(RuntimeError):
            numerical_report.validate_rows(rows, 4, "double", 16)


if __name__ == "__main__":
    unittest.main()
