#!/usr/bin/env python3
"""Contract tests for the unified workload report collector."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "geo_workload_report", ROOT / "tools" / "workload_report.py"
)
assert SPEC is not None and SPEC.loader is not None
workload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workload
SPEC.loader.exec_module(workload)


def cuda_rows(
    *,
    precision: str = "double",
    batch: int = 32,
    iterations: int = 4,
    warmup: int = 1,
    seed: int = 1234,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for operation, backend, scope in sorted(
        workload.CUDA_CAPABILITY_MANIFEST
    ):
        upload, download, logical = workload.expected_cuda_byte_counts(
            (operation, backend, scope), precision, batch
        )
        rows.append({
            "operation": operation,
            "backend": backend,
            "timing_scope": scope,
            "precision": precision,
            "batch": str(batch),
            "iterations": str(iterations),
            "warmup": str(warmup),
            "seed": str(seed),
            "upload_bytes": str(upload),
            "download_bytes": str(download),
            "logical_kernel_bytes": str(logical),
            "ns_per_item": "2.5",
            "max_absolute_error": "0",
            "max_relative_error": "0",
            "mismatches": "0",
        })
    return rows


def validate(rows: list[dict[str, str]], **overrides: object) -> object:
    arguments: dict[str, object] = {
        "precision": "double",
        "batch": 32,
        "iterations": 4,
        "warmup": 1,
        "seed": 1234,
    }
    arguments.update(overrides)
    return workload.validate_cuda_csv_rows(rows, **arguments)


class ManifestTests(unittest.TestCase):
    def test_cpu_configuration_is_authoritative(self) -> None:
        valid = (
            "configuration: precision=double iterations=100 warmup=10 "
            "seed=1234\n"
        )
        arguments = {
            "precision": "double",
            "iterations": 100,
            "warmup": 10,
            "seed": 1234,
        }
        workload.validate_cpu_config(valid, **arguments)
        for key, wrong in (
            ("precision", "float"),
            ("iterations", 99),
            ("warmup", 9),
            ("seed", 1235),
        ):
            mismatched = dict(arguments)
            mismatched[key] = wrong
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, "CPU harness configuration"
            ):
                workload.validate_cpu_config(valid, **mismatched)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            workload.validate_cpu_config(valid + valid, **arguments)

    def test_exact_cpu_and_cuda_manifests(self) -> None:
        workload.validate_exact_manifest(
            set(workload.CPU_CAPABILITY_MANIFEST),
            workload.CPU_CAPABILITY_MANIFEST,
            "CPU",
        )
        parsed = validate(cuda_rows())
        self.assertEqual(len(parsed), 11)
        self.assertIn(
            ("reverse", "cuda_public_api", "host_end_to_end"),
            workload.CUDA_CAPABILITY_MANIFEST,
        )
        self.assertNotIn(
            ("reverse", "cuda_generated_schedule", "device_kernel"),
            workload.CUDA_CAPABILITY_MANIFEST,
        )
        self.assertIn(
            ("reverse_product", "direct_c", "host_operation"),
            workload.CPU_CAPABILITY_MANIFEST,
        )

    def test_manifest_rejects_omission_extra_and_duplicate(self) -> None:
        rows = cuda_rows()
        with self.assertRaisesRegex(RuntimeError, "missing="):
            validate(rows[:-1])

        extra = dict(rows[0])
        extra["operation"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "extra="):
            validate(rows + [extra])

        with self.assertRaisesRegex(RuntimeError, "duplicate CUDA"):
            validate(rows + [dict(rows[0])])

    def test_backend_scope_precision_and_mismatches_are_enforced(self) -> None:
        rows = cuda_rows()
        rows[0]["backend"] = "cuda_kernel"
        with self.assertRaisesRegex(RuntimeError, "capability manifest"):
            validate(rows)

        rows = cuda_rows()
        rows[0]["timing_scope"] = "host_end_to_end"
        with self.assertRaisesRegex(RuntimeError, "capability manifest"):
            validate(rows)

        rows = cuda_rows()
        rows[0]["precision"] = "float"
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            validate(rows)

        rows = cuda_rows()
        rows[0]["mismatches"] = "1"
        with self.assertRaisesRegex(RuntimeError, "recorded 1 mismatches"):
            validate(rows)

    def test_seed_is_validated_in_banner_and_csv(self) -> None:
        valid = (
            "configuration: precision=double batch=32 iterations=4 "
            "warmup=1 seed=1234 operation=all\n"
        )
        workload.validate_cuda_config(
            valid,
            precision="double",
            batch=32,
            iterations=4,
            warmup=1,
            seed=1234,
        )
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            workload.validate_cuda_config(
                valid,
                precision="double",
                batch=32,
                iterations=4,
                warmup=1,
                seed=9,
            )
        rows = cuda_rows()
        rows[0]["seed"] = "9"
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            validate(rows)

    def test_byte_accounting_is_operation_and_path_specific(self) -> None:
        rows = cuda_rows()
        public_wedge = next(
            row for row in rows
            if row["operation"] == "vector_wedge" and
            row["backend"] == "cuda_public_api"
        )
        generated_wedge = next(
            row for row in rows
            if row["operation"] == "vector_wedge" and
            row["backend"] == "cuda_generated_schedule"
        )
        self.assertNotEqual(
            public_wedge["download_bytes"],
            generated_wedge["download_bytes"],
        )
        generated_wedge["logical_kernel_bytes"] = "1"
        with self.assertRaisesRegex(RuntimeError, "byte accounting mismatch"):
            validate(rows)


class CudaAvailabilityTests(unittest.TestCase):
    def collect(self, require_cuda: bool) -> tuple[bool, str, object]:
        return workload.collect_cuda(
            Path("bench_cuda"),
            1,
            [32],
            4,
            1,
            1234,
            "double",
            defaultdict(workload.Samples),
            require_cuda=require_cuda,
        )

    def test_optional_cuda_skip_forwards_seed(self) -> None:
        skipped = subprocess.CompletedProcess(
            ["bench_cuda"], 77, stdout="", stderr="no device"
        )
        with mock.patch.object(workload, "run_command", return_value=skipped) as run:
            included, reason, device = self.collect(False)
        self.assertFalse(included)
        self.assertEqual(reason, "no device")
        self.assertIsNone(device)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--seed") + 1], "1234")
        self.assertTrue(run.call_args.kwargs["allow_skip"])

    def test_required_cuda_skip_is_failure(self) -> None:
        skipped = subprocess.CompletedProcess(
            ["bench_cuda"], 77, stdout="", stderr="no device"
        )
        with mock.patch.object(workload, "run_command", return_value=skipped):
            with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                self.collect(True)


if __name__ == "__main__":
    unittest.main()
