#!/usr/bin/env python3
"""Regression and semantic tests for the V4.1 fixed-blade extension."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE_EXACT = load_module("geo_identity_discovery", TOOLS / "geo_identity_discovery.py")
EXACT = load_module("geo_identity_v4_1_exact", TOOLS / "geo_identity_v4_1_exact.py")
BASE_COMPILER = load_module("geo_identity_compiler", TOOLS / "geo_identity_compiler.py")
COMPILER = load_module(
    "geo_identity_compiler_v4_1", TOOLS / "geo_identity_compiler_v4_1.py"
)


def fixed(blade: int, coefficient: int = 1) -> dict:
    return {"fixed_blade": {"blade": blade, "coefficient": coefficient}}


def gp(left: dict, right: dict) -> dict:
    return {"op": "gp", "args": [left, right]}


def grade(value: int, arg: dict) -> dict:
    return {"op": "grade", "grade": value, "arg": arg}


def spec(lhs: dict, rhs: dict, *, expected: str = "identity", prime: int = 65521) -> dict:
    return {
        "schema_version": 1,
        "name": "fixed_blade_test",
        "description": "V4.1 fixed-blade regression fixture",
        "expected": expected,
        "dimension": 4,
        "signature": [1, 1, -1, -1],
        "prime": prime,
        "coefficient_bound": 3,
        "seed": 3141592653,
        "variables": [
            {"name": "v", "grades": [1]},
            {"name": "B", "grades": [2]},
            {"name": "T", "grades": [3]},
        ],
        "lhs": lhs,
        "rhs": rhs,
    }


class FixedBladeExactTests(unittest.TestCase):
    def test_supported_blade_constants_validate(self) -> None:
        for blade in (0, 1, 3, 15):
            fixture = spec(fixed(blade), fixed(blade))
            self.assertIs(EXACT.validate_spec(fixture), fixture)
            polynomial = EXACT.extract_polynomial(fixture)
            self.assertTrue(polynomial["zero"], blade)

    def test_pseudoscalar_square_is_one_exactly(self) -> None:
        fixture = spec(gp(fixed(15), fixed(15)), {"scalar": 1})
        polynomial = EXACT.extract_polynomial(fixture)
        self.assertTrue(polynomial["zero"])
        self.assertEqual(polynomial["total_terms"], 0)

    def test_malformed_nodes_are_rejected(self) -> None:
        malformed = [
            {"fixed_blade": {"blade": True, "coefficient": 1}},
            {"fixed_blade": {"blade": -1, "coefficient": 1}},
            {"fixed_blade": {"blade": 16, "coefficient": 1}},
            {"fixed_blade": {"blade": 15, "coefficient": False}},
            {"fixed_blade": {"blade": 15, "coefficient": 0}},
            {"fixed_blade": {"blade": 15, "coefficient": 1, "extra": 0}},
            {"fixed_blade": 15},
            {"fixed_blade": {"blade": 15, "coefficient": 1}, "extra": 0},
        ]
        for expression in malformed:
            with self.subTest(expression=expression):
                with self.assertRaises(EXACT.DiscoveryError):
                    EXACT.validate_spec(spec(expression, {"scalar": 0}))

    def test_dual_grade_support_is_exact(self) -> None:
        cases = (("v", 3), ("B", 2), ("T", 1))
        for variable, target_grade in cases:
            dual = gp({"var": variable}, fixed(15))
            fixture = spec(dual, grade(target_grade, dual))
            self.assertTrue(EXACT.extract_polynomial(fixture)["zero"], variable)


class FixedBladeCompilerTests(unittest.TestCase):
    def write_spec(self, payload: dict, directory: Path, name: str = "fixture.json") -> Path:
        path = directory / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_builder_support_and_python_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = spec(gp(fixed(15), fixed(15)), {"scalar": 1})
            identity = COMPILER.load_identity(self.write_spec(payload, directory))
            constants = [node for node in identity.nodes if node.op == "fixed_blade"]
            self.assertEqual(len(constants), 1)
            self.assertEqual(constants[0].grade, 15)
            self.assertEqual(constants[0].support_mask, 1 << 15)
            for assignment in range(32):
                self.assertTrue(COMPILER.evaluate_identity(identity, assignment)[0])

    def test_dual_round_trip_for_declared_grades(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, variable in enumerate(("v", "B", "T")):
                dual_twice = gp(gp({"var": variable}, fixed(15)), fixed(15))
                payload = spec(dual_twice, {"var": variable})
                payload["name"] = f"dual_round_trip_{variable}"
                identity = COMPILER.load_identity(
                    self.write_spec(payload, directory, f"fixture_{index}.json")
                )
                for assignment in range(64):
                    self.assertTrue(
                        COMPILER.evaluate_identity(identity, assignment)[0],
                        (variable, assignment),
                    )

    def test_generated_header_targets_declared_blade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = spec(gp(fixed(15), fixed(15)), {"scalar": 1})
            path = self.write_spec(payload, directory)
            identity = COMPILER.load_identity(path)
            header = COMPILER.emit_header([identity], [path.as_posix()])
            self.assertIn(".c[15] = 1;", header)
            self.assertNotIn("fixed_blade", header)

    def test_compiler_rejects_invalid_and_zero_mod_prime_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cases = [
                fixed(16),
                fixed(15, 0),
                fixed(15, 65521),
                {"fixed_blade": {"blade": 15, "coefficient": 1, "extra": 0}},
            ]
            for index, expression in enumerate(cases):
                payload = spec(expression, {"scalar": 0})
                payload["name"] = f"invalid_{index}"
                with self.subTest(expression=expression):
                    with self.assertRaises(COMPILER.IdentityError):
                        COMPILER.load_identity(
                            self.write_spec(payload, directory, f"invalid_{index}.json")
                        )


if __name__ == "__main__":
    unittest.main()
