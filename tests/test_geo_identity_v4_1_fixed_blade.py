#!/usr/bin/env python3
"""Regression and semantic tests for the V4.1 fixed-blade extension."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TOOLS_PATH = str(TOOLS)
if TOOLS_PATH in sys.path:
    sys.path.remove(TOOLS_PATH)
sys.path.insert(0, TOOLS_PATH)

import geo_identity_v4_1_ir as ir


def fixed(blade: int, coefficient: int = 1) -> dict:
    return {"fixed_blade": {"blade": blade, "coefficient": coefficient}}


def gp(left: dict, right: dict) -> dict:
    return {"op": "gp", "args": [left, right]}


def grade(value: int, arg: dict) -> dict:
    return {"op": "grade", "grade": value, "arg": arg}


class FixedBladeTests(unittest.TestCase):
    def base_spec(self) -> dict:
        return {
            "schema_version": 1,
            "name": "fixed_blade_test",
            "description": "V4.1 fixed-blade regression fixture",
            "expected": "identity",
            "dimension": 4,
            "signature": [1, 1, -1, -1],
            "prime": 65521,
            "coefficient_bound": 3,
            "seed": 3141592653,
            "variables": [
                {"name": "v", "grades": [1]},
                {"name": "B", "grades": [2]},
                {"name": "T", "grades": [3]},
            ],
            "lhs": fixed(15),
            "rhs": fixed(15),
        }

    def write_spec(self, payload: dict, directory: Path, name: str) -> Path:
        path = directory / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_validation_accepts_representative_blades(self) -> None:
        for blade in (0, 1, 3, 15):
            spec = self.base_spec()
            spec["lhs"] = fixed(blade, -2)
            spec["rhs"] = copy.deepcopy(spec["lhs"])
            self.assertIs(ir.validate_spec(spec), spec)
            self.assertTrue(ir.extract_polynomial(spec)["zero"], blade)

    def test_validation_rejects_malformed_and_zero_nodes(self) -> None:
        bad_nodes = [
            fixed(-1),
            fixed(16),
            {"fixed_blade": {"blade": True, "coefficient": 1}},
            {"fixed_blade": {"blade": 15, "coefficient": False}},
            fixed(15, 0),
            {"fixed_blade": {"blade": 15}},
            {"fixed_blade": {"blade": 15, "coefficient": 1, "extra": 0}},
            {"fixed_blade": {"blade": 15, "coefficient": 1}, "extra": 0},
            {"fixed_blade": 15},
        ]
        for node in bad_nodes:
            spec = self.base_spec()
            spec["lhs"] = node
            with self.subTest(node=node):
                with self.assertRaises((ir.DiscoveryError, ValueError)):
                    ir.validate_spec(spec)

    def test_exact_pseudoscalar_square(self) -> None:
        spec = self.base_spec()
        spec["lhs"] = gp(fixed(15), fixed(15))
        spec["rhs"] = {"scalar": 1}
        polynomial = ir.extract_polynomial(spec)
        self.assertTrue(polynomial["zero"])
        self.assertEqual(polynomial["total_terms"], 0)

    def test_exact_dual_grade_complements(self) -> None:
        for variable, target_grade in (("v", 3), ("B", 2), ("T", 1)):
            dual = gp({"var": variable}, fixed(15))
            spec = self.base_spec()
            spec["lhs"] = dual
            spec["rhs"] = grade(target_grade, copy.deepcopy(dual))
            self.assertTrue(ir.extract_polynomial(spec)["zero"], variable)

    def test_python_dual_round_trip_for_declared_grades(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, variable in enumerate(("v", "B", "T")):
                spec = self.base_spec()
                spec["name"] = f"dual_round_trip_{variable}"
                spec["lhs"] = gp(gp({"var": variable}, fixed(15)), fixed(15))
                spec["rhs"] = {"var": variable}
                identity = ir.load_identity(
                    self.write_spec(spec, directory, f"round_trip_{index}.json")
                )
                for assignment in range(64):
                    self.assertTrue(
                        ir.evaluate_identity(identity, assignment)[0],
                        (variable, assignment),
                    )

    def test_compiler_support_is_exactly_one_blade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identity = ir.load_identity(
                self.write_spec(self.base_spec(), directory, "support.json")
            )
            constants = [node for node in identity.nodes if node.op == "fixed_blade"]
            self.assertEqual(len(constants), 1)
            self.assertEqual(constants[0].grade, 15)
            self.assertEqual(constants[0].support_mask, 1 << 15)

    def test_generated_header_places_constant_on_declared_blade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_spec(self.base_spec(), directory, "emission.json")
            identity = ir.load_identity(path)
            header = ir.emit_header([identity], ["emission.json"])
        self.assertEqual(header.count("nodes[0].c[15] = 1;"), 1)
        self.assertNotIn("nodes[0].c[0] = 1;", header)

    def test_compiler_rejects_zero_modulo_prime_coefficient(self) -> None:
        spec = self.base_spec()
        spec["lhs"] = fixed(15, spec["prime"])
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_spec(spec, Path(temporary), "zero_mod_prime.json")
            with self.assertRaises(ir.IdentityError):
                ir.load_identity(path)

    def test_install_is_idempotent(self) -> None:
        validate_hook = ir.exact._validate_expression
        parse_hook = ir.compiler.Builder.parse
        ir.install()
        self.assertIs(ir.exact._validate_expression, validate_hook)
        self.assertIs(ir.compiler.Builder.parse, parse_hook)


if __name__ == "__main__":
    unittest.main()
