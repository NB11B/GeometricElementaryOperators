from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import tools.geo_identity_v4_1_ir as ir


class FixedBladeTests(unittest.TestCase):
    def base_spec(self) -> dict:
        return {
            "schema_version": 1,
            "name": "fixed_blade_test",
            "description": "fixed blade test",
            "expected": "identity",
            "dimension": 4,
            "signature": [1, 1, -1, -1],
            "prime": 65521,
            "coefficient_bound": 3,
            "seed": 123,
            "variables": [{"name": "v", "grades": [1]}],
            "lhs": {"fixed_blade": {"blade": 15, "coefficient": 1}},
            "rhs": {"fixed_blade": {"blade": 15, "coefficient": 1}},
        }

    def test_validation_accepts_representative_blades(self) -> None:
        for blade in (0, 1, 3, 15):
            spec = self.base_spec()
            spec["lhs"] = {"fixed_blade": {"blade": blade, "coefficient": -2}}
            spec["rhs"] = copy.deepcopy(spec["lhs"])
            ir.validate_spec(spec)

    def test_validation_rejects_malformed_nodes(self) -> None:
        bad_nodes = [
            {"fixed_blade": {"blade": -1, "coefficient": 1}},
            {"fixed_blade": {"blade": 16, "coefficient": 1}},
            {"fixed_blade": {"blade": True, "coefficient": 1}},
            {"fixed_blade": {"blade": 15, "coefficient": False}},
            {"fixed_blade": {"blade": 15}},
            {"fixed_blade": {"blade": 15, "coefficient": 1, "extra": 0}},
            {"fixed_blade": {"blade": 15, "coefficient": 1}, "extra": 0},
        ]
        for node in bad_nodes:
            spec = self.base_spec()
            spec["lhs"] = node
            with self.assertRaises((ir.DiscoveryError, ValueError)):
                ir.validate_spec(spec)

    def test_exact_pseudoscalar_square(self) -> None:
        spec = self.base_spec()
        pseudoscalar = {"fixed_blade": {"blade": 15, "coefficient": 1}}
        spec["lhs"] = {"op": "gp", "args": [pseudoscalar, pseudoscalar]}
        spec["rhs"] = {"scalar": 1}
        self.assertTrue(ir.extract_polynomial(spec)["zero"])

    def test_python_finite_field_evaluator(self) -> None:
        spec = self.base_spec()
        pseudoscalar = {"fixed_blade": {"blade": 15, "coefficient": 1}}
        spec["lhs"] = {"op": "gp", "args": [{"var": "v"}, pseudoscalar]}
        spec["rhs"] = copy.deepcopy(spec["lhs"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            identity = ir.load_identity(path)
            for assignment in range(32):
                self.assertTrue(ir.evaluate_identity(identity, assignment)[0])

    def test_generated_header_places_constant_on_declared_blade(self) -> None:
        spec = self.base_spec()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            identity = ir.load_identity(path)
            header = ir.emit_header([identity], ["identity.json"])
        self.assertIn(".c[15] = 1;", header)


if __name__ == "__main__":
    unittest.main()
