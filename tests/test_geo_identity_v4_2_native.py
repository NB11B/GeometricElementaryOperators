#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) in sys.path:
    sys.path.remove(str(TOOLS))
sys.path.insert(0, str(TOOLS))

import geo_identity_compiler as legacy_compiler
import geo_identity_v4_2_corpus as corpus
import geo_identity_v4_2_engine as engine


class NativeV42Tests(unittest.TestCase):
    def variables(self) -> list[dict]:
        return [
            {"name": "v", "grades": [1]},
            {"name": "B", "grades": [2]},
            {"name": "T", "grades": [3]},
            {"name": "E", "grades": [0, 2, 4]},
            {"name": "O", "grades": [1, 3]},
            {"name": "M", "grades": [0, 1, 2, 3, 4]},
        ]

    def spec(self, signature: list[int], lhs: dict, rhs: dict, expected: str = "identity") -> dict:
        return {
            "schema_version": 1,
            "name": "v4_2_test",
            "description": "native V4.2 fixture",
            "expected": expected,
            "dimension": 4,
            "signature": signature,
            "prime": 65521,
            "coefficient_bound": 3,
            "seed": 123456789,
            "variables": self.variables(),
            "lhs": lhs,
            "rhs": rhs,
        }

    def write_spec(self, directory: Path, spec: dict, name: str = "identity.json") -> Path:
        path = directory / name
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        return path

    def test_pseudoscalar_square_signature_matrix(self) -> None:
        matrix = [
            ([1, 1, 1, 1], 1),
            ([1, 1, 1, -1], -1),
            ([1, 1, -1, -1], 1),
            ([1, -1, -1, -1], -1),
            ([-1, -1, -1, -1], 1),
        ]
        for signature, expected in matrix:
            with self.subTest(signature=signature):
                self.assertEqual(engine.pseudoscalar_square(signature), expected)
                I = engine.pseudoscalar(signature)
                spec = self.spec(signature, engine.gp(I, I), engine.scalar(expected))
                self.assertTrue(engine.extract_polynomial(spec)["zero"])

    def test_dual_square_and_round_trip_for_mixed_supports(self) -> None:
        for signature in ([1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, -1]):
            square = engine.pseudoscalar_square(signature)
            for name in ("v", "B", "T", "E", "O", "M"):
                value = engine.variable(name)
                dual = engine.right_dual(value, signature)
                square_spec = self.spec(signature, engine.right_dual(dual, signature), engine.scale(square, value))
                round_trip = self.spec(signature, engine.right_undual(dual, signature), value)
                self.assertTrue(engine.extract_polynomial(square_spec)["zero"], (signature, name, "square"))
                self.assertTrue(engine.extract_polynomial(round_trip)["zero"], (signature, name, "round_trip"))

    def test_native_compiler_owns_fixed_blade_node(self) -> None:
        signature = [1, 1, -1, -1]
        spec = self.spec(signature, engine.pseudoscalar(signature), engine.pseudoscalar(signature))
        legacy_parse = legacy_compiler.Builder.parse
        with tempfile.TemporaryDirectory() as temporary:
            identity = engine.load_identity(self.write_spec(Path(temporary), spec))
            fixed_nodes = [node for node in identity.nodes if node.op == "fixed_blade"]
            self.assertEqual(len(fixed_nodes), 1)
            self.assertEqual(fixed_nodes[0].support_mask, 1 << 15)
            header = engine.emit_header([identity], ["identity.json"])
        self.assertIs(legacy_compiler.Builder.parse, legacy_parse)
        self.assertIn("nodes[0].c[15] = 1;", header)

    def test_fixed_blade_lowering_is_signed_permutation(self) -> None:
        signature = [1, 1, 1, -1]
        for side in ("left", "right"):
            plan = engine.fixed_blade_permutation(4, signature, 15, engine.pseudoscalar_square(signature), side)
            self.assertEqual(len(plan), 16)
            self.assertEqual({row["source"] for row in plan}, set(range(16)))
            self.assertEqual({row["target"] for row in plan}, set(range(16)))
            self.assertTrue(all(row["factor"] in (-1, 1) for row in plan))

    def test_contraction_duality_discovery_has_exact_hits(self) -> None:
        config_path = ROOT / "experiments" / "geometric_identity_engine_v4_2" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        relations, controls = corpus.build_relations(config, [1, 1, -1, -1])
        self.assertGreaterEqual(len(relations), 20)
        self.assertEqual(len(controls), 4)
        discovered = [item for item in relations if item.get("classification") == "discovered-sign"]
        self.assertGreaterEqual(len(discovered), 1)
        self.assertTrue(all(item.get("sign") in (-1, 1) for item in discovered))

    def test_corpus_build_is_deterministic(self) -> None:
        config_path = ROOT / "experiments" / "geometric_identity_engine_v4_2" / "config.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = corpus.build(config_path, root / "first", max_relations_per_signature=8)
            second = corpus.build(config_path, root / "second", max_relations_per_signature=8)
            first_rows = [{key: value for key, value in row.items() if key != "path"} for row in first["statements"]]
            second_rows = [{key: value for key, value in row.items() if key != "path"} for row in second["statements"]]
            self.assertEqual(first_rows, second_rows)

    def test_wrong_signature_sign_is_nonzero(self) -> None:
        signature = [1, 1, 1, -1]
        I = engine.pseudoscalar(signature)
        spec = self.spec(signature, engine.gp(I, I), engine.scalar(1), expected="counterexample")
        self.assertFalse(engine.extract_polynomial(spec)["zero"])


if __name__ == "__main__":
    unittest.main()
