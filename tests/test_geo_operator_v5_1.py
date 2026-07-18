from __future__ import annotations

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

import geo_operator_v5_1 as operator
import verify_geo_operator_certificate as verifier


class OperatorV51Tests(unittest.TestCase):
    def test_fixed_specialization_matches_generic_all_dimension_four_signatures(self) -> None:
        for _, signature in operator.signatures(4):
            for side in ("left", "right"):
                for blade in range(16):
                    rows = operator.sparse_plan(4, signature, [{"blade": blade, "coefficient": 1}], side)
                    value = [index - 7 for index in range(16)]
                    constant = [0] * 16
                    constant[blade] = 1
                    generic = operator.generic_gp(constant, value, signature) if side == "left" else operator.generic_gp(value, constant, signature)
                    self.assertEqual(operator.apply_sparse(value, rows), generic)

    def test_sparse_operator_and_matrix(self) -> None:
        signature = [1, 1, -1, -1]
        terms = [{"blade": 0, "coefficient": 1}, {"blade": 3, "coefficient": -2}, {"blade": 15, "coefficient": 1}]
        rows = operator.sparse_plan(4, signature, terms, "right")
        matrix = operator.dense_matrix(rows)
        features = operator.structural_features(matrix)
        self.assertEqual(features["size"], 16)
        self.assertEqual(features["maximum_row_nonzero"], 3)
        self.assertFalse(features["monomial"])

    def test_pseudoscalar_is_monomial(self) -> None:
        for dimension in range(2, 7):
            for _, signature in operator.signatures(dimension):
                rows = operator.sparse_plan(dimension, signature, [{"blade": (1 << dimension) - 1, "coefficient": 1}], "right")
                self.assertTrue(operator.structural_features(operator.dense_matrix(rows))["monomial"])

    def test_certificate_independent_verifier(self) -> None:
        signature = [1, 1, -1, -1]
        terms = [{"blade": 0, "coefficient": 1}, {"blade": 15, "coefficient": -1}]
        rows = operator.sparse_plan(4, signature, terms, "right")
        certificate = operator.certificate_payload(4, "cl22", signature, "right", terms, rows)
        operator.verify_certificate(certificate)
        verifier.verify(certificate)
        damaged = json.loads(json.dumps(certificate))
        damaged["rows"][0][0]["factor"] *= -1
        with self.assertRaises(ValueError):
            verifier.verify(damaged)

    def test_full_pipeline_smoke(self) -> None:
        config = {
            "dimensions": [2, 3],
            "all_fixed_blades": True,
            "specialization_iterations": 2,
            "sparse_iterations": 2,
            "sparse_terms": [
                {"blade": 0, "coefficient": 1},
                {"blade": 1, "coefficient": -2},
                {"blade": -1, "coefficient": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = operator.run(config_path, root / "evidence", 501)
            self.assertEqual(report["validation"], "PASS")
            self.assertGreater(report["certificate_count"], 0)
            self.assertTrue((root / "evidence" / "geo_operator_plans_v5_1.h").is_file())


if __name__ == "__main__":
    unittest.main()
