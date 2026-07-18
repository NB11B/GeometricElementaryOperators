#!/usr/bin/env python3
"""Tests for relation normalization, quotienting, and novelty ranking."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "geo_identity_relation_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("geo_identity_relation_audit", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module()


def certificate(lhs, rhs, lhs_scale=1, rhs_scale=1, grades=None):
    grades = grades or {"a": [1], "b": [1]}
    variables = [{"name": name, "grades": value} for name, value in grades.items()]
    return {
        "schema_version": 1,
        "relation_id": "test",
        "grammar": "test",
        "lhs_scale": lhs_scale,
        "rhs_scale": rhs_scale,
        "lhs": {"expression": lhs, "label": "lhs"},
        "rhs": None if rhs is None else {"expression": rhs, "label": "rhs"},
        "scope": {"dimension": 4, "signature": [1, 1, 1, 1], "variables": variables},
        "score": 1,
    }


class RelationAuditTests(unittest.TestCase):
    def test_first_occurrence_variable_renaming(self):
        expr = {"op": "gp", "args": [{"var": "z"}, {"var": "x"}]}
        renamed, mapping = AUDIT.rename_variables(expr)
        self.assertEqual(renamed, {"op": "gp", "args": [{"var": "v0"}, {"var": "v1"}]})
        self.assertEqual(mapping, {"z": "v0", "x": "v1"})

    def test_double_reverse_and_grade_support_simplify(self):
        support = {"a": (1,)}
        double_reverse = {"op": "reverse", "arg": {"op": "reverse", "arg": {"var": "a"}}}
        projected = {"op": "grade", "grade": 1, "arg": {"var": "a"}}
        self.assertEqual(AUDIT.simplify(double_reverse, support), {"var": "a"})
        self.assertEqual(AUDIT.simplify(projected, support), {"var": "a"})

    def test_side_swap_and_simultaneous_sign_share_family(self):
        left = {"op": "commutator", "args": [{"var": "a"}, {"var": "b"}]}
        right = {
            "op": "scale",
            "value": 2,
            "arg": {"op": "wedge", "args": [{"var": "a"}, {"var": "b"}]},
        }
        first = AUDIT.normalize_relation(certificate(left, right))
        second = AUDIT.normalize_relation(certificate(right, left, -1, -1))
        self.assertEqual(first["family_key"], second["family_key"])

    def test_commutator_wedge_is_known(self):
        cert = certificate(
            {"op": "commutator", "args": [{"var": "a"}, {"var": "b"}]},
            {
                "op": "scale",
                "value": 2,
                "arg": {"op": "wedge", "args": [{"var": "a"}, {"var": "b"}]},
            },
        )
        cert["lhs"]["label"] = "[a,b]"
        cert["rhs"]["label"] = "2*((a ^ b))"
        family, status = AUDIT.identify_known_family(cert, AUDIT.normalize_relation(cert))
        self.assertEqual((family, status), ("vector_commutator_wedge", "known"))

    def test_projection_idempotence_is_tautological(self):
        cert = certificate(
            {
                "op": "grade",
                "grade": 0,
                "arg": {"op": "gp", "args": [{"var": "a"}, {"var": "b"}]},
            },
            {
                "op": "grade",
                "grade": 0,
                "arg": {
                    "op": "grade",
                    "grade": 0,
                    "arg": {"op": "gp", "args": [{"var": "b"}, {"var": "a"}]},
                },
            },
        )
        cert["lhs"]["label"] = "<(a * b)>_0"
        cert["rhs"]["label"] = "<<(b * a)>_0>_0"
        family, status = AUDIT.identify_known_family(cert, AUDIT.normalize_relation(cert))
        self.assertEqual((family, status), ("projection_idempotence", "tautological"))

    def test_audit_groups_equivalent_presentations(self):
        base = certificate(
            {"op": "gp", "args": [{"var": "a"}, {"var": "b"}]},
            {"op": "gp", "args": [{"var": "a"}, {"var": "b"}]},
        )
        base["relation_id"] = "one"

        negated_product = {
            "op": "neg",
            "arg": {"op": "gp", "args": [{"var": "x"}, {"var": "y"}]},
        }
        other = certificate(
            negated_product,
            negated_product,
            grades={"x": [1], "y": [1]},
        )
        other["relation_id"] = "two"

        report = AUDIT.audit([base, other])
        self.assertEqual(report["certificate_count"], 2)
        self.assertEqual(report["normalized_family_count"], 1)
        self.assertEqual(report["families"][0]["member_count"], 2)


if __name__ == "__main__":
    unittest.main()
