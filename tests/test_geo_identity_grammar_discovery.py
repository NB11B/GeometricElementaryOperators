#!/usr/bin/env python3
"""Tests for grammar-bounded exact geometric identity discovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_ROOT = ROOT / "experiments" / "geometric_identity_engine_v3" / "grammars"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GRAMMAR = load_module(
    "geo_identity_grammar_discovery",
    ROOT / "tools" / "geo_identity_grammar_discovery.py",
)
COMPILER = load_module(
    "geo_identity_compiler_for_v3",
    ROOT / "tools" / "geo_identity_compiler.py",
)


def write_small_grammar(path: Path) -> None:
    data = {
        "schema_version": 1,
        "name": "small_vector_grammar",
        "description": "small deterministic test grammar",
        "dimension": 3,
        "signature": [1, 1, 1],
        "prime": 65521,
        "coefficient_bound": 2,
        "seed": 123456789,
        "variables": [
            {"name": "a", "grades": [1]},
            {"name": "b", "grades": [1]},
        ],
        "grammar": {
            "max_cost": 4,
            "max_expressions": 240,
            "max_per_cost": 90,
            "max_representatives_per_polynomial": 5,
            "candidate_multiplier": 10,
            "operators": {
                "unary": ["neg", "reverse"],
                "binary": ["add", "sub", "gp", "wedge", "commutator"],
                "scales": [-2, 2],
                "grades": [0, 1, 2],
            },
            "constants": [0],
            "families": [
                {"kind": "vector_product_decomposition", "variables": ["a", "b"]},
                {"kind": "reverse_product", "variables": ["a", "b"]},
            ],
            "max_relations": 8,
            "max_controls": 2,
            "max_pairs_per_class": 16,
            "max_relations_per_primitive_class": 2,
            "min_relation_variables": 2,
            "require_geometric_operator": True,
            "term_limit": 50000,
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class GrammarDiscoveryTests(unittest.TestCase):
    def test_repository_grammars_load(self) -> None:
        paths = sorted(GRAMMAR_ROOT.glob("*.json"))
        self.assertEqual(len(paths), 3)
        self.assertEqual(
            [GRAMMAR.load_grammar(path).name for path in paths],
            [
                "vector_product_relations",
                "general_reversion_relations",
                "commutator_jacobi_family",
            ],
        )

    def test_vector_commutator_is_twice_wedge(self) -> None:
        grammar = GRAMMAR.load_grammar(
            GRAMMAR_ROOT / "01_vector_product_relations.json"
        )
        classifier = GRAMMAR.SymbolicClassifier(grammar)
        a, b = {"var": "a"}, {"var": "b"}
        commutator = classifier.classify({"op": "commutator", "args": [a, b]})
        wedge = classifier.classify({"op": "wedge", "args": [a, b]})
        self.assertEqual(commutator.primitive_hash, wedge.primitive_hash)
        self.assertEqual(GRAMMAR.relation_multipliers(commutator, wedge), (1, 2))

    def test_reversion_reverses_general_product(self) -> None:
        grammar = GRAMMAR.load_grammar(
            GRAMMAR_ROOT / "02_general_reversion_relations.json"
        )
        classifier = GRAMMAR.SymbolicClassifier(grammar)
        a, b = {"var": "a"}, {"var": "b"}
        left = classifier.classify(
            {"op": "reverse", "arg": {"op": "gp", "args": [a, b]}}
        )
        right = classifier.classify(
            {
                "op": "gp",
                "args": [
                    {"op": "reverse", "arg": b},
                    {"op": "reverse", "arg": a},
                ],
            }
        )
        self.assertEqual(left.exact_hash, right.exact_hash)
        self.assertEqual(GRAMMAR.relation_multipliers(left, right), (1, 1))

    def test_jacobi_family_has_zero_polynomial(self) -> None:
        grammar = GRAMMAR.load_grammar(
            GRAMMAR_ROOT / "03_commutator_jacobi.json"
        )
        classifier = GRAMMAR.SymbolicClassifier(grammar)
        jacobi = max(GRAMMAR.family_expressions(grammar), key=GRAMMAR.expression_cost)
        record = classifier.classify(jacobi)
        self.assertTrue(record.zero)
        self.assertEqual(record.total_terms, 0)

    def test_enumeration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            grammar_path = Path(temporary) / "grammar.json"
            write_small_grammar(grammar_path)
            grammar = GRAMMAR.load_grammar(grammar_path)
            first, first_stats = GRAMMAR.enumerate_expressions(grammar)
            second, second_stats = GRAMMAR.enumerate_expressions(grammar)
            self.assertEqual(first_stats, second_stats)
            self.assertEqual(
                [(record.cost, record.key, record.exact_hash) for record in first],
                [(record.cost, record.key, record.exact_hash) for record in second],
            )
            self.assertEqual(
                [relation.relation_id for relation in GRAMMAR.discover_relations(grammar, first)],
                [relation.relation_id for relation in GRAMMAR.discover_relations(grammar, second)],
            )

    def test_selected_relations_include_product_decomposition(self) -> None:
        grammar = GRAMMAR.load_grammar(
            GRAMMAR_ROOT / "01_vector_product_relations.json"
        )
        records, _ = GRAMMAR.enumerate_expressions(grammar)
        relations = GRAMMAR.discover_relations(grammar, records)
        self.assertTrue(
            any(
                "[a,b]" in relation.lhs.label
                and relation.rhs is not None
                and "a ^ b" in relation.rhs.label
                for relation in relations
            )
        )

    def test_build_corpus_is_deterministic_and_v1_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grammar_path = root / "grammar.json"
            write_small_grammar(grammar_path)

            def build(output: Path) -> bytes:
                arguments = argparse.Namespace(
                    grammar=[grammar_path],
                    output_dir=output,
                    prime=[65521, 65519],
                    precheck_assignments=128,
                    term_limit=50000,
                    max_relations=5,
                    max_controls=1,
                    clean=True,
                )
                self.assertEqual(GRAMMAR.build_corpus(arguments), 0)
                manifest_path = output / "corpus-manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertGreater(len(manifest["entries"]), 4)
                self.assertEqual(
                    {entry["expected"] for entry in manifest["entries"]},
                    {"identity", "counterexample"},
                )
                for entry in manifest["entries"]:
                    COMPILER.load_identity(output / entry["file"])
                return manifest_path.read_bytes()

            self.assertEqual(build(root / "first"), build(root / "second"))

    def test_controls_have_exact_precheck_witnesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            grammar_path = Path(temporary) / "grammar.json"
            write_small_grammar(grammar_path)
            grammar = GRAMMAR.load_grammar(grammar_path)
            records, _ = GRAMMAR.enumerate_expressions(grammar)
            controls = GRAMMAR.generate_controls(
                grammar,
                GRAMMAR.discover_relations(grammar, records),
                precheck_assignments=128,
                term_limit=50000,
            )
            self.assertTrue(controls)
            for control in controls:
                self.assertFalse(control.polynomial["zero"])
                self.assertGreaterEqual(control.witness_assignment, 0)
                self.assertIsNotNone(control.witness)


if __name__ == "__main__":
    unittest.main()
