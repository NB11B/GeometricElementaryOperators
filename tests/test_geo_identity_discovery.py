#!/usr/bin/env python3
"""Tests for multi-prime geometric identity discovery and witness reduction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_PATH = ROOT / "tools" / "geo_identity_discovery.py"
COMPILER_PATH = ROOT / "tools" / "geo_identity_compiler.py"
CORPUS = ROOT / "experiments" / "geometric_identity_engine" / "corpus"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DISCOVERY = load_module("geo_identity_discovery", DISCOVERY_PATH)
COMPILER = load_module("geo_identity_compiler_for_discovery", COMPILER_PATH)


class GeometricIdentityDiscoveryTests(unittest.TestCase):
    def test_default_primes_are_prime(self) -> None:
        self.assertEqual(len(DISCOVERY.DEFAULT_PRIMES), 4)
        for prime in DISCOVERY.DEFAULT_PRIMES:
            self.assertTrue(DISCOVERY.is_prime(prime), prime)

    def test_polynomial_extraction_proves_known_corpus_identities(self) -> None:
        for path in sorted(CORPUS.glob("*.json")):
            spec = DISCOVERY.load_spec(path)
            polynomial = DISCOVERY.extract_polynomial(spec)
            if spec["expected"] == "identity":
                self.assertTrue(polynomial["zero"], path.name)
                self.assertEqual(polynomial["total_terms"], 0)
            else:
                self.assertFalse(polynomial["zero"], path.name)
                self.assertGreater(polynomial["total_terms"], 0)

    def test_polynomial_hash_is_prime_independent(self) -> None:
        source = DISCOVERY.load_spec(CORPUS / "02_reverse_product_order.json")
        first = DISCOVERY.extract_polynomial(source)
        source["prime"] = 65519
        second = DISCOVERY.extract_polynomial(source)
        self.assertEqual(first["canonical_hash"], second["canonical_hash"])
        self.assertEqual(first["zero"], second["zero"])

    def test_exact_evaluator_matches_v1_compiler(self) -> None:
        for path in sorted(CORPUS.glob("*.json")):
            spec = DISCOVERY.load_spec(path)
            compiled = COMPILER.load_identity(path)
            for assignment in range(32):
                mismatch = DISCOVERY.evaluate_assignment(spec, assignment)
                equal, blade, lhs, rhs = COMPILER.evaluate_identity(
                    compiled, assignment
                )
                if mismatch is None:
                    self.assertTrue(equal, (path.name, assignment, blade, lhs, rhs))
                else:
                    self.assertFalse(equal, (path.name, assignment))
                    self.assertEqual(mismatch.blade, blade)
                    expected_lhs = lhs - compiled.prime if lhs > compiled.prime // 2 else lhs
                    expected_rhs = rhs - compiled.prime if rhs > compiled.prime // 2 else rhs
                    self.assertEqual(mismatch.lhs, expected_lhs)
                    self.assertEqual(mismatch.rhs, expected_rhs)

    def test_mutations_are_deterministic_and_unique(self) -> None:
        spec = DISCOVERY.load_spec(CORPUS / "02_reverse_product_order.json")
        first = DISCOVERY.generate_mutations(spec, 12)
        second = DISCOVERY.generate_mutations(spec, 12)
        self.assertEqual(
            [(mutation.stable_id, mutation.kind, mutation.path) for mutation, _ in first],
            [(mutation.stable_id, mutation.kind, mutation.path) for mutation, _ in second],
        )
        self.assertEqual(
            len({DISCOVERY.canonical_json({"lhs": value["lhs"], "rhs": value["rhs"]}) for _, value in first}),
            len(first),
        )

    def test_generated_corpus_is_deterministic_and_v1_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = argparse.Namespace(
                source=[
                    CORPUS / "01_vector_square_scalar.json",
                    CORPUS / "04_vector_product_commutativity_false.json",
                ],
                output_dir=root / "first",
                prime=[65521, 65519],
                precheck_assignments=32,
                max_mutations=4,
                term_limit=200_000,
                mutate_counterexamples=False,
                clean=True,
            )
            self.assertEqual(DISCOVERY.build_corpus(arguments), 0)
            first_manifest = (arguments.output_dir / "corpus-manifest.json").read_bytes()
            arguments.output_dir = root / "second"
            self.assertEqual(DISCOVERY.build_corpus(arguments), 0)
            second_manifest = (arguments.output_dir / "corpus-manifest.json").read_bytes()
            self.assertEqual(first_manifest, second_manifest)
            generated = sorted((arguments.output_dir / "corpus").glob("*.json"))
            self.assertGreater(len(generated), 4)
            for path in generated:
                COMPILER.load_identity(path)

    def test_nonzero_polynomial_mutations_find_small_witnesses(self) -> None:
        spec = DISCOVERY.load_spec(CORPUS / "01_vector_square_scalar.json")
        mutations = DISCOVERY.generate_mutations(spec, 8)
        nonzero = []
        for mutation, candidate in mutations:
            polynomial = DISCOVERY.extract_polynomial(candidate)
            if polynomial["zero"]:
                continue
            nonzero.append(mutation.stable_id)
            candidate["prime"] = 65521
            assignment, mismatch = DISCOVERY.precheck(candidate, 128)
            self.assertIsNotNone(assignment, mutation.kind)
            self.assertIsNotNone(mismatch, mutation.kind)
        self.assertTrue(nonzero)

    def test_witness_minimization_preserves_exact_failure(self) -> None:
        spec = DISCOVERY.load_spec(
            CORPUS / "04_vector_product_commutativity_false.json"
        )
        initial_variables = DISCOVERY.variables_to_signed(
            DISCOVERY.generate_assignment(spec, 0), spec["prime"]
        )
        initial_score = DISCOVERY.witness_score(initial_variables)
        reduced, mismatch, reduced_score = DISCOVERY.minimize_witness(spec, 0)
        self.assertIsNotNone(mismatch)
        self.assertLessEqual(reduced_score, initial_score)
        self.assertIsNotNone(
            DISCOVERY.evaluate_with_variables(
                spec, DISCOVERY.variables_to_mod(reduced, spec["prime"])
            )
        )

    def test_result_reducer_reproduces_and_reduces_witness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_arguments = argparse.Namespace(
                source=[CORPUS / "04_vector_product_commutativity_false.json"],
                output_dir=root / "generated",
                prime=[65521],
                precheck_assignments=16,
                max_mutations=0,
                term_limit=200_000,
                mutate_counterexamples=False,
                clean=True,
            )
            DISCOVERY.build_corpus(build_arguments)
            manifest = json.loads(
                (build_arguments.output_dir / "corpus-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            entry = manifest["entries"][0]
            witness = entry["precheck_witness"]
            csv_path = root / "results.csv"
            fieldnames = [
                "identity",
                "expected",
                "dimension",
                "signature",
                "prime",
                "variables",
                "nodes",
                "assignments",
                "cpu_checks",
                "kernel_us",
                "assignments_per_second",
                "found_counterexample",
                "witness_assignment",
                "witness_blade",
                "witness_lhs",
                "witness_rhs",
                "result",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "identity": entry["name"],
                        "expected": "counterexample",
                        "dimension": 4,
                        "signature": "+1,+1,+1,+1",
                        "prime": 65521,
                        "variables": 2,
                        "nodes": 4,
                        "assignments": 1024,
                        "cpu_checks": 16,
                        "kernel_us": 1.0,
                        "assignments_per_second": 1_000_000_000.0,
                        "found_counterexample": "true",
                        "witness_assignment": witness["assignment"],
                        "witness_blade": witness["blade"],
                        "witness_lhs": witness["lhs"],
                        "witness_rhs": witness["rhs"],
                        "result": "pass",
                    }
                )
            reduce_arguments = argparse.Namespace(
                manifest=build_arguments.output_dir / "corpus-manifest.json",
                csv=csv_path,
                output_json=root / "report.json",
                markdown_out=root / "report.md",
            )
            self.assertEqual(DISCOVERY.reduce_results(reduce_arguments), 0)
            report = json.loads(reduce_arguments.output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["validation"], "pass")
            reduced = report["records"][0]["reduced_witness"]
            self.assertGreater(reduced["score"]["nonzero_coefficients"], 0)


if __name__ == "__main__":
    unittest.main()
