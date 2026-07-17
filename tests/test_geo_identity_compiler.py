#!/usr/bin/env python3
"""Tests for the exact geometric identity compiler."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPILER_PATH = ROOT / "tools" / "geo_identity_compiler.py"
CORPUS_DIRECTORY = ROOT / "experiments" / "geometric_identity_engine" / "corpus"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPILER = load_module("geo_identity_compiler", COMPILER_PATH)


class GeometricIdentityCompilerTests(unittest.TestCase):
    @classmethod
    def identity_paths(cls) -> list[Path]:
        return sorted(CORPUS_DIRECTORY.glob("*.json"))

    def test_corpus_loads_with_unique_names(self) -> None:
        identities = COMPILER.load_corpus(self.identity_paths())
        self.assertEqual(len(identities), 5)
        self.assertEqual(len({identity.name for identity in identities}), 5)

    def test_known_identities_hold_in_exact_python_backend(self) -> None:
        for identity in COMPILER.load_corpus(self.identity_paths()):
            if identity.expected != "identity":
                continue
            for assignment in range(128):
                equal, blade, lhs, rhs = COMPILER.evaluate_identity(
                    identity, assignment
                )
                self.assertTrue(
                    equal,
                    (identity.name, assignment, blade, lhs, rhs),
                )

    def test_mutated_identities_produce_small_counterexamples(self) -> None:
        for identity in COMPILER.load_corpus(self.identity_paths()):
            if identity.expected != "counterexample":
                continue
            witness = None
            for assignment in range(128):
                result = COMPILER.evaluate_identity(identity, assignment)
                if not result[0]:
                    witness = (assignment, *result[1:])
                    break
            self.assertIsNotNone(witness, identity.name)

    def test_common_subexpression_elimination_reuses_vector_square(self) -> None:
        identity = COMPILER.load_identity(
            CORPUS_DIRECTORY / "01_vector_square_scalar.json"
        )
        geometric_products = [
            node for node in identity.nodes if node.op == "gp"
        ]
        self.assertEqual(len(geometric_products), 1)
        self.assertEqual(identity.nodes[identity.lhs].op, "gp")

    def test_emission_is_deterministic(self) -> None:
        paths = self.identity_paths()
        identities = COMPILER.load_corpus(paths)
        sources = [path.relative_to(ROOT).as_posix() for path in paths]
        first = COMPILER.emit_header(identities, sources)
        second = COMPILER.emit_header(identities, sources)
        self.assertEqual(first, second)
        self.assertIn("GEO_IDENTITY_FOR_EACH", first)
        self.assertIn("vector_square_is_scalar", first)

    def test_cli_generate_check_and_python_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "geo_identity_corpus.cuh"
            base = [sys.executable, str(COMPILER_PATH)]
            for path in self.identity_paths():
                base.extend(["--identity", str(path.relative_to(ROOT))])

            generated = subprocess.run(
                [
                    *base,
                    "--output",
                    str(output),
                    "--python-checks",
                    "64",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)

            checked = subprocess.run(
                [
                    *base,
                    "--output",
                    str(output),
                    "--check",
                    "--python-checks",
                    "64",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("PASS:", checked.stdout)

    def test_invalid_signature_is_rejected(self) -> None:
        invalid = {
            "schema_version": 1,
            "name": "invalid_signature",
            "expected": "identity",
            "dimension": 2,
            "signature": [1, 0],
            "variables": [{"name": "a", "grades": [1]}],
            "lhs": {"var": "a"},
            "rhs": {"var": "a"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(COMPILER.IdentityError):
                COMPILER.load_identity(path)

    def test_grade_outside_dimension_is_rejected(self) -> None:
        invalid = {
            "schema_version": 1,
            "name": "invalid_grade",
            "expected": "identity",
            "dimension": 2,
            "signature": [1, 1],
            "variables": [{"name": "a", "grades": [3]}],
            "lhs": {"var": "a"},
            "rhs": {"var": "a"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(COMPILER.IdentityError):
                COMPILER.load_identity(path)


if __name__ == "__main__":
    unittest.main()
