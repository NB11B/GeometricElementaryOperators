#!/usr/bin/env python3
"""Tests for fixed and generated identity-result validation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER_PATH = (
    ROOT
    / "benchmarks"
    / "geo_identity_search"
    / "scripts"
    / "summarize_identity_results.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUMMARIZER = load_module("summarize_identity_results", SUMMARIZER_PATH)


class IdentityResultSummarizerTests(unittest.TestCase):
    @staticmethod
    def row(name: str, expected: str, found: bool):
        return SUMMARIZER.Row(
            identity=name,
            expected=expected,
            dimension=4,
            signature="+1,+1,+1,+1",
            prime=65521,
            variables=2,
            nodes=4,
            assignments=1024,
            cpu_checks=16,
            kernel_us=1.0,
            assignments_per_second=1_024_000_000.0,
            found_counterexample=found,
            witness_assignment=0 if found else None,
            witness_blade=3 if found else None,
            witness_lhs=-1 if found else None,
            witness_rhs=1 if found else None,
            result="pass",
        )

    def test_generated_discovery_names_are_detected(self) -> None:
        rows = [
            self.row("source__original__p65521", "identity", False),
            self.row(
                "source__m001_root_sign_flip_deadbeef__p65521",
                "counterexample",
                True,
            ),
        ]
        self.assertTrue(SUMMARIZER.is_generated_dynamic_corpus(rows))
        self.assertEqual(
            SUMMARIZER.validate(rows, allow_dynamic_corpus=True),
            [],
        )

    def test_fixed_validator_still_rejects_generated_names(self) -> None:
        rows = [self.row("source__original__p65521", "identity", False)]
        failures = SUMMARIZER.validate(rows, allow_dynamic_corpus=False)
        self.assertTrue(any("missing identities" in failure for failure in failures))
        self.assertTrue(any("unexpected identities" in failure for failure in failures))

    def test_dynamic_validator_enforces_declared_expectation(self) -> None:
        rows = [self.row("source__mutation__p65521", "identity", True)]
        failures = SUMMARIZER.validate(rows, allow_dynamic_corpus=True)
        self.assertTrue(any("true identity produced a witness" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
