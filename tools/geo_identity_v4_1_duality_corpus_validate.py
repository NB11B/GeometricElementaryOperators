#!/usr/bin/env python3
"""Validate the generated V4.1 duality corpus before host/CUDA execution."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import geo_identity_v4_1_exact as exact


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "corpus-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    statements = manifest.get("statements")
    if not isinstance(statements, list) or not statements:
        raise ValueError("manifest statements must be a non-empty list")

    names: set[str] = set()
    paths: set[str] = set()
    counts: Counter[str] = Counter()
    polynomial_counts: Counter[str] = Counter()
    variant_primes: dict[tuple[str, str], set[int]] = {}

    for index, row in enumerate(statements):
        if not isinstance(row, dict):
            raise ValueError(f"statement row {index} is not an object")
        name = str(row.get("name", ""))
        relative_path = str(row.get("path", ""))
        expected = str(row.get("expected", ""))
        relation_id = str(row.get("relation_id", ""))
        variant = str(row.get("variant", ""))
        prime = int(row.get("prime", 0))
        if not name or name in names:
            raise ValueError(f"invalid or duplicate statement name: {name!r}")
        if not relative_path or relative_path in paths:
            raise ValueError(f"invalid or duplicate statement path: {relative_path!r}")
        if expected not in {"identity", "counterexample"}:
            raise ValueError(f"invalid expected value for {name}: {expected!r}")
        names.add(name)
        paths.add(relative_path)
        counts[expected] += 1
        variant_primes.setdefault((relation_id, variant), set()).add(prime)

        specification_path = root / relative_path
        if not specification_path.is_file():
            raise ValueError(f"missing statement file: {specification_path}")
        specification = load_json(specification_path)
        exact.validate_spec(specification)
        polynomial = exact.extract_polynomial(specification)
        is_zero = bool(polynomial["zero"])
        if expected == "identity" and not is_zero:
            raise ValueError(f"identity statement has nonzero polynomial: {name}")
        if expected == "counterexample" and is_zero:
            raise ValueError(f"control statement has zero polynomial: {name}")
        if expected == "counterexample":
            witness_assignment, witness = exact.precheck(specification, 512)
            if witness_assignment is None or witness is None:
                raise ValueError(f"control has no deterministic witness: {name}")
        polynomial_counts["zero" if is_zero else "nonzero"] += 1

    expected_primes = {int(value) for value in manifest.get("primes", [])}
    if not expected_primes:
        raise ValueError("manifest primes must be non-empty")
    for key, observed in sorted(variant_primes.items()):
        if observed != expected_primes:
            raise ValueError(
                f"prime matrix incomplete for {key}: expected {sorted(expected_primes)}, "
                f"observed {sorted(observed)}"
            )

    identity_definitions = int(manifest.get("identity_definition_count", -1))
    control_definitions = int(manifest.get("control_definition_count", -1))
    expected_statement_count = (
        identity_definitions + control_definitions
    ) * len(expected_primes)
    if expected_statement_count != len(statements):
        raise ValueError(
            f"statement count mismatch: expected {expected_statement_count}, "
            f"observed {len(statements)}"
        )
    if int(manifest.get("statement_count", -1)) != len(statements):
        raise ValueError("manifest statement_count does not match statements")
    if int(manifest.get("identity_row_count", -1)) != counts["identity"]:
        raise ValueError("manifest identity_row_count mismatch")
    if int(manifest.get("control_row_count", -1)) != counts["counterexample"]:
        raise ValueError("manifest control_row_count mismatch")

    duality = manifest.get("duality")
    if duality != {
        "side": "right",
        "pseudoscalar_blade": 15,
        "pseudoscalar_coefficient": 1,
        "pseudoscalar_square": 1,
    }:
        raise ValueError("manifest duality contract does not match the V4.1 Cl(2,2) gate")

    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_duality_corpus_validator",
        "statement_count": len(statements),
        "identity_count": counts["identity"],
        "control_count": counts["counterexample"],
        "zero_polynomial_count": polynomial_counts["zero"],
        "nonzero_polynomial_count": polynomial_counts["nonzero"],
        "prime_count": len(expected_primes),
        "primes": sorted(expected_primes),
        "validation": "PASS",
    }


def markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V4.1 duality corpus validation",
            "",
            f"- statements: {report['statement_count']}",
            f"- identity rows: {report['identity_count']}",
            f"- control rows: {report['control_count']}",
            f"- zero-polynomial rows: {report['zero_polynomial_count']}",
            f"- nonzero-polynomial rows: {report['nonzero_polynomial_count']}",
            f"- primes: `{report['primes']}`",
            f"- validation: **{report['validation']}**",
            "",
            "Every identity is an exact zero polynomial under the fixed-blade contract.",
            "Every control is exact, nonzero, and has a deterministic finite-field witness.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate(args.corpus_root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, json.JSONDecodeError, exact.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_DUALITY_CORPUS_VALIDATION: PASS "
        f"statements={report['statement_count']} "
        f"identities={report['identity_count']} controls={report['control_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
