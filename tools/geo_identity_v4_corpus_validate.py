#!/usr/bin/env python3
"""Validate a generated V4 audited corpus before physical CUDA execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_TOOL = ROOT / "tools" / "geo_identity_discovery.py"


def load_discovery():
    spec = importlib.util.spec_from_file_location("geo_identity_discovery", DISCOVERY_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {DISCOVERY_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DISCOVERY = load_discovery()


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
    relation_primes: dict[tuple[str, str], set[int]] = {}
    polynomial_status: Counter[str] = Counter()

    for index, row in enumerate(statements):
        if not isinstance(row, dict):
            raise ValueError(f"statement row {index} is not an object")
        name = str(row.get("name", ""))
        relative_path = str(row.get("path", ""))
        expected = str(row.get("expected", ""))
        variant = str(row.get("variant", ""))
        relation_id = str(row.get("relation_id", ""))
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
        relation_primes.setdefault((relation_id, variant), set()).add(prime)

        spec_path = root / relative_path
        if not spec_path.is_file():
            raise ValueError(f"missing statement file: {spec_path}")
        spec = load_json(spec_path)
        DISCOVERY.validate_spec(spec)
        polynomial = DISCOVERY.extract_polynomial(spec)
        is_zero = bool(polynomial.get("zero"))
        if expected == "identity" and not is_zero:
            raise ValueError(f"identity statement has nonzero polynomial: {name}")
        if expected == "counterexample" and is_zero:
            raise ValueError(f"control statement has zero polynomial: {name}")
        polynomial_status["zero" if is_zero else "nonzero"] += 1

    expected_primes = {int(value) for value in manifest.get("primes", [])}
    if not expected_primes:
        raise ValueError("manifest primes must be non-empty")
    for key, observed in sorted(relation_primes.items()):
        if observed != expected_primes:
            raise ValueError(
                f"prime matrix incomplete for relation/variant {key}: "
                f"expected {sorted(expected_primes)}, observed {sorted(observed)}"
            )

    selected = int(manifest.get("selected_relation_count", -1))
    controls = int(manifest.get("control_count", -1))
    expected_statement_count = (selected + controls) * len(expected_primes)
    if expected_statement_count != len(statements):
        raise ValueError(
            f"statement count mismatch: expected {expected_statement_count}, "
            f"observed {len(statements)}"
        )
    if int(manifest.get("statement_count", -1)) != len(statements):
        raise ValueError("manifest statement_count does not match statements")

    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_corpus_validator",
        "statement_count": len(statements),
        "identity_count": counts["identity"],
        "control_count": counts["counterexample"],
        "zero_polynomial_count": polynomial_status["zero"],
        "nonzero_polynomial_count": polynomial_status["nonzero"],
        "relation_count": selected,
        "control_variant_count": controls,
        "prime_count": len(expected_primes),
        "primes": sorted(expected_primes),
        "validation": "PASS",
    }


def markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V4 audited corpus validation",
            "",
            f"- statements: {report['statement_count']}",
            f"- identity rows: {report['identity_count']}",
            f"- control rows: {report['control_count']}",
            f"- zero-polynomial rows: {report['zero_polynomial_count']}",
            f"- nonzero-polynomial rows: {report['nonzero_polynomial_count']}",
            f"- selected relations: {report['relation_count']}",
            f"- control variants: {report['control_variant_count']}",
            f"- primes: `{report['primes']}`",
            f"- validation: **{report['validation']}**",
            "",
            "Each identity row has an exact zero difference polynomial. Each control row",
            "has an exact nonzero difference polynomial. Every relation/variant is present",
            "for the complete declared prime matrix.",
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
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        DISCOVERY.DiscoveryError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_CORPUS_VALIDATION: PASS "
        f"statements={report['statement_count']} "
        f"identities={report['identity_count']} "
        f"controls={report['control_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
