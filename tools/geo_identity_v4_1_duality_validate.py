#!/usr/bin/env python3
"""Validate a generated V4.1 fixed-blade duality corpus before host/CUDA use."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import geo_identity_v4_1_ir as ir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads((args.corpus_root / "corpus-manifest.json").read_text(encoding="utf-8"))
        rows = manifest.get("statements")
        if not isinstance(rows, list) or not rows:
            raise ValueError("manifest statements must be non-empty")
        names: set[str] = set()
        paths: set[str] = set()
        counts: Counter[str] = Counter()
        matrix: dict[tuple[str, str], set[int]] = {}
        expected_primes = {int(value) for value in manifest.get("primes", [])}
        for row in rows:
            name = str(row["name"])
            relative = str(row["path"])
            expected = str(row["expected"])
            if name in names or relative in paths:
                raise ValueError(f"duplicate manifest row: {name}")
            names.add(name)
            paths.add(relative)
            spec_path = args.corpus_root / relative
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            ir.validate_spec(spec)
            polynomial = ir.extract_polynomial(spec)
            if expected == "identity" and not polynomial["zero"]:
                raise ValueError(f"identity has nonzero polynomial: {name}")
            if expected == "counterexample" and polynomial["zero"]:
                raise ValueError(f"control has zero polynomial: {name}")
            counts[expected] += 1
            matrix.setdefault((str(row["relation_id"]), str(row["variant"])), set()).add(int(row["prime"]))
        for key, observed in matrix.items():
            if observed != expected_primes:
                raise ValueError(f"incomplete prime matrix for {key}: {sorted(observed)}")
        if int(manifest.get("statement_count", -1)) != len(rows):
            raise ValueError("statement_count mismatch")
        report = {
            "schema_version": 1,
            "engine": "geometric_identity_v4_1_duality_validator",
            "statement_count": len(rows),
            "identity_count": counts["identity"],
            "control_count": counts["counterexample"],
            "primes": sorted(expected_primes),
            "validation": "PASS",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.markdown_out.write_text(
            "\n".join(
                [
                    "# V4.1 fixed-blade duality corpus validation",
                    "",
                    f"- statements: {report['statement_count']}",
                    f"- identity rows: {report['identity_count']}",
                    f"- control rows: {report['control_count']}",
                    f"- primes: `{report['primes']}`",
                    "- validation: **PASS**",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ir.IdentityError, ir.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_DUALITY_VALIDATION: PASS "
        f"statements={report['statement_count']} identities={report['identity_count']} "
        f"controls={report['control_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
