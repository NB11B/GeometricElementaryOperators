#!/usr/bin/env python3
"""Validate a V4.2 signature-matrix corpus before host/CUDA execution."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import geo_identity_v4_2_engine as engine
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_engine as engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--python-checks", type=int, default=64)
    args = parser.parse_args()
    try:
        manifest_path = args.corpus_root / "corpus-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("statements")
        if not isinstance(rows, list) or not rows:
            raise ValueError("manifest statements must be non-empty")
        expected_primes = {int(value) for value in manifest.get("primes", [])}
        if not expected_primes:
            raise ValueError("manifest primes must be non-empty")
        names: set[str] = set()
        paths: set[str] = set()
        counts: Counter[str] = Counter()
        signatures: Counter[str] = Counter()
        matrix: dict[tuple[str, str], set[int]] = defaultdict(set)
        family_counts: Counter[str] = Counter()
        for row in rows:
            name = str(row["name"])
            relative = str(row["path"])
            if name in names or relative in paths:
                raise ValueError(f"duplicate statement row: {name}")
            names.add(name)
            paths.add(relative)
            spec_path = args.corpus_root / relative
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            engine.validate_spec(spec)
            if spec["name"] != name:
                raise ValueError(f"name mismatch: {name}")
            expected = str(row["expected"])
            if spec["expected"] != expected:
                raise ValueError(f"expected mismatch: {name}")
            if list(spec["signature"]) != list(row["signature"]):
                raise ValueError(f"signature mismatch: {name}")
            square = engine.pseudoscalar_square(spec["signature"])
            if square != int(row["pseudoscalar_square"]):
                raise ValueError(f"pseudoscalar-square mismatch: {name}")
            polynomial = engine.extract_polynomial(spec)
            if expected == "identity" and not polynomial["zero"]:
                raise ValueError(f"identity has nonzero polynomial: {name}")
            if expected == "counterexample" and polynomial["zero"]:
                raise ValueError(f"control has zero polynomial: {name}")
            identity = engine.load_identity(spec_path)
            found = False
            for assignment in range(args.python_checks):
                equal, _, _, _ = engine.evaluate_identity(identity, assignment)
                if not equal:
                    found = True
                    if expected == "identity":
                        raise ValueError(f"identity failed Python evaluation: {name} assignment={assignment}")
                    break
            if args.python_checks and expected == "counterexample" and not found:
                raise ValueError(f"control produced no witness in {args.python_checks} checks: {name}")
            counts[expected] += 1
            signatures[str(row["signature_name"])] += 1
            family_counts[str(row["family"])] += 1
            matrix[(str(row["definition_id"]), expected)].add(int(row["prime"]))
        for key, observed in matrix.items():
            if observed != expected_primes:
                raise ValueError(f"incomplete prime matrix for {key}: {sorted(observed)}")
        if int(manifest.get("statement_count", -1)) != len(rows):
            raise ValueError("statement_count mismatch")
        report = {
            "schema_version": 1,
            "engine": "geometric_identity_v4_2_validator",
            "statement_count": len(rows),
            "identity_count": counts["identity"],
            "control_count": counts["counterexample"],
            "signature_counts": dict(sorted(signatures.items())),
            "family_count": len(family_counts),
            "primes": sorted(expected_primes),
            "python_checks": args.python_checks,
            "validation": "PASS",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# V4.2 native signature-matrix validation",
            "",
            f"- statements: {report['statement_count']}",
            f"- identity rows: {report['identity_count']}",
            f"- control rows: {report['control_count']}",
            f"- normalized source families: {report['family_count']}",
            f"- primes: `{report['primes']}`",
            f"- Python checks per statement: {report['python_checks']}",
            "- validation: **PASS**",
            "",
            "## Signature rows",
            "",
        ]
        for name, count in report["signature_counts"].items():
            lines.append(f"- `{name}`: {count}")
        lines.append("")
        args.markdown_out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, engine.IdentityError, engine.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_2_VALIDATION: PASS "
        f"statements={report['statement_count']} identities={report['identity_count']} controls={report['control_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
