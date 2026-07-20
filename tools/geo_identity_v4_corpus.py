#!/usr/bin/env python3
"""Build a V1-compatible V4 identity/control corpus from audited host relations.

The tool selects audited nonzero relation representatives, reconstructs their ASTs
from the preflight record table, emits exact identity specifications across a prime
matrix, and creates deterministic negated-right-hand-side controls. Each control is
accepted only when the exact V3 polynomial classifier confirms that the mutation is
not equivalent to the source identity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3_TOOL = ROOT / "tools" / "geo_identity_grammar_discovery.py"


def load_v3():
    spec = importlib.util.spec_from_file_location("geo_identity_grammar_discovery", V3_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {V3_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_v3()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def stable_id(payload: object, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:length]


def sanitize(value: str) -> str:
    output = "".join(character if character.isalnum() else "_" for character in value)
    return output.strip("_") or "relation"


def relation_priority(record: dict[str, Any]) -> tuple[int, int, str, str]:
    order = {
        "known-direct": 0,
        "known-derived-1": 1,
        "support-derived": 2,
        "unclassified": 3,
    }
    return (
        order.get(str(record.get("classification")), 9),
        int(record.get("combined_cost", 0)),
        str(record.get("known_family", "")),
        str(record.get("lhs", "")) + "=" + str(record.get("rhs", "")),
    )


def build_corpus(
    grammar_path: Path,
    preflight_path: Path,
    audit_path: Path,
    output_root: Path,
    primes: list[int],
    max_relations: int,
    max_controls: int,
) -> dict[str, Any]:
    grammar = V3.load_grammar(grammar_path)
    classifier = V3.SymbolicClassifier(grammar)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    record_by_label = {
        str(record["label"]): record for record in preflight.get("records", [])
    }
    audited = sorted(audit.get("relations", []), key=relation_priority)
    selected = audited[:max_relations]
    if not selected:
        raise ValueError("audit contains no relations")

    corpus_dir = output_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    controls_created = 0

    for relation_index, relation in enumerate(selected, 1):
        lhs_label = str(relation["lhs"])
        rhs_label = str(relation["rhs"])
        if lhs_label not in record_by_label or rhs_label not in record_by_label:
            raise ValueError(f"missing preflight expression for {lhs_label!r} or {rhs_label!r}")
        lhs = copy.deepcopy(record_by_label[lhs_label]["expression"])
        rhs = copy.deepcopy(record_by_label[rhs_label]["expression"])
        lhs_record = classifier.classify(lhs)
        rhs_record = classifier.classify(rhs)
        if lhs_record.exact_hash != rhs_record.exact_hash or lhs_record.zero or rhs_record.zero:
            raise ValueError(f"audited relation is not a nonzero exact equivalence: {lhs_label} = {rhs_label}")

        relation_payload = {
            "lhs": lhs,
            "rhs": rhs,
            "family": relation.get("known_family"),
            "classification": relation.get("classification"),
        }
        relation_id = f"r{relation_index:03d}_{stable_id(relation_payload)}"

        variants = [
            {
                "variant": "identity",
                "expected": "identity",
                "lhs": lhs,
                "rhs": rhs,
                "mutation": None,
            }
        ]
        if controls_created < max_controls:
            control_rhs = {"op": "neg", "arg": copy.deepcopy(rhs)}
            control_record = classifier.classify(control_rhs)
            if control_record.exact_hash != lhs_record.exact_hash:
                controls_created += 1
                variants.append(
                    {
                        "variant": f"control_{controls_created:02d}",
                        "expected": "counterexample",
                        "lhs": lhs,
                        "rhs": control_rhs,
                        "mutation": "negate_rhs",
                    }
                )

        for variant in variants:
            for prime in primes:
                if not V3.exact.is_prime(prime):
                    raise ValueError(f"not prime: {prime}")
                name = sanitize(
                    f"v4_{grammar.name}_{relation_id}_{variant['variant']}_p{prime}"
                )
                specification = {
                    "schema_version": 1,
                    "name": name,
                    "description": (
                        f"V4 {relation.get('classification')} relation "
                        f"{lhs_label} = {rhs_label}; variant={variant['variant']}"
                    ),
                    "expected": variant["expected"],
                    "dimension": grammar.dimension,
                    "signature": list(grammar.signature),
                    "prime": prime,
                    "coefficient_bound": grammar.coefficient_bound,
                    "seed": grammar.seed,
                    "variables": [copy.deepcopy(variable) for variable in grammar.variables],
                    "lhs": copy.deepcopy(variant["lhs"]),
                    "rhs": copy.deepcopy(variant["rhs"]),
                }
                filename = f"{name}.json"
                write_json(corpus_dir / filename, specification)
                manifest_rows.append(
                    {
                        "name": name,
                        "path": str(Path("corpus") / filename).replace("\\", "/"),
                        "relation_id": relation_id,
                        "family": relation.get("known_family"),
                        "classification": relation.get("classification"),
                        "variant": variant["variant"],
                        "expected": variant["expected"],
                        "prime": prime,
                        "lhs_label": lhs_label,
                        "rhs_label": rhs_label,
                        "mutation": variant["mutation"],
                    }
                )

    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_v4_corpus",
        "grammar": grammar.name,
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "primes": primes,
        "selected_relation_count": len(selected),
        "control_count": controls_created,
        "statement_count": len(manifest_rows),
        "statements": manifest_rows,
    }
    write_json(output_root / "corpus-manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primes", type=int, nargs="+", default=[65521, 65519])
    parser.add_argument("--max-relations", type=int, default=12)
    parser.add_argument("--max-controls", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_corpus(
            args.grammar,
            args.preflight_json,
            args.audit_json,
            args.output_root,
            list(args.primes),
            args.max_relations,
            args.max_controls,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, V3.GrammarError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_CORPUS: PASS "
        f"relations={manifest['selected_relation_count']} "
        f"controls={manifest['control_count']} "
        f"statements={manifest['statement_count']}"
    )
    print(f"manifest: {args.output_root / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
