#!/usr/bin/env python3
"""V4 host-only preflight for macro-expanded Clifford involutions.

This stage deliberately reuses the exact V3 symbolic classifier. New operators are
expanded into the already validated AST vocabulary before classification:

- grade involution: sum_k (-1)^k <x>_k
- Clifford conjugation: reverse(grade_involution(x))

The tool enumerates a bounded seed set, classifies every expanded expression by
exact blade-wise integer polynomials, groups exact equivalence classes, and emits
candidate relations for review before any CUDA corpus is generated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
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
Expr = dict[str, Any]


def neg(expr: Expr) -> Expr:
    return {"op": "neg", "arg": expr}


def add_many(expressions: list[Expr]) -> Expr:
    if not expressions:
        return {"scalar": 0}
    current = expressions[0]
    for expression in expressions[1:]:
        current = {"op": "add", "args": [current, expression]}
    return current


def grade_involution(expr: Expr, dimension: int) -> Expr:
    terms: list[Expr] = []
    for grade in range(dimension + 1):
        projected = {"op": "grade", "grade": grade, "arg": expr}
        terms.append(projected if grade % 2 == 0 else neg(projected))
    return add_many(terms)


def clifford_conjugation(expr: Expr, dimension: int) -> Expr:
    return {"op": "reverse", "arg": grade_involution(expr, dimension)}


def seed_expressions(grammar) -> list[tuple[str, Expr]]:
    variables = [(variable["name"], {"var": variable["name"]}) for variable in grammar.variables]
    seeds = list(variables)
    for left_name, left in variables:
        for right_name, right in variables:
            seeds.append((f"({left_name}*{right_name})", {"op": "gp", "args": [left, right]}))
            seeds.append((f"({left_name}^{right_name})", {"op": "wedge", "args": [left, right]}))
            seeds.append((f"[{left_name},{right_name}]", {"op": "commutator", "args": [left, right]}))
    return seeds


def build_records(grammar_path: Path) -> dict[str, Any]:
    grammar = V3.load_grammar(grammar_path)
    classifier = V3.SymbolicClassifier(grammar)
    records = []
    seen: set[str] = set()

    for seed_label, seed in seed_expressions(grammar):
        variants = [
            (seed_label, seed, "seed"),
            (f"hat({seed_label})", grade_involution(seed, grammar.dimension), "grade_involution"),
            (f"bar({seed_label})", clifford_conjugation(seed, grammar.dimension), "clifford_conjugation"),
            (f"reverse({seed_label})", {"op": "reverse", "arg": seed}, "reverse"),
        ]
        for label, expression, source in variants:
            canonical = V3.canonicalize_expression(expression)
            key = V3.canonical_json(canonical)
            if key in seen:
                continue
            seen.add(key)
            classified = classifier.classify(canonical)
            records.append(
                {
                    "label": label,
                    "source": source,
                    "expression": canonical,
                    "expression_label": classified.label,
                    "cost": classified.cost,
                    "variables": list(classified.variables),
                    "operators": dict(classified.operators),
                    "exact_hash": classified.exact_hash,
                    "primitive_hash": classified.primitive_hash,
                    "zero": classified.zero,
                    "total_terms": classified.total_terms,
                    "maximum_degree": classified.maximum_degree,
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["exact_hash"]].append(record)

    relations = []
    for exact_hash, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: (item["cost"], item["label"]))
        representative = members[0]
        for other in members[1:]:
            if representative["label"] == other["label"]:
                continue
            relations.append(
                {
                    "exact_hash": exact_hash,
                    "lhs": representative["label"],
                    "rhs": other["label"],
                    "lhs_source": representative["source"],
                    "rhs_source": other["source"],
                    "combined_cost": representative["cost"] + other["cost"],
                }
            )

    relations.sort(key=lambda item: (item["combined_cost"], item["lhs"], item["rhs"]))
    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_host_preflight",
        "grammar": grammar.name,
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "expression_count": len(records),
        "equivalence_class_count": len(groups),
        "relation_count": len(relations),
        "records": records,
        "relations": relations,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4 host-only mixed-involution preflight",
        "",
        f"- grammar: `{report['grammar']}`",
        f"- dimension: {report['dimension']}",
        f"- signature: `{report['signature']}`",
        f"- expressions: {report['expression_count']}",
        f"- exact equivalence classes: {report['equivalence_class_count']}",
        f"- candidate relations: {report['relation_count']}",
        "",
        "| lhs | rhs | sources | cost |",
        "|---|---|---|---:|",
    ]
    for relation in report["relations"]:
        lines.append(
            f"| `{relation['lhs']}` | `{relation['rhs']}` | "
            f"{relation['lhs_source']} / {relation['rhs_source']} | "
            f"{relation['combined_cost']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Every listed relation is an exact polynomial equivalence in the declared scope.",
            "This host preflight does not yet generate near-miss controls or CUDA evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_records(args.grammar)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, RuntimeError, V3.GrammarError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_PREFLIGHT: PASS "
        f"expressions={report['expression_count']} "
        f"classes={report['equivalence_class_count']} "
        f"relations={report['relation_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
