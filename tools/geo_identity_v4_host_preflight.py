#!/usr/bin/env python3
"""V4 host-only preflight for macro-expanded Clifford involutions.

This stage deliberately reuses the exact V3 symbolic classifier. New operators are
expanded into the already validated AST vocabulary before classification:

- grade involution: sum_k (-1)^k <x>_k
- Clifford conjugation: reverse(grade_involution(x))

The tool enumerates a bounded seed set, classifies every expanded expression by
exact blade-wise integer polynomials, groups exact equivalence classes, removes
the universal zero class from relation ranking, and emits nonzero exact relations
for review before any CUDA corpus is generated.
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

    zero_classes = []
    nonzero_groups: dict[str, list[dict[str, Any]]] = {}
    for exact_hash, members in groups.items():
        if all(member["zero"] for member in members):
            zero_classes.append(
                {
                    "exact_hash": exact_hash,
                    "member_count": len(members),
                    "members": sorted(member["label"] for member in members),
                }
            )
        else:
            nonzero_groups[exact_hash] = members

    relations = []
    for exact_hash, members in sorted(nonzero_groups.items()):
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
    zero_classes.sort(key=lambda item: (-item["member_count"], item["exact_hash"]))
    return {
        "schema_version": 2,
        "engine": "geometric_identity_v4_host_preflight",
        "grammar": grammar.name,
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "expression_count": len(records),
        "equivalence_class_count": len(groups),
        "nonzero_equivalence_class_count": len(nonzero_groups),
        "zero_equivalence_class_count": len(zero_classes),
        "zero_expression_count": sum(item["member_count"] for item in zero_classes),
        "relation_count": len(relations),
        "records": records,
        "zero_classes": zero_classes,
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
        f"- nonzero equivalence classes: {report['nonzero_equivalence_class_count']}",
        f"- zero equivalence classes: {report['zero_equivalence_class_count']}",
        f"- expressions in zero classes: {report['zero_expression_count']}",
        f"- ranked nonzero relations: {report['relation_count']}",
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
            "## Excluded zero classes",
            "",
            "Expressions that classify to the zero polynomial are recorded but not paired",
            "as candidate relations. This prevents unrelated structural zeros such as",
            "`v^v = 0` and `[B,B] = 0` from appearing as meaningful identities between",
            "their surface expressions.",
            "",
        ]
    )
    for zero_class in report["zero_classes"]:
        members = ", ".join(f"`{member}`" for member in zero_class["members"])
        lines.append(f"- {zero_class['member_count']} expressions: {members}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Every ranked relation is a nonzero exact polynomial equivalence in the declared scope.",
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
        f"nonzero_relations={report['relation_count']} "
        f"zero_expressions={report['zero_expression_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
