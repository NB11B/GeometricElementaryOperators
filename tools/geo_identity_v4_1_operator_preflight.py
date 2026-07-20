#!/usr/bin/env python3
"""V4.1 host preflight for exact contractions and the pseudoscalar duality contract.

Contractions are macro-expanded into the already validated V3 AST vocabulary.
Duality is contract-checked at this stage: the pseudoscalar blade, square, and
inverse scale are derived exactly from the declared diagonal signature. Actual
dual-expression classification remains gated on basis-blade constant support in
the shared AST/compiler.
"""
from __future__ import annotations

import argparse
import copy
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


def add_many(expressions: list[Expr]) -> Expr:
    if not expressions:
        return {"scalar": 0}
    value = copy.deepcopy(expressions[0])
    for expression in expressions[1:]:
        value = {"op": "add", "args": [value, copy.deepcopy(expression)]}
    return value


def grade(expr: Expr, value: int) -> Expr:
    return {"op": "grade", "grade": value, "arg": copy.deepcopy(expr)}


def reverse(expr: Expr) -> Expr:
    return {"op": "reverse", "arg": copy.deepcopy(expr)}


def gp(left: Expr, right: Expr) -> Expr:
    return {"op": "gp", "args": [copy.deepcopy(left), copy.deepcopy(right)]}


def wedge(left: Expr, right: Expr) -> Expr:
    return {"op": "wedge", "args": [copy.deepcopy(left), copy.deepcopy(right)]}


def left_contraction(left: Expr, right: Expr, dimension: int) -> Expr:
    terms: list[Expr] = []
    for r in range(dimension + 1):
        for s in range(r, dimension + 1):
            terms.append(grade(gp(reverse(grade(left, r)), grade(right, s)), s - r))
    return add_many(terms)


def right_contraction(left: Expr, right: Expr, dimension: int) -> Expr:
    terms: list[Expr] = []
    for r in range(dimension + 1):
        for s in range(r + 1):
            terms.append(grade(gp(grade(left, r), reverse(grade(right, s))), r - s))
    return add_many(terms)


def make_grammar(raw: dict[str, Any]):
    dimension = int(raw["dimension"])
    variables = tuple(
        {"name": str(item["name"]), "grades": sorted({int(g) for g in item["grades"]})}
        for item in raw["variables"]
    )
    search = raw.get("search", {})
    return V3.GrammarConfig(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        dimension=dimension,
        signature=tuple(int(value) for value in raw["signature"]),
        prime=int(raw.get("prime", 65521)),
        coefficient_bound=int(raw.get("coefficient_bound", 3)),
        seed=int(raw.get("seed", 0x243F6A88)),
        variables=variables,
        max_cost=64,
        max_expressions=5000,
        max_per_cost=1000,
        max_representatives_per_polynomial=64,
        candidate_multiplier=1,
        unary_ops=("neg", "reverse"),
        binary_ops=("add", "sub", "gp", "wedge", "commutator"),
        scales=(-2, 2),
        grades=tuple(range(dimension + 1)),
        constants=(0,),
        families=(),
        max_relations=128,
        max_controls=16,
        max_pairs_per_class=128,
        max_relations_per_primitive_class=16,
        min_relation_variables=1,
        require_geometric_operator=True,
        term_limit=int(search.get("term_limit", 300000)),
    )


def pseudoscalar_contract(signature: tuple[int, ...]) -> dict[str, Any]:
    dimension = len(signature)
    blade = (1 << dimension) - 1
    square = V3.exact.gp_sign(blade, blade, signature)
    if square not in (-1, 1):
        raise ValueError(f"unsupported pseudoscalar square: {square}")
    inverse_scale = square
    return {
        "blade": blade,
        "blade_label": V3.exact.blade_label(blade, dimension),
        "square": square,
        "inverse_scale": inverse_scale,
        "right_dual": "A I^{-1}",
        "left_dual": "I^{-1} A",
        "relation_generation": False,
        "relation_generation_blocker": "shared AST/compiler lacks a fixed basis-blade constant node",
    }


def build_report(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    grammar = make_grammar(raw)
    classifier = V3.SymbolicClassifier(grammar)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    variable_grades = {
        str(item["name"]): tuple(int(g) for g in item["grades"])
        for item in raw["variables"]
    }
    variables = [(name, {"var": name}) for name in variable_grades]

    candidates: list[tuple[str, Expr, str]] = []
    for name, expr in variables:
        candidates.append((name, expr, "seed"))
    for left_name, left in variables:
        for right_name, right in variables:
            candidates.extend(
                [
                    (f"({left_name}*{right_name})", gp(left, right), "geometric_product"),
                    (f"({left_name}^{right_name})", wedge(left, right), "wedge"),
                    (f"lcon({left_name},{right_name})", left_contraction(left, right, grammar.dimension), "left_contraction"),
                    (f"rcon({left_name},{right_name})", right_contraction(left, right, grammar.dimension), "right_contraction"),
                ]
            )
            for r in variable_grades[left_name]:
                for s in variable_grades[right_name]:
                    difference = abs(s - r)
                    candidates.append(
                        (
                            f"grade{difference}({left_name}*{right_name})",
                            grade(gp(left, right), difference),
                            "product_grade_projection",
                        )
                    )
                    if r + s <= grammar.dimension:
                        candidates.append(
                            (
                                f"grade{r+s}({left_name}*{right_name})",
                                grade(gp(left, right), r + s),
                                "product_grade_projection",
                            )
                        )

    for label, expression, source in candidates:
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

    zero_classes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for exact_hash, members in groups.items():
        if all(member["zero"] for member in members):
            zero_classes.append(
                {
                    "exact_hash": exact_hash,
                    "member_count": len(members),
                    "members": sorted(member["label"] for member in members),
                }
            )
            continue
        ordered = sorted(members, key=lambda item: (item["cost"], item["label"]))
        if len(ordered) < 2:
            continue
        representative = ordered[0]
        for other in ordered[1:]:
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
    duality = pseudoscalar_contract(grammar.signature)
    declared = raw.get("operator_contract", {})
    if int(declared.get("pseudoscalar_blade", duality["blade"])) != duality["blade"]:
        raise ValueError("declared pseudoscalar blade does not match dimension")
    if int(declared.get("pseudoscalar_square", duality["square"])) != duality["square"]:
        raise ValueError("declared pseudoscalar square does not match signature")
    if int(declared.get("pseudoscalar_inverse_scale", duality["inverse_scale"])) != duality["inverse_scale"]:
        raise ValueError("declared pseudoscalar inverse scale does not match exact contract")

    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_operator_preflight",
        "grammar": grammar.name,
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "conventions": {
            "left_contraction": declared.get("left_contraction"),
            "right_contraction": declared.get("right_contraction"),
        },
        "duality_contract": duality,
        "expression_count": len(records),
        "equivalence_class_count": len(groups),
        "zero_equivalence_class_count": len(zero_classes),
        "zero_expression_count": sum(item["member_count"] for item in zero_classes),
        "relation_count": len(relations),
        "records": records,
        "relations": relations,
        "zero_classes": zero_classes,
    }


def markdown(report: dict[str, Any]) -> str:
    duality = report["duality_contract"]
    lines = [
        "# V4.1 contraction and duality-contract preflight",
        "",
        f"- grammar: `{report['grammar']}`",
        f"- dimension: {report['dimension']}",
        f"- signature: `{report['signature']}`",
        f"- expressions: {report['expression_count']}",
        f"- exact equivalence classes: {report['equivalence_class_count']}",
        f"- ranked nonzero relations: {report['relation_count']}",
        f"- zero classes: {report['zero_equivalence_class_count']}",
        f"- zero expressions excluded: {report['zero_expression_count']}",
        "",
        "## Operator contract",
        "",
        f"- left contraction: `{report['conventions']['left_contraction']}`",
        f"- right contraction: `{report['conventions']['right_contraction']}`",
        f"- pseudoscalar: `{duality['blade_label']}` (blade {duality['blade']})",
        f"- pseudoscalar square: `{duality['square']}`",
        f"- inverse: `I^(-1) = {duality['inverse_scale']} I`",
        f"- dual relation generation: `{duality['relation_generation']}`",
        f"- blocker: {duality['relation_generation_blocker']}",
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
            "Contraction expressions are exact macro expansions over grade projections,",
            "reversion, and geometric products. Duality is only contract-checked in this",
            "stage; no dual identity is ranked until fixed basis-blade constants are",
            "supported by the shared exact AST, host compiler, and CUDA emitter.",
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
        report = build_report(args.grammar)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, V3.GrammarError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_OPERATOR_PREFLIGHT: PASS "
        f"expressions={report['expression_count']} "
        f"classes={report['equivalence_class_count']} "
        f"relations={report['relation_count']} "
        f"zero_expressions={report['zero_expression_count']} "
        f"I2={report['duality_contract']['square']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
