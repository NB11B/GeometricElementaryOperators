#!/usr/bin/env python3
"""Exact host-only duality preflight with a fixed basis-blade pseudoscalar.

This stage deliberately does not claim end-to-end compiler support for basis constants.
It extends the exact polynomial classifier locally with one fixed multivector constant,
I = e1234, and checks the declared Cl(2,2) duality contract before the shared IR,
host emitter, and CUDA emitter are modified.
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


def fixed_blade(blade: int, coefficient: int = 1) -> Expr:
    return {"fixed_blade": blade, "coefficient": coefficient}


def canonical(expr: Expr) -> Expr:
    if "fixed_blade" in expr:
        return {
            "fixed_blade": int(expr["fixed_blade"]),
            "coefficient": int(expr.get("coefficient", 1)),
        }
    if "var" in expr:
        return {"var": expr["var"]}
    if "scalar" in expr:
        return {"scalar": int(expr["scalar"])}
    output: Expr = {"op": expr["op"]}
    if "value" in expr:
        output["value"] = int(expr["value"])
    if "grade" in expr:
        output["grade"] = int(expr["grade"])
    if "arg" in expr:
        output["arg"] = canonical(expr["arg"])
    if "args" in expr:
        output["args"] = [canonical(value) for value in expr["args"]]
    return output


class DualClassifier:
    def __init__(self, grammar) -> None:
        self.grammar = grammar
        self.base = V3.SymbolicClassifier(grammar)
        self.memo: dict[str, Any] = {}

    def polynomial(self, expr: Expr):
        expr = canonical(expr)
        key = json.dumps(expr, sort_keys=True, separators=(",", ":"))
        if key in self.memo:
            return copy.deepcopy(self.memo[key])
        count = 1 << self.grammar.dimension
        if "fixed_blade" in expr:
            blade = int(expr["fixed_blade"])
            coefficient = int(expr.get("coefficient", 1))
            if blade < 0 or blade >= count:
                raise ValueError(f"fixed blade {blade} is outside dimension")
            value = [{} for _ in range(count)]
            if coefficient:
                value[blade] = {(): coefficient}
        elif "var" in expr:
            value = copy.deepcopy(self.base.variable_values[expr["var"]])
        elif "scalar" in expr:
            value = [{} for _ in range(count)]
            if expr["scalar"]:
                value[0] = {(): int(expr["scalar"])}
        else:
            op = expr["op"]
            if op == "gp":
                value = V3.exact.polymv_gp(
                    self.polynomial(expr["args"][0]),
                    self.polynomial(expr["args"][1]),
                    self.grammar.signature,
                    self.grammar.term_limit,
                )
            elif op == "wedge":
                value = V3.exact.polymv_wedge(
                    self.polynomial(expr["args"][0]),
                    self.polynomial(expr["args"][1]),
                    self.grammar.term_limit,
                )
            elif op == "add":
                value = V3.exact.polymv_add(
                    [self.polynomial(item) for item in expr["args"]], count
                )
            elif op == "sub":
                left = self.polynomial(expr["args"][0])
                right = V3.exact.polymv_scale(self.polynomial(expr["args"][1]), -1)
                value = V3.exact.polymv_add((left, right), count)
            elif op == "neg":
                value = V3.exact.polymv_scale(self.polynomial(expr["arg"]), -1)
            elif op == "scale":
                value = V3.exact.polymv_scale(
                    self.polynomial(expr["arg"]), int(expr["value"])
                )
            elif op == "reverse":
                source = self.polynomial(expr["arg"])
                value = []
                for blade, polynomial in enumerate(source):
                    grade = blade.bit_count()
                    sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                    value.append(V3.exact.poly_scale(polynomial, sign))
            elif op == "grade":
                source = self.polynomial(expr["arg"])
                target = int(expr["grade"])
                value = [
                    polynomial if blade.bit_count() == target else {}
                    for blade, polynomial in enumerate(source)
                ]
            else:
                raise ValueError(f"unsupported local duality operation: {op}")
        self.memo[key] = copy.deepcopy(value)
        return value

    def metadata(self, expr: Expr) -> dict[str, Any]:
        return V3.polynomial_metadata(
            self.polynomial(expr),
            dimension=self.grammar.dimension,
            signature=self.grammar.signature,
            symbols=self.base.symbols,
        )


def gp(a: Expr, b: Expr) -> Expr:
    return {"op": "gp", "args": [a, b]}


def reverse(a: Expr) -> Expr:
    return {"op": "reverse", "arg": a}


def dual(a: Expr, inverse_i: Expr) -> Expr:
    return gp(a, inverse_i)


def undual(a: Expr, i: Expr) -> Expr:
    return gp(a, i)


def build(grammar_path: Path) -> dict[str, Any]:
    grammar = V3.load_grammar(grammar_path)
    if grammar.dimension != 4 or tuple(grammar.signature) != (1, 1, -1, -1):
        raise ValueError("this preflight currently requires Cl(2,2)")
    classifier = DualClassifier(grammar)
    I = fixed_blade(15, 1)
    I_inv = fixed_blade(15, 1)

    i2 = classifier.metadata(gp(I, I))
    scalar_one = classifier.metadata({"scalar": 1})
    if i2["exact_hash"] != scalar_one["exact_hash"]:
        raise ValueError("pseudoscalar square contract failed")

    variables = [(v["name"], {"var": v["name"]}) for v in grammar.variables]
    records: list[dict[str, Any]] = []
    for name, expr in variables:
        candidates = [
            (name, expr, "seed"),
            (f"dual({name})", dual(expr, I_inv), "dual"),
            (f"undual(dual({name}))", undual(dual(expr, I_inv), I), "dual_round_trip"),
            (f"dual(dual({name}))", dual(dual(expr, I_inv), I_inv), "dual_square"),
            (f"reverse(dual({name}))", reverse(dual(expr, I_inv)), "reverse_dual"),
        ]
        for label, expression, source in candidates:
            metadata = classifier.metadata(expression)
            records.append(
                {
                    "label": label,
                    "source": source,
                    "expression": canonical(expression),
                    "exact_hash": metadata["exact_hash"],
                    "zero": metadata["zero"],
                    "total_terms": metadata["total_terms"],
                    "maximum_degree": metadata["maximum_degree"],
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["exact_hash"]].append(record)
    relations = []
    for exact_hash, members in sorted(groups.items()):
        if len(members) < 2 or all(member["zero"] for member in members):
            continue
        members = sorted(members, key=lambda item: (item["source"], item["label"]))
        representative = members[0]
        for other in members[1:]:
            relations.append(
                {
                    "exact_hash": exact_hash,
                    "lhs": representative["label"],
                    "rhs": other["label"],
                    "lhs_source": representative["source"],
                    "rhs_source": other["source"],
                }
            )

    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_duality_preflight",
        "grammar": grammar.name,
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "pseudoscalar": {"blade": 15, "coefficient": 1, "square": 1, "inverse": 1},
        "fixed_blade_ast_status": "host-local-only",
        "expression_count": len(records),
        "equivalence_class_count": len(groups),
        "relation_count": len(relations),
        "records": records,
        "relations": relations,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.1 exact duality preflight",
        "",
        f"- grammar: `{report['grammar']}`",
        f"- dimension: {report['dimension']}",
        f"- signature: `{report['signature']}`",
        "- pseudoscalar: `I = e1234`",
        "- exact contract: `I^2 = 1`, therefore `I^-1 = I`",
        f"- expressions: {report['expression_count']}",
        f"- exact equivalence classes: {report['equivalence_class_count']}",
        f"- ranked exact relations: {report['relation_count']}",
        "- shared AST status: `host-local-only`",
        "",
        "| lhs | rhs | sources |",
        "|---|---|---|",
    ]
    for relation in report["relations"]:
        lines.append(
            f"| `{relation['lhs']}` | `{relation['rhs']}` | "
            f"{relation['lhs_source']} / {relation['rhs_source']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is an exact characteristic-zero polynomial preflight using a local",
            "fixed-blade constant extension. It does not yet establish that the shared",
            "JSON IR, generated host evaluator, or CUDA evaluator can execute the node.",
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
        report = build(args.grammar)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, RuntimeError, V3.GrammarError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_DUALITY_PREFLIGHT: PASS "
        f"expressions={report['expression_count']} "
        f"classes={report['equivalence_class_count']} "
        f"relations={report['relation_count']} I2=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
