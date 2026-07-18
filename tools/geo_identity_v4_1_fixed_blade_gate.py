#!/usr/bin/env python3
"""Validate the proposed fixed-blade constant IR for V4.1 duality.

The proposed expression node is:

    {"fixed_blade": {"blade": 15, "coefficient": 1}}

This host gate validates the node, evaluates it exactly over the integers, and
checks the Cl(2,2) pseudoscalar contract I*I = 1. It is intentionally separate
from the shared v1 compiler until the generated host/CUDA evaluator is extended.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_PATH = ROOT / "tools" / "geo_identity_discovery.py"


def load_discovery():
    spec = importlib.util.spec_from_file_location("geo_identity_discovery", DISCOVERY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {DISCOVERY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = load_discovery()
Expr = dict[str, Any]


def fixed_blade(blade: int, coefficient: int = 1) -> Expr:
    return {"fixed_blade": {"blade": blade, "coefficient": coefficient}}


def validate_expression(expr: object, dimension: int, variables: set[str], context: str) -> None:
    if not isinstance(expr, dict):
        raise ValueError(f"{context} must be an expression object")
    if "fixed_blade" in expr:
        if set(expr) != {"fixed_blade"} or not isinstance(expr["fixed_blade"], dict):
            raise ValueError(f"{context}.fixed_blade must be an object")
        payload = expr["fixed_blade"]
        if set(payload) != {"blade", "coefficient"}:
            raise ValueError(f"{context}.fixed_blade requires blade and coefficient")
        blade = payload["blade"]
        coefficient = payload["coefficient"]
        if not isinstance(blade, int) or isinstance(blade, bool) or not 0 <= blade < (1 << dimension):
            raise ValueError(f"{context}.fixed_blade.blade is outside dimension")
        if not isinstance(coefficient, int) or isinstance(coefficient, bool):
            raise ValueError(f"{context}.fixed_blade.coefficient must be an integer")
        return
    if "var" in expr:
        if set(expr) != {"var"} or expr["var"] not in variables:
            raise ValueError(f"{context} references an unknown variable")
        return
    if "scalar" in expr:
        if set(expr) != {"scalar"} or not isinstance(expr["scalar"], int):
            raise ValueError(f"{context}.scalar must be an integer")
        return
    op = expr.get("op")
    if op in {"neg", "reverse"}:
        validate_expression(expr.get("arg"), dimension, variables, f"{context}.arg")
        return
    if op == "grade":
        grade = expr.get("grade")
        if not isinstance(grade, int) or not 0 <= grade <= dimension:
            raise ValueError(f"{context}.grade is outside dimension")
        validate_expression(expr.get("arg"), dimension, variables, f"{context}.arg")
        return
    if op == "scale":
        if not isinstance(expr.get("value"), int):
            raise ValueError(f"{context}.value must be an integer")
        validate_expression(expr.get("arg"), dimension, variables, f"{context}.arg")
        return
    if op in {"add", "sub", "gp", "wedge", "commutator"}:
        args = expr.get("args")
        required = 2 if op != "add" else None
        if not isinstance(args, list) or (required is not None and len(args) != required) or (op == "add" and len(args) < 2):
            raise ValueError(f"{context}.args has invalid arity")
        for index, arg in enumerate(args):
            validate_expression(arg, dimension, variables, f"{context}.args[{index}]")
        return
    raise ValueError(f"{context} uses unsupported operation {op!r}")


def poly_expression(spec: dict[str, Any], expr: Expr, variable_values: dict[str, Any], term_limit: int, memo: dict[str, Any]) -> Any:
    dimension = spec["dimension"]
    blade_count = 1 << dimension
    key = json.dumps(expr, sort_keys=True, separators=(",", ":"))
    if key in memo:
        return memo[key]
    if "fixed_blade" in expr:
        payload = expr["fixed_blade"]
        value = [{} for _ in range(blade_count)]
        coefficient = int(payload["coefficient"])
        if coefficient:
            value[int(payload["blade"])] = {(): coefficient}
    elif "var" in expr:
        value = variable_values[expr["var"]]
    elif "scalar" in expr:
        value = [{} for _ in range(blade_count)]
        if expr["scalar"]:
            value[0] = {(): int(expr["scalar"])}
    else:
        op = expr["op"]
        if op == "add":
            value = D.polymv_add([poly_expression(spec, arg, variable_values, term_limit, memo) for arg in expr["args"]], blade_count)
        elif op == "sub":
            left = poly_expression(spec, expr["args"][0], variable_values, term_limit, memo)
            right = poly_expression(spec, expr["args"][1], variable_values, term_limit, memo)
            value = D.polymv_add((left, D.polymv_scale(right, -1)), blade_count)
        elif op == "neg":
            value = D.polymv_scale(poly_expression(spec, expr["arg"], variable_values, term_limit, memo), -1)
        elif op == "scale":
            value = D.polymv_scale(poly_expression(spec, expr["arg"], variable_values, term_limit, memo), int(expr["value"]))
        elif op == "gp":
            value = D.polymv_gp(
                poly_expression(spec, expr["args"][0], variable_values, term_limit, memo),
                poly_expression(spec, expr["args"][1], variable_values, term_limit, memo),
                spec["signature"], term_limit)
        elif op == "wedge":
            value = D.polymv_wedge(
                poly_expression(spec, expr["args"][0], variable_values, term_limit, memo),
                poly_expression(spec, expr["args"][1], variable_values, term_limit, memo), term_limit)
        elif op == "commutator":
            left = poly_expression(spec, expr["args"][0], variable_values, term_limit, memo)
            right = poly_expression(spec, expr["args"][1], variable_values, term_limit, memo)
            value = D.polymv_add((D.polymv_gp(left, right, spec["signature"], term_limit), D.polymv_scale(D.polymv_gp(right, left, spec["signature"], term_limit), -1)), blade_count)
        elif op == "reverse":
            source = poly_expression(spec, expr["arg"], variable_values, term_limit, memo)
            value = []
            for blade, polynomial in enumerate(source):
                grade = blade.bit_count()
                sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                value.append(D.poly_scale(polynomial, sign))
        elif op == "grade":
            source = poly_expression(spec, expr["arg"], variable_values, term_limit, memo)
            value = [polynomial if blade.bit_count() == expr["grade"] else {} for blade, polynomial in enumerate(source)]
        else:
            raise AssertionError(op)
    memo[key] = value
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    dimension = 4
    signature = [1, 1, -1, -1]
    I = fixed_blade(15, 1)
    lhs = {"op": "gp", "args": [I, I]}
    rhs = {"scalar": 1}
    variables: set[str] = set()
    validate_expression(lhs, dimension, variables, "lhs")
    validate_expression(rhs, dimension, variables, "rhs")
    spec = {"dimension": dimension, "signature": signature}
    left = poly_expression(spec, lhs, {}, 1000, {})
    right = poly_expression(spec, rhs, {}, 1000, {})
    difference = [D.poly_add(a, D.poly_scale(b, -1)) for a, b in zip(left, right)]
    passed = not any(difference)
    report = {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_fixed_blade_gate",
        "node": I,
        "dimension": dimension,
        "signature": signature,
        "pseudoscalar_square": 1 if passed else None,
        "validation": "PASS" if passed else "FAIL",
        "shared_compiler_status": "pending",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(
        "# V4.1 fixed-blade IR gate\n\n"
        "- proposed node: `{'fixed_blade': {'blade': 15, 'coefficient': 1}}`\n"
        "- dimension: 4\n"
        "- signature: `[1, 1, -1, -1]`\n"
        f"- exact check: `I^2 = 1` -> **{report['validation']}**\n"
        "- shared compiler/CUDA status: `pending`\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"V4_1_FIXED_BLADE_GATE: {report['validation']} blade=15 I2={report['pseudoscalar_square']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
