#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
from typing import Any, Sequence
try:
    import geo_identity_v4_2_compiler as compiler
    import geo_identity_v4_2_exact as exact
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_compiler as compiler
    from tools import geo_identity_v4_2_exact as exact
Expression = dict[str, Any]
IdentityError = compiler.IdentityError
DiscoveryError = exact.DiscoveryError
validate_spec = exact.validate_spec
extract_polynomial = exact.extract_polynomial
load_identity = compiler.load_identity
load_corpus = compiler.load_corpus
evaluate_identity = compiler.evaluate_identity
emit_header = compiler.emit_header
gp_sign = exact.gp_sign
pseudoscalar_square = exact.pseudoscalar_square

def fixed_blade(blade: int, coefficient: int = 1) -> Expression:
    return {"fixed_blade": {"blade": int(blade), "coefficient": int(coefficient)}}

def scalar(value: int) -> Expression:
    return {"scalar": int(value)}

def variable(name: str) -> Expression:
    return {"var": name}

def unary(op: str, argument: Expression) -> Expression:
    return {"op": op, "arg": copy.deepcopy(argument)}

def binary(op: str, left: Expression, right: Expression) -> Expression:
    return {"op": op, "args": [copy.deepcopy(left), copy.deepcopy(right)]}

def gp(left: Expression, right: Expression) -> Expression:
    return binary("gp", left, right)

def wedge(left: Expression, right: Expression) -> Expression:
    return binary("wedge", left, right)

def reverse(argument: Expression) -> Expression:
    return unary("reverse", argument)

def neg(argument: Expression) -> Expression:
    return unary("neg", argument)

def scale(factor: int, argument: Expression) -> Expression:
    return copy.deepcopy(argument) if factor == 1 else {"op": "scale", "value": int(factor), "arg": copy.deepcopy(argument)}

def grade(projected_grade: int, argument: Expression) -> Expression:
    return {"op": "grade", "grade": int(projected_grade), "arg": copy.deepcopy(argument)}

def add_many(expressions: Sequence[Expression]) -> Expression:
    values = [copy.deepcopy(value) for value in expressions]
    if not values:
        return scalar(0)
    if len(values) == 1:
        return values[0]
    return {"op": "add", "args": values}

def pseudoscalar(signature: Sequence[int], inverse: bool = False) -> Expression:
    coefficient = pseudoscalar_square(signature) if inverse else 1
    return fixed_blade((1 << len(signature)) - 1, coefficient)

def right_dual(argument: Expression, signature: Sequence[int]) -> Expression:
    return gp(argument, pseudoscalar(signature, True))

def left_dual(argument: Expression, signature: Sequence[int]) -> Expression:
    return gp(pseudoscalar(signature, True), argument)

def right_undual(argument: Expression, signature: Sequence[int]) -> Expression:
    return gp(argument, pseudoscalar(signature, False))

def left_undual(argument: Expression, signature: Sequence[int]) -> Expression:
    return gp(pseudoscalar(signature, False), argument)

def left_contraction(left: Expression, right: Expression, dimension: int) -> Expression:
    terms: list[Expression] = []
    for left_grade in range(dimension + 1):
        for right_grade in range(left_grade, dimension + 1):
            terms.append(grade(right_grade - left_grade, gp(reverse(grade(left_grade, left)), grade(right_grade, right))))
    return add_many(terms)

def right_contraction(left: Expression, right: Expression, dimension: int) -> Expression:
    terms: list[Expression] = []
    for left_grade in range(dimension + 1):
        for right_grade in range(left_grade + 1):
            terms.append(grade(left_grade - right_grade, gp(grade(left_grade, left), reverse(grade(right_grade, right)))))
    return add_many(terms)

def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def canonicalize(expression: Expression) -> Expression:
    if "fixed_blade" in expression:
        payload = expression["fixed_blade"]
        return fixed_blade(payload["blade"], payload["coefficient"])
    if "var" in expression:
        return variable(str(expression["var"]))
    if "scalar" in expression:
        return scalar(int(expression["scalar"]))
    op = expression["op"]
    if "arg" in expression:
        output: Expression = {"op": op, "arg": canonicalize(expression["arg"])}
        if op == "scale":
            output["value"] = int(expression["value"])
        if op == "grade":
            output["grade"] = int(expression["grade"])
        return output
    arguments = [canonicalize(argument) for argument in expression["args"]]
    if op == "add":
        arguments.sort(key=canonical_json)
    return {"op": op, "args": arguments}

def expression_cost(expression: Expression) -> int:
    if any(key in expression for key in ("var", "scalar", "fixed_blade")):
        return 1
    if "arg" in expression:
        return 1 + expression_cost(expression["arg"])
    return 1 + sum(expression_cost(argument) for argument in expression.get("args", []))

def expression_variables(expression: Expression) -> set[str]:
    if "var" in expression:
        return {str(expression["var"])}
    output: set[str] = set()
    if "arg" in expression:
        output.update(expression_variables(expression["arg"]))
    for argument in expression.get("args", []):
        output.update(expression_variables(argument))
    return output

def fixed_blade_permutation(dimension: int, signature: Sequence[int], blade: int, coefficient: int = 1, side: str = "right") -> list[dict[str, int]]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if not 0 <= blade < (1 << dimension):
        raise ValueError("blade is outside dimension")
    rows: list[dict[str, int]] = []
    for source in range(1 << dimension):
        if side == "right":
            target = source ^ blade
            factor = coefficient * gp_sign(source, blade, signature)
        else:
            target = blade ^ source
            factor = coefficient * gp_sign(blade, source, signature)
        rows.append({"source": source, "target": target, "factor": factor})
    return rows
