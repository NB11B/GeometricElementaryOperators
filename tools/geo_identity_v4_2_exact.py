#!/usr/bin/env python3
"""Native characteristic-zero Clifford polynomial backend for V4.2.

This module owns fixed-blade validation and exact evaluation. It does not patch
legacy modules. Expressions are classified blade-by-blade as sparse integer
polynomials in the active coefficients of the declared variables.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable, Sequence

try:
    import geo_identity_discovery as legacy
except ModuleNotFoundError:
    from tools import geo_identity_discovery as legacy

DiscoveryError = legacy.DiscoveryError
Expression = dict[str, Any]
Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]
PolyMV = list[Polynomial]
SUPPORTED_OPS = {"add", "sub", "neg", "scale", "gp", "wedge", "commutator", "reverse", "grade"}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def blade_label(blade: int, dimension: int) -> str:
    if blade == 0:
        return "1"
    return "e" + "".join(str(index + 1) for index in range(dimension) if blade & (1 << index))


def active_blades(dimension: int, grades: Iterable[int]) -> list[int]:
    grade_set = set(grades)
    return [blade for blade in range(1 << dimension) if blade.bit_count() in grade_set]


def gp_sign(blade_left: int, blade_right: int, signature: Sequence[int]) -> int:
    sign = 1
    for index, metric in enumerate(signature):
        bit = 1 << index
        if blade_left & bit:
            if (blade_right & (bit - 1)).bit_count() & 1:
                sign = -sign
            if blade_right & bit:
                sign *= int(metric)
    return sign


def wedge_sign(blade_left: int, blade_right: int) -> int:
    if blade_left & blade_right:
        return 0
    sign = 1
    value = blade_left
    while value:
        lowest = value & -value
        if (blade_right & (lowest - 1)).bit_count() & 1:
            sign = -sign
        value ^= lowest
    return sign


def pseudoscalar_square(signature: Sequence[int]) -> int:
    dimension = len(signature)
    result = -1 if ((dimension * (dimension - 1) // 2) & 1) else 1
    for metric in signature:
        if metric not in (-1, 1):
            raise ValueError("signature entries must be +/-1")
        result *= int(metric)
    return result


def parse_fixed_blade(raw: object, dimension: int, context: str) -> tuple[int, int] | None:
    if not isinstance(raw, dict) or "fixed_blade" not in raw:
        return None
    if set(raw) != {"fixed_blade"}:
        raise ValueError(f"{context}: fixed_blade expression has extra fields")
    payload = raw["fixed_blade"]
    if not isinstance(payload, dict) or set(payload) != {"blade", "coefficient"}:
        raise ValueError(f"{context}.fixed_blade must contain exactly blade and coefficient")
    blade = payload["blade"]
    coefficient = payload["coefficient"]
    if not is_integer(blade):
        raise ValueError(f"{context}.fixed_blade.blade must be an integer")
    if not 0 <= blade < (1 << dimension):
        raise ValueError(f"{context}.fixed_blade.blade must be in [0,{(1 << dimension) - 1}]")
    if not is_integer(coefficient):
        raise ValueError(f"{context}.fixed_blade.coefficient must be an integer")
    if coefficient == 0:
        raise ValueError(f"{context}.fixed_blade.coefficient must be nonzero")
    return int(blade), int(coefficient)


def validate_expression(expression: object, variables: set[str], dimension: int, context: str) -> None:
    if not isinstance(expression, dict):
        raise DiscoveryError(f"{context} must be an expression object")
    try:
        fixed = parse_fixed_blade(expression, dimension, context)
    except ValueError as exc:
        raise DiscoveryError(str(exc)) from exc
    if fixed is not None:
        return
    if "var" in expression:
        if set(expression) != {"var"} or expression["var"] not in variables:
            raise DiscoveryError(f"{context}: invalid variable expression")
        return
    if "scalar" in expression:
        if set(expression) != {"scalar"} or not is_integer(expression["scalar"]):
            raise DiscoveryError(f"{context}: invalid scalar expression")
        return
    op = expression.get("op")
    if op not in SUPPORTED_OPS:
        raise DiscoveryError(f"{context}: unsupported operation {op!r}")
    if op in {"neg", "reverse"}:
        if set(expression) != {"op", "arg"}:
            raise DiscoveryError(f"{context}: invalid {op} fields")
        validate_expression(expression["arg"], variables, dimension, f"{context}.arg")
        return
    if op == "scale":
        if set(expression) != {"op", "value", "arg"} or not is_integer(expression["value"]):
            raise DiscoveryError(f"{context}: invalid scale expression")
        validate_expression(expression["arg"], variables, dimension, f"{context}.arg")
        return
    if op == "grade":
        if set(expression) != {"op", "grade", "arg"} or not is_integer(expression["grade"]):
            raise DiscoveryError(f"{context}: invalid grade expression")
        if not 0 <= expression["grade"] <= dimension:
            raise DiscoveryError(f"{context}.grade is outside dimension")
        validate_expression(expression["arg"], variables, dimension, f"{context}.arg")
        return
    arguments = expression.get("args")
    if not isinstance(arguments, list):
        raise DiscoveryError(f"{context}: {op}.args must be a list")
    if op == "add":
        if len(arguments) < 2:
            raise DiscoveryError(f"{context}: add requires at least two arguments")
    elif len(arguments) != 2:
        raise DiscoveryError(f"{context}: {op} requires exactly two arguments")
    if set(expression) != {"op", "args"}:
        raise DiscoveryError(f"{context}: invalid {op} fields")
    for index, argument in enumerate(arguments):
        validate_expression(argument, variables, dimension, f"{context}.args[{index}]")


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise DiscoveryError("spec must be a schema_version 1 object")
    dimension = spec.get("dimension")
    if not is_integer(dimension) or not 1 <= dimension <= 6:
        raise DiscoveryError("dimension must be an integer in [1,6]")
    signature = spec.get("signature")
    if not isinstance(signature, list) or len(signature) != dimension or any(value not in (-1, 1) for value in signature):
        raise DiscoveryError("signature must contain one +/-1 entry per dimension")
    prime = spec.get("prime", 65521)
    if not is_integer(prime) or not legacy.is_prime(prime):
        raise DiscoveryError("prime must be prime")
    if spec.get("expected") not in {"identity", "counterexample"}:
        raise DiscoveryError("expected must be identity or counterexample")
    variables = spec.get("variables")
    if not isinstance(variables, list) or not variables:
        raise DiscoveryError("variables must be a non-empty list")
    names: set[str] = set()
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict) or set(variable) != {"name", "grades"}:
            raise DiscoveryError(f"variables[{index}] is invalid")
        name = variable["name"]
        grades = variable["grades"]
        if not isinstance(name, str) or not name or name in names:
            raise DiscoveryError(f"variables[{index}].name is invalid or duplicate")
        if not isinstance(grades, list) or not grades or any(not is_integer(value) or not 0 <= value <= dimension for value in grades):
            raise DiscoveryError(f"variables[{index}].grades is invalid")
        names.add(name)
    validate_expression(spec.get("lhs"), names, dimension, "lhs")
    validate_expression(spec.get("rhs"), names, dimension, "rhs")
    return spec


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    output = dict(left)
    for monomial, coefficient in right.items():
        value = output.get(monomial, 0) + coefficient
        if value:
            output[monomial] = value
        else:
            output.pop(monomial, None)
    return output


def poly_scale(value: Polynomial, factor: int) -> Polynomial:
    if factor == 0:
        return {}
    return {monomial: coefficient * factor for monomial, coefficient in value.items() if coefficient * factor}


def poly_multiply(left: Polynomial, right: Polynomial, term_limit: int) -> Polynomial:
    output: Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(sorted(left_monomial + right_monomial))
            coefficient = output.get(monomial, 0) + left_coefficient * right_coefficient
            if coefficient:
                output[monomial] = coefficient
            else:
                output.pop(monomial, None)
            if len(output) > term_limit:
                raise DiscoveryError(f"polynomial term limit exceeded: {term_limit}")
    return output


def mv_zero(dimension: int) -> PolyMV:
    return [{} for _ in range(1 << dimension)]


def mv_add(left: PolyMV, right: PolyMV) -> PolyMV:
    return [poly_add(a, b) for a, b in zip(left, right)]


def mv_scale(value: PolyMV, factor: int) -> PolyMV:
    return [poly_scale(polynomial, factor) for polynomial in value]


def mv_gp(left: PolyMV, right: PolyMV, signature: Sequence[int], term_limit: int) -> PolyMV:
    output = mv_zero(len(signature))
    for blade_left, polynomial_left in enumerate(left):
        if not polynomial_left:
            continue
        for blade_right, polynomial_right in enumerate(right):
            if not polynomial_right:
                continue
            blade = blade_left ^ blade_right
            product = poly_multiply(polynomial_left, polynomial_right, term_limit)
            output[blade] = poly_add(output[blade], poly_scale(product, gp_sign(blade_left, blade_right, signature)))
    return output


def mv_wedge(left: PolyMV, right: PolyMV, dimension: int, term_limit: int) -> PolyMV:
    output = mv_zero(dimension)
    for blade_left, polynomial_left in enumerate(left):
        if not polynomial_left:
            continue
        for blade_right, polynomial_right in enumerate(right):
            if not polynomial_right:
                continue
            sign = wedge_sign(blade_left, blade_right)
            if not sign:
                continue
            blade = blade_left | blade_right
            product = poly_multiply(polynomial_left, polynomial_right, term_limit)
            output[blade] = poly_add(output[blade], poly_scale(product, sign))
    return output


def symbolic_variables(spec: dict[str, Any]) -> tuple[dict[str, PolyMV], list[dict[str, Any]]]:
    dimension = spec["dimension"]
    values: dict[str, PolyMV] = {}
    symbols: list[dict[str, Any]] = []
    symbol_id = 0
    for variable in spec["variables"]:
        value = mv_zero(dimension)
        for blade in active_blades(dimension, variable["grades"]):
            value[blade] = {(symbol_id,): 1}
            symbols.append({"id": symbol_id, "variable": variable["name"], "blade": blade, "blade_label": blade_label(blade, dimension)})
            symbol_id += 1
        values[variable["name"]] = value
    return values, symbols


def polynomial_expression(spec: dict[str, Any], expression: Expression, variable_values: dict[str, PolyMV], term_limit: int, memo: dict[str, PolyMV] | None = None) -> PolyMV:
    if memo is None:
        memo = {}
    key = canonical_json(expression)
    if key in memo:
        return copy.deepcopy(memo[key])
    dimension = spec["dimension"]
    signature = spec["signature"]
    fixed = parse_fixed_blade(expression, dimension, "expression")
    if fixed is not None:
        blade, coefficient = fixed
        value = mv_zero(dimension)
        value[blade] = {(): coefficient}
    elif "var" in expression:
        value = copy.deepcopy(variable_values[expression["var"]])
    elif "scalar" in expression:
        value = mv_zero(dimension)
        if expression["scalar"]:
            value[0] = {(): int(expression["scalar"])}
    else:
        op = expression["op"]
        if op == "add":
            value = mv_zero(dimension)
            for argument in expression["args"]:
                value = mv_add(value, polynomial_expression(spec, argument, variable_values, term_limit, memo))
        elif op == "sub":
            left, right = expression["args"]
            value = mv_add(polynomial_expression(spec, left, variable_values, term_limit, memo), mv_scale(polynomial_expression(spec, right, variable_values, term_limit, memo), -1))
        elif op == "neg":
            value = mv_scale(polynomial_expression(spec, expression["arg"], variable_values, term_limit, memo), -1)
        elif op == "scale":
            value = mv_scale(polynomial_expression(spec, expression["arg"], variable_values, term_limit, memo), int(expression["value"]))
        elif op in {"gp", "wedge", "commutator"}:
            left_raw, right_raw = expression["args"]
            left = polynomial_expression(spec, left_raw, variable_values, term_limit, memo)
            right = polynomial_expression(spec, right_raw, variable_values, term_limit, memo)
            if op == "gp":
                value = mv_gp(left, right, signature, term_limit)
            elif op == "wedge":
                value = mv_wedge(left, right, dimension, term_limit)
            else:
                value = mv_add(mv_gp(left, right, signature, term_limit), mv_scale(mv_gp(right, left, signature, term_limit), -1))
        elif op == "reverse":
            argument = polynomial_expression(spec, expression["arg"], variable_values, term_limit, memo)
            value = mv_zero(dimension)
            for blade, polynomial in enumerate(argument):
                grade = blade.bit_count()
                value[blade] = poly_scale(polynomial, -1 if ((grade * (grade - 1) // 2) & 1) else 1)
        elif op == "grade":
            argument = polynomial_expression(spec, expression["arg"], variable_values, term_limit, memo)
            projected = int(expression["grade"])
            value = [copy.deepcopy(polynomial) if blade.bit_count() == projected else {} for blade, polynomial in enumerate(argument)]
        else:
            raise DiscoveryError(f"unsupported operation {op!r}")
    memo[key] = copy.deepcopy(value)
    return value


def polynomial_rows(value: PolyMV, dimension: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for blade, polynomial in enumerate(value):
        if not polynomial:
            continue
        rows.append({"blade": blade, "blade_label": blade_label(blade, dimension), "terms": [{"monomial": list(monomial), "coefficient": polynomial[monomial]} for monomial in sorted(polynomial)]})
    return rows


def extract_polynomial(spec: dict[str, Any], term_limit: int = 500_000) -> dict[str, Any]:
    validate_spec(spec)
    variables, symbols = symbolic_variables(spec)
    memo: dict[str, PolyMV] = {}
    lhs = polynomial_expression(spec, spec["lhs"], variables, term_limit, memo)
    rhs = polynomial_expression(spec, spec["rhs"], variables, term_limit, memo)
    difference = mv_add(lhs, mv_scale(rhs, -1))
    rows = polynomial_rows(difference, spec["dimension"])
    degrees = [len(term["monomial"]) for row in rows for term in row["terms"]]
    payload = {"dimension": spec["dimension"], "signature": spec["signature"], "symbols": symbols, "blades": rows}
    return {
        "zero": not rows,
        "total_terms": sum(len(row["terms"]) for row in rows),
        "maximum_degree": max(degrees, default=0),
        "hash": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "symbols": symbols,
        "blades": rows,
    }
