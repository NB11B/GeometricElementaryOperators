#!/usr/bin/env python3
"""Discovery layer for the exact geometric identity engine.

This tool extends the v1 Clifford identity IR without changing its execution
backend. It can:

* expand a checked corpus across several odd primes;
* generate deterministic one-edit mutations from known identities;
* extract a canonical blade-wise polynomial difference over the integers;
* classify statements as formal polynomial identities or nonzero candidates;
* perform exact deterministic finite-field prechecks;
* reproduce and greedily minimize GPU counterexamples;
* aggregate cross-prime discovery results.

The polynomial extractor is exact for the bounded v1 expression language:
addition, subtraction, integer scale, geometric product, wedge product,
reversion, grade projection, and commutator. A zero extracted polynomial proves
that the two expressions agree for the declared dimension, signature, and
variable grade supports. It is not a proof across undeclared dimensions or
signatures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


class DiscoveryError(ValueError):
    """Raised when a discovery input violates the supported contract."""


MASK64 = (1 << 64) - 1
DEFAULT_PRIMES = (65521, 65519, 65497, 32749)
NONCOMMUTATIVE_BINARY_OPS = {"sub", "gp", "wedge", "commutator"}


@dataclass(frozen=True)
class Mutation:
    kind: str
    path: tuple[object, ...]
    side: str
    expression: dict[str, Any]
    stable_id: str


@dataclass(frozen=True)
class Mismatch:
    blade: int
    lhs: int
    rhs: int


Poly = dict[tuple[int, ...], int]
PolyMV = list[Poly]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def validate_spec(spec: object, context: str = "identity") -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise DiscoveryError(f"{context} must be a JSON object")
    required = {
        "schema_version",
        "name",
        "expected",
        "dimension",
        "signature",
        "variables",
        "lhs",
        "rhs",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise DiscoveryError(f"{context} is missing: {', '.join(missing)}")
    if spec["schema_version"] != 1:
        raise DiscoveryError(f"{context}.schema_version must be 1")
    name = spec["name"]
    if not isinstance(name, str) or not name:
        raise DiscoveryError(f"{context}.name must be non-empty")
    if spec["expected"] not in {"identity", "counterexample"}:
        raise DiscoveryError(f"{context}.expected must be identity or counterexample")
    dimension = spec["dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool) or not 1 <= dimension <= 6:
        raise DiscoveryError(f"{context}.dimension must be in [1,6]")
    signature = spec["signature"]
    if (
        not isinstance(signature, list)
        or len(signature) != dimension
        or any(value not in (-1, 1) for value in signature)
    ):
        raise DiscoveryError(
            f"{context}.signature must contain {dimension} entries from {{-1,+1}}"
        )
    prime = spec.get("prime", 65521)
    if (
        not isinstance(prime, int)
        or isinstance(prime, bool)
        or prime < 3
        or prime > 1_000_003
        or prime % 2 == 0
        or not is_prime(prime)
    ):
        raise DiscoveryError(f"{context}.prime must be an odd prime in [3,1000003]")
    bound = spec.get("coefficient_bound", 3)
    if (
        not isinstance(bound, int)
        or isinstance(bound, bool)
        or bound < 1
        or 2 * bound + 1 >= prime
    ):
        raise DiscoveryError(f"{context}.coefficient_bound is invalid for prime {prime}")
    seed = spec.get("seed", 0x243F6A88)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MASK64:
        raise DiscoveryError(f"{context}.seed must fit uint64")
    variables = spec["variables"]
    if not isinstance(variables, list) or not variables:
        raise DiscoveryError(f"{context}.variables must be non-empty")
    names: set[str] = set()
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            raise DiscoveryError(f"{context}.variables[{index}] must be an object")
        variable_name = variable.get("name")
        grades = variable.get("grades")
        if not isinstance(variable_name, str) or not variable_name:
            raise DiscoveryError(f"{context}.variables[{index}].name is invalid")
        if variable_name in names:
            raise DiscoveryError(f"{context}.variables contains duplicate {variable_name}")
        names.add(variable_name)
        if (
            not isinstance(grades, list)
            or not grades
            or any(
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade < 0
                or grade > dimension
                for grade in grades
            )
        ):
            raise DiscoveryError(f"{context}.variables[{index}].grades is invalid")
    _validate_expression(spec["lhs"], names, dimension, f"{context}.lhs")
    _validate_expression(spec["rhs"], names, dimension, f"{context}.rhs")
    return spec


def _validate_expression(
    expression: object,
    variables: set[str],
    dimension: int,
    context: str,
) -> None:
    if not isinstance(expression, dict):
        raise DiscoveryError(f"{context} must be an expression object")
    if "var" in expression:
        if set(expression) != {"var"} or expression["var"] not in variables:
            raise DiscoveryError(f"{context} references an unknown variable")
        return
    if "scalar" in expression:
        if set(expression) != {"scalar"} or not isinstance(expression["scalar"], int):
            raise DiscoveryError(f"{context}.scalar must be an integer")
        return
    op = expression.get("op")
    if op == "add":
        args = expression.get("args")
        if not isinstance(args, list) or len(args) < 2:
            raise DiscoveryError(f"{context}.args must contain at least two expressions")
        for index, argument in enumerate(args):
            _validate_expression(argument, variables, dimension, f"{context}.args[{index}]")
        return
    if op in NONCOMMUTATIVE_BINARY_OPS:
        args = expression.get("args")
        if not isinstance(args, list) or len(args) != 2:
            raise DiscoveryError(f"{context}.args must contain exactly two expressions")
        for index, argument in enumerate(args):
            _validate_expression(argument, variables, dimension, f"{context}.args[{index}]")
        return
    if op in {"neg", "reverse"}:
        _validate_expression(expression.get("arg"), variables, dimension, f"{context}.arg")
        return
    if op == "scale":
        if not isinstance(expression.get("value"), int):
            raise DiscoveryError(f"{context}.value must be an integer")
        _validate_expression(expression.get("arg"), variables, dimension, f"{context}.arg")
        return
    if op == "grade":
        grade = expression.get("grade")
        if not isinstance(grade, int) or not 0 <= grade <= dimension:
            raise DiscoveryError(f"{context}.grade is outside dimension")
        _validate_expression(expression.get("arg"), variables, dimension, f"{context}.arg")
        return
    raise DiscoveryError(f"{context} uses unsupported operation {op!r}")


def load_spec(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"unable to read {path}: {exc}") from exc
    return validate_spec(raw, str(path))


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
                sign *= metric
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


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def generate_assignment(spec: dict[str, Any], assignment: int) -> list[list[int]]:
    dimension = spec["dimension"]
    prime = spec.get("prime", 65521)
    bound = spec.get("coefficient_bound", 3)
    seed = spec.get("seed", 0x243F6A88)
    count = 1 << dimension
    span = 2 * bound + 1
    output: list[list[int]] = []
    for variable_index, variable in enumerate(spec["variables"]):
        coefficients = [0] * count
        active = active_blades(dimension, variable["grades"])
        nonzero = False
        for blade in active:
            mixed = (
                seed
                ^ ((assignment + 1) * 0xD1B54A32D192ED03)
                ^ ((variable_index + 1) * 0x94D049BB133111EB)
                ^ ((blade + 1) * 0x9E3779B97F4A7C15)
            ) & MASK64
            signed = int(splitmix64(mixed) % span) - bound
            coefficients[blade] = signed % prime
            nonzero = nonzero or signed != 0
        if not nonzero:
            coefficients[active[0]] = 1
        output.append(coefficients)
    return output


def mv_add(values: Sequence[Sequence[int]], prime: int) -> list[int]:
    if not values:
        return []
    return [sum(entries) % prime for entries in zip(*values)]


def mv_neg(value: Sequence[int], prime: int) -> list[int]:
    return [(-coefficient) % prime for coefficient in value]


def mv_scale(value: Sequence[int], factor: int, prime: int) -> list[int]:
    return [(factor * coefficient) % prime for coefficient in value]


def mv_gp(
    left: Sequence[int],
    right: Sequence[int],
    signature: Sequence[int],
    prime: int,
) -> list[int]:
    output = [0] * len(left)
    for blade_left, coefficient_left in enumerate(left):
        if coefficient_left == 0:
            continue
        for blade_right, coefficient_right in enumerate(right):
            if coefficient_right == 0:
                continue
            blade = blade_left ^ blade_right
            output[blade] = (
                output[blade]
                + gp_sign(blade_left, blade_right, signature)
                * coefficient_left
                * coefficient_right
            ) % prime
    return output


def mv_wedge(left: Sequence[int], right: Sequence[int], prime: int) -> list[int]:
    output = [0] * len(left)
    for blade_left, coefficient_left in enumerate(left):
        if coefficient_left == 0:
            continue
        for blade_right, coefficient_right in enumerate(right):
            if coefficient_right == 0:
                continue
            sign = wedge_sign(blade_left, blade_right)
            if sign == 0:
                continue
            blade = blade_left | blade_right
            output[blade] = (
                output[blade] + sign * coefficient_left * coefficient_right
            ) % prime
    return output


def evaluate_expression(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variables: Sequence[Sequence[int]],
    variable_index: dict[str, int],
    memo: dict[str, list[int]] | None = None,
) -> list[int]:
    prime = spec.get("prime", 65521)
    dimension = spec["dimension"]
    count = 1 << dimension
    if memo is None:
        memo = {}
    key = canonical_json(expression)
    cached = memo.get(key)
    if cached is not None:
        return list(cached)
    if "var" in expression:
        value = list(variables[variable_index[expression["var"]]])
    elif "scalar" in expression:
        value = [0] * count
        value[0] = expression["scalar"] % prime
    else:
        op = expression["op"]
        if op == "add":
            value = mv_add(
                [
                    evaluate_expression(spec, argument, variables, variable_index, memo)
                    for argument in expression["args"]
                ],
                prime,
            )
        elif op == "sub":
            left = evaluate_expression(
                spec, expression["args"][0], variables, variable_index, memo
            )
            right = evaluate_expression(
                spec, expression["args"][1], variables, variable_index, memo
            )
            value = mv_add((left, mv_neg(right, prime)), prime)
        elif op == "neg":
            value = mv_neg(
                evaluate_expression(spec, expression["arg"], variables, variable_index, memo),
                prime,
            )
        elif op == "scale":
            value = mv_scale(
                evaluate_expression(spec, expression["arg"], variables, variable_index, memo),
                expression["value"],
                prime,
            )
        elif op == "gp":
            value = mv_gp(
                evaluate_expression(
                    spec, expression["args"][0], variables, variable_index, memo
                ),
                evaluate_expression(
                    spec, expression["args"][1], variables, variable_index, memo
                ),
                spec["signature"],
                prime,
            )
        elif op == "wedge":
            value = mv_wedge(
                evaluate_expression(
                    spec, expression["args"][0], variables, variable_index, memo
                ),
                evaluate_expression(
                    spec, expression["args"][1], variables, variable_index, memo
                ),
                prime,
            )
        elif op == "commutator":
            left = evaluate_expression(
                spec, expression["args"][0], variables, variable_index, memo
            )
            right = evaluate_expression(
                spec, expression["args"][1], variables, variable_index, memo
            )
            value = mv_add(
                (
                    mv_gp(left, right, spec["signature"], prime),
                    mv_neg(mv_gp(right, left, spec["signature"], prime), prime),
                ),
                prime,
            )
        elif op == "reverse":
            source = evaluate_expression(
                spec, expression["arg"], variables, variable_index, memo
            )
            value = [0] * count
            for blade, coefficient in enumerate(source):
                grade = blade.bit_count()
                sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                value[blade] = sign * coefficient % prime
        elif op == "grade":
            source = evaluate_expression(
                spec, expression["arg"], variables, variable_index, memo
            )
            value = [
                coefficient if blade.bit_count() == expression["grade"] else 0
                for blade, coefficient in enumerate(source)
            ]
        else:
            raise AssertionError(op)
    memo[key] = list(value)
    return value


def signed_value(value: int, prime: int) -> int:
    return value - prime if value > prime // 2 else value


def evaluate_with_variables(
    spec: dict[str, Any],
    variables: Sequence[Sequence[int]],
) -> Mismatch | None:
    variable_index = {
        variable["name"]: index for index, variable in enumerate(spec["variables"])
    }
    memo: dict[str, list[int]] = {}
    lhs = evaluate_expression(spec, spec["lhs"], variables, variable_index, memo)
    rhs = evaluate_expression(spec, spec["rhs"], variables, variable_index, memo)
    prime = spec.get("prime", 65521)
    for blade, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return Mismatch(blade, signed_value(left, prime), signed_value(right, prime))
    return None


def evaluate_assignment(spec: dict[str, Any], assignment: int) -> Mismatch | None:
    return evaluate_with_variables(spec, generate_assignment(spec, assignment))


def precheck(spec: dict[str, Any], assignments: int) -> tuple[int | None, Mismatch | None]:
    for assignment in range(assignments):
        mismatch = evaluate_assignment(spec, assignment)
        if mismatch is not None:
            return assignment, mismatch
    return None, None


def poly_add(left: Poly, right: Poly) -> Poly:
    output = dict(left)
    for monomial, coefficient in right.items():
        output[monomial] = output.get(monomial, 0) + coefficient
        if output[monomial] == 0:
            del output[monomial]
    return output


def poly_scale(poly: Poly, factor: int) -> Poly:
    if factor == 0:
        return {}
    return {monomial: coefficient * factor for monomial, coefficient in poly.items()}


def poly_mul(left: Poly, right: Poly, term_limit: int) -> Poly:
    output: Poly = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(sorted(left_monomial + right_monomial))
            output[monomial] = (
                output.get(monomial, 0) + left_coefficient * right_coefficient
            )
            if output[monomial] == 0:
                del output[monomial]
            if len(output) > term_limit:
                raise DiscoveryError(
                    f"polynomial term limit {term_limit} exceeded during multiplication"
                )
    return output


def polymv_add(values: Sequence[PolyMV], blade_count: int) -> PolyMV:
    output: PolyMV = [{} for _ in range(blade_count)]
    for value in values:
        for blade in range(blade_count):
            output[blade] = poly_add(output[blade], value[blade])
    return output


def polymv_scale(value: PolyMV, factor: int) -> PolyMV:
    return [poly_scale(poly, factor) for poly in value]


def polymv_gp(
    left: PolyMV,
    right: PolyMV,
    signature: Sequence[int],
    term_limit: int,
) -> PolyMV:
    output: PolyMV = [{} for _ in left]
    for blade_left, polynomial_left in enumerate(left):
        if not polynomial_left:
            continue
        for blade_right, polynomial_right in enumerate(right):
            if not polynomial_right:
                continue
            blade = blade_left ^ blade_right
            product = poly_scale(
                poly_mul(polynomial_left, polynomial_right, term_limit),
                gp_sign(blade_left, blade_right, signature),
            )
            output[blade] = poly_add(output[blade], product)
            if sum(len(poly) for poly in output) > term_limit:
                raise DiscoveryError(
                    f"polynomial term limit {term_limit} exceeded during geometric product"
                )
    return output


def polymv_wedge(left: PolyMV, right: PolyMV, term_limit: int) -> PolyMV:
    output: PolyMV = [{} for _ in left]
    for blade_left, polynomial_left in enumerate(left):
        if not polynomial_left:
            continue
        for blade_right, polynomial_right in enumerate(right):
            if not polynomial_right:
                continue
            sign = wedge_sign(blade_left, blade_right)
            if sign == 0:
                continue
            blade = blade_left | blade_right
            product = poly_scale(
                poly_mul(polynomial_left, polynomial_right, term_limit), sign
            )
            output[blade] = poly_add(output[blade], product)
            if sum(len(poly) for poly in output) > term_limit:
                raise DiscoveryError(
                    f"polynomial term limit {term_limit} exceeded during wedge product"
                )
    return output


def polynomial_expression(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variable_values: dict[str, PolyMV],
    term_limit: int,
    memo: dict[str, PolyMV] | None = None,
) -> PolyMV:
    dimension = spec["dimension"]
    blade_count = 1 << dimension
    if memo is None:
        memo = {}
    key = canonical_json(expression)
    cached = memo.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    if "var" in expression:
        value = copy.deepcopy(variable_values[expression["var"]])
    elif "scalar" in expression:
        value = [{} for _ in range(blade_count)]
        if expression["scalar"] != 0:
            value[0] = {(): expression["scalar"]}
    else:
        op = expression["op"]
        if op == "add":
            value = polymv_add(
                [
                    polynomial_expression(
                        spec, argument, variable_values, term_limit, memo
                    )
                    for argument in expression["args"]
                ],
                blade_count,
            )
        elif op == "sub":
            left = polynomial_expression(
                spec, expression["args"][0], variable_values, term_limit, memo
            )
            right = polynomial_expression(
                spec, expression["args"][1], variable_values, term_limit, memo
            )
            value = polymv_add((left, polymv_scale(right, -1)), blade_count)
        elif op == "neg":
            value = polymv_scale(
                polynomial_expression(
                    spec, expression["arg"], variable_values, term_limit, memo
                ),
                -1,
            )
        elif op == "scale":
            value = polymv_scale(
                polynomial_expression(
                    spec, expression["arg"], variable_values, term_limit, memo
                ),
                expression["value"],
            )
        elif op == "gp":
            value = polymv_gp(
                polynomial_expression(
                    spec, expression["args"][0], variable_values, term_limit, memo
                ),
                polynomial_expression(
                    spec, expression["args"][1], variable_values, term_limit, memo
                ),
                spec["signature"],
                term_limit,
            )
        elif op == "wedge":
            value = polymv_wedge(
                polynomial_expression(
                    spec, expression["args"][0], variable_values, term_limit, memo
                ),
                polynomial_expression(
                    spec, expression["args"][1], variable_values, term_limit, memo
                ),
                term_limit,
            )
        elif op == "commutator":
            left = polynomial_expression(
                spec, expression["args"][0], variable_values, term_limit, memo
            )
            right = polynomial_expression(
                spec, expression["args"][1], variable_values, term_limit, memo
            )
            value = polymv_add(
                (
                    polymv_gp(left, right, spec["signature"], term_limit),
                    polymv_scale(
                        polymv_gp(right, left, spec["signature"], term_limit), -1
                    ),
                ),
                blade_count,
            )
        elif op == "reverse":
            source = polynomial_expression(
                spec, expression["arg"], variable_values, term_limit, memo
            )
            value = []
            for blade, polynomial in enumerate(source):
                grade = blade.bit_count()
                sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                value.append(poly_scale(polynomial, sign))
        elif op == "grade":
            source = polynomial_expression(
                spec, expression["arg"], variable_values, term_limit, memo
            )
            value = [
                polynomial if blade.bit_count() == expression["grade"] else {}
                for blade, polynomial in enumerate(source)
            ]
        else:
            raise AssertionError(op)
    if sum(len(poly) for poly in value) > term_limit:
        raise DiscoveryError(f"polynomial term limit {term_limit} exceeded")
    memo[key] = copy.deepcopy(value)
    return value


def extract_polynomial(
    spec: dict[str, Any],
    term_limit: int = 200_000,
) -> dict[str, Any]:
    validate_spec(spec)
    dimension = spec["dimension"]
    blade_count = 1 << dimension
    symbols: list[dict[str, Any]] = []
    variable_values: dict[str, PolyMV] = {}
    symbol_id = 0
    for variable in spec["variables"]:
        value: PolyMV = [{} for _ in range(blade_count)]
        for blade in active_blades(dimension, variable["grades"]):
            value[blade] = {(symbol_id,): 1}
            symbols.append(
                {
                    "id": symbol_id,
                    "variable": variable["name"],
                    "blade": blade,
                    "blade_label": blade_label(blade, dimension),
                    "name": f"{variable['name']}_{blade_label(blade, dimension)}",
                }
            )
            symbol_id += 1
        variable_values[variable["name"]] = value
    memo: dict[str, PolyMV] = {}
    lhs = polynomial_expression(spec, spec["lhs"], variable_values, term_limit, memo)
    rhs = polynomial_expression(spec, spec["rhs"], variable_values, term_limit, memo)
    difference = [
        poly_add(left, poly_scale(right, -1)) for left, right in zip(lhs, rhs)
    ]
    coefficients = [
        abs(coefficient)
        for polynomial in difference
        for coefficient in polynomial.values()
        if coefficient
    ]
    content = 0
    for coefficient in coefficients:
        content = math.gcd(content, coefficient)
    primitive = copy.deepcopy(difference)
    if content > 1:
        primitive = [
            {monomial: coefficient // content for monomial, coefficient in polynomial.items()}
            for polynomial in primitive
        ]
    leading: int | None = None
    for polynomial in primitive:
        if polynomial:
            first_monomial = sorted(polynomial)[0]
            leading = polynomial[first_monomial]
            break
    primitive_sign = 1
    if leading is not None and leading < 0:
        primitive_sign = -1
        primitive = [poly_scale(polynomial, -1) for polynomial in primitive]
    blade_rows: list[dict[str, Any]] = []
    primitive_rows: list[dict[str, Any]] = []
    for blade in range(blade_count):
        if difference[blade]:
            blade_rows.append(
                {
                    "blade": blade,
                    "blade_label": blade_label(blade, dimension),
                    "terms": [
                        {
                            "monomial": list(monomial),
                            "coefficient": difference[blade][monomial],
                        }
                        for monomial in sorted(difference[blade])
                    ],
                }
            )
        if primitive[blade]:
            primitive_rows.append(
                {
                    "blade": blade,
                    "blade_label": blade_label(blade, dimension),
                    "terms": [
                        {
                            "monomial": list(monomial),
                            "coefficient": primitive[blade][monomial],
                        }
                        for monomial in sorted(primitive[blade])
                    ],
                }
            )
    canonical_payload = {
        "dimension": dimension,
        "signature": spec["signature"],
        "symbols": symbols,
        "primitive_blades": primitive_rows,
    }
    canonical_hash = hashlib.sha256(
        canonical_json(canonical_payload).encode("utf-8")
    ).hexdigest()
    degrees = [
        len(monomial)
        for polynomial in difference
        for monomial in polynomial
    ]
    return {
        "schema_version": 1,
        "identity_name": spec["name"],
        "dimension": dimension,
        "signature": spec["signature"],
        "symbols": symbols,
        "zero": not blade_rows,
        "total_terms": sum(len(polynomial) for polynomial in difference),
        "maximum_degree": max(degrees, default=0),
        "content_gcd": content,
        "primitive_sign": primitive_sign,
        "canonical_hash": canonical_hash,
        "blades": blade_rows,
        "primitive_blades": primitive_rows,
    }


def walk_expression(
    expression: dict[str, Any],
    path: tuple[object, ...] = (),
) -> Iterator[tuple[tuple[object, ...], dict[str, Any]]]:
    yield path, expression
    if "op" not in expression:
        return
    if "arg" in expression:
        yield from walk_expression(expression["arg"], path + ("arg",))
    if "args" in expression:
        for index, argument in enumerate(expression["args"]):
            yield from walk_expression(argument, path + ("args", index))


def replace_path(
    expression: dict[str, Any],
    path: tuple[object, ...],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(replacement)
    output = copy.deepcopy(expression)
    cursor: Any = output
    for segment in path[:-1]:
        cursor = cursor[segment]
    cursor[path[-1]] = copy.deepcopy(replacement)
    return output


def mutation_edit_candidates(
    node: dict[str, Any],
    dimension: int,
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    op = node.get("op")
    if op in NONCOMMUTATIVE_BINARY_OPS:
        swapped = copy.deepcopy(node)
        swapped["args"] = [copy.deepcopy(node["args"][1]), copy.deepcopy(node["args"][0])]
        candidates.append(("swap_operands", swapped))
    if op == "gp":
        for replacement_op in ("wedge", "commutator"):
            replacement = copy.deepcopy(node)
            replacement["op"] = replacement_op
            candidates.append((f"gp_to_{replacement_op}", replacement))
    elif op == "wedge":
        replacement = copy.deepcopy(node)
        replacement["op"] = "gp"
        candidates.append(("wedge_to_gp", replacement))
    elif op == "commutator":
        for replacement_op in ("gp", "wedge"):
            replacement = copy.deepcopy(node)
            replacement["op"] = replacement_op
            candidates.append((f"commutator_to_{replacement_op}", replacement))
    elif op == "sub":
        replacement = {"op": "add", "args": copy.deepcopy(node["args"])}
        candidates.append(("sub_to_add", replacement))
    elif op == "reverse":
        candidates.append(("remove_reverse", copy.deepcopy(node["arg"])))
    elif op == "grade":
        grade = node["grade"]
        for new_grade in (grade - 1, grade + 1):
            if 0 <= new_grade <= dimension:
                replacement = copy.deepcopy(node)
                replacement["grade"] = new_grade
                candidates.append((f"grade_{grade}_to_{new_grade}", replacement))
        candidates.append(("remove_grade_projection", copy.deepcopy(node["arg"])))
    elif op == "scale":
        replacement = copy.deepcopy(node)
        replacement["value"] = -replacement["value"]
        candidates.append(("negate_scale", replacement))
    return candidates


def generate_mutations(
    spec: dict[str, Any],
    max_mutations: int,
) -> list[tuple[Mutation, dict[str, Any]]]:
    validate_spec(spec)
    source_pair = canonical_json({"lhs": spec["lhs"], "rhs": spec["rhs"]})
    seen: set[str] = {source_pair}
    output: list[tuple[Mutation, dict[str, Any]]] = []
    for side in ("lhs", "rhs"):
        sign_flipped = copy.deepcopy(spec)
        sign_flipped[side] = {"op": "neg", "arg": copy.deepcopy(spec[side])}
        payload = canonical_json({"lhs": sign_flipped["lhs"], "rhs": sign_flipped["rhs"]})
        if payload not in seen:
            seen.add(payload)
            stable_id = hashlib.sha256(
                f"{spec['name']}|{side}|root_sign_flip|{payload}".encode("utf-8")
            ).hexdigest()[:12]
            mutation = Mutation("root_sign_flip", (), side, sign_flipped[side], stable_id)
            output.append((mutation, sign_flipped))
    for side in ("lhs", "rhs"):
        for path, node in walk_expression(spec[side]):
            for kind, replacement in mutation_edit_candidates(node, spec["dimension"]):
                mutated = copy.deepcopy(spec)
                mutated[side] = replace_path(spec[side], path, replacement)
                payload = canonical_json({"lhs": mutated["lhs"], "rhs": mutated["rhs"]})
                if payload in seen:
                    continue
                seen.add(payload)
                path_text = "/".join(str(segment) for segment in path) or "root"
                stable_id = hashlib.sha256(
                    f"{spec['name']}|{side}|{path_text}|{kind}|{payload}".encode("utf-8")
                ).hexdigest()[:12]
                mutation = Mutation(kind, path, side, replacement, stable_id)
                output.append((mutation, mutated))
                if len(output) >= max_mutations:
                    return output
    return output[:max_mutations]


def polynomial_nonzero_mod_prime(polynomial: dict[str, Any], prime: int) -> bool:
    for blade in polynomial["blades"]:
        for term in blade["terms"]:
            if term["coefficient"] % prime:
                return True
    return False


def stable_variant_name(
    source_name: str,
    variant_index: int,
    mutation: Mutation | None,
    prime: int,
) -> str:
    if mutation is None:
        return f"{source_name}__original__p{prime}"
    kind = "".join(character if character.isalnum() else "_" for character in mutation.kind)
    return f"{source_name}__m{variant_index:03d}_{kind}_{mutation.stable_id}__p{prime}"


def build_corpus(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    corpus_dir = output_dir / "corpus"
    polynomial_dir = output_dir / "polynomials"
    if output_dir.exists() and args.clean:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    polynomial_dir.mkdir(parents=True, exist_ok=True)
    primes = tuple(dict.fromkeys(args.prime or DEFAULT_PRIMES))
    for prime in primes:
        if not is_prime(prime) or not 3 <= prime <= 1_000_003:
            raise DiscoveryError(f"invalid configured prime: {prime}")
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    file_index = 1
    for source_path in args.source:
        source = load_spec(source_path)
        source_polynomial = extract_polynomial(source, args.term_limit)
        sources.append(
            {
                "path": source_path.as_posix(),
                "name": source["name"],
                "expected": source["expected"],
                "polynomial_zero": source_polynomial["zero"],
                "polynomial_hash": source_polynomial["canonical_hash"],
            }
        )
        variants: list[tuple[Mutation | None, dict[str, Any], dict[str, Any]]] = [
            (None, copy.deepcopy(source), source_polynomial)
        ]
        if source["expected"] == "identity" or args.mutate_counterexamples:
            for mutation, mutated in generate_mutations(source, args.max_mutations):
                mutated_polynomial = extract_polynomial(mutated, args.term_limit)
                variants.append((mutation, mutated, mutated_polynomial))
        for variant_index, (mutation, variant, polynomial) in enumerate(variants):
            variant_id = (
                f"{source['name']}:original"
                if mutation is None
                else f"{source['name']}:{mutation.stable_id}"
            )
            polynomial_path = polynomial_dir / f"{hashlib.sha256(variant_id.encode()).hexdigest()[:16]}.json"
            if not polynomial_path.exists():
                polynomial_record = copy.deepcopy(polynomial)
                polynomial_record["variant_id"] = variant_id
                polynomial_record["source_identity"] = source["name"]
                polynomial_record["mutation"] = (
                    None
                    if mutation is None
                    else {
                        "stable_id": mutation.stable_id,
                        "kind": mutation.kind,
                        "side": mutation.side,
                        "path": list(mutation.path),
                    }
                )
                write_json(polynomial_path, polynomial_record)
            for prime in primes:
                generated = copy.deepcopy(variant)
                generated["prime"] = prime
                generated_name = stable_variant_name(
                    source["name"], variant_index, mutation, prime
                )
                generated["name"] = generated_name
                provenance = (
                    "original cross-prime control"
                    if mutation is None
                    else f"automatic mutation {mutation.kind} at {mutation.side}:{list(mutation.path)}"
                )
                generated["description"] = (
                    f"{variant.get('description', '')} [v2 {provenance}; "
                    f"source={source['name']}; variant={variant_id}; prime={prime}]"
                ).strip()
                if polynomial["zero"]:
                    generated["expected"] = "identity"
                elif polynomial_nonzero_mod_prime(polynomial, prime):
                    generated["expected"] = "counterexample"
                else:
                    generated["expected"] = "identity"
                validate_spec(generated, generated_name)
                first_assignment, mismatch = precheck(
                    generated, args.precheck_assignments
                )
                filename = f"{file_index:04d}_{generated_name}.json"
                file_index += 1
                write_json(corpus_dir / filename, generated)
                entries.append(
                    {
                        "file": f"corpus/{filename}",
                        "name": generated_name,
                        "variant_id": variant_id,
                        "source_identity": source["name"],
                        "source_expected": source["expected"],
                        "prime": prime,
                        "expected": generated["expected"],
                        "mutation": (
                            None
                            if mutation is None
                            else {
                                "stable_id": mutation.stable_id,
                                "kind": mutation.kind,
                                "side": mutation.side,
                                "path": list(mutation.path),
                            }
                        ),
                        "polynomial_file": polynomial_path.relative_to(output_dir).as_posix(),
                        "polynomial_zero": polynomial["zero"],
                        "polynomial_hash": polynomial["canonical_hash"],
                        "polynomial_terms": polynomial["total_terms"],
                        "maximum_degree": polynomial["maximum_degree"],
                        "precheck_assignments": args.precheck_assignments,
                        "precheck_found_counterexample": mismatch is not None,
                        "precheck_witness": (
                            None
                            if mismatch is None
                            else {
                                "assignment": first_assignment,
                                "blade": mismatch.blade,
                                "blade_label": blade_label(
                                    mismatch.blade, generated["dimension"]
                                ),
                                "lhs": mismatch.lhs,
                                "rhs": mismatch.rhs,
                            }
                        ),
                    }
                )
    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_discovery_v2",
        "primes": list(primes),
        "precheck_assignments": args.precheck_assignments,
        "max_mutations_per_source": args.max_mutations,
        "term_limit": args.term_limit,
        "sources": sources,
        "entries": entries,
    }
    write_json(output_dir / "corpus-manifest.json", manifest)
    identity_list = "\n".join(entry["file"] for entry in entries) + "\n"
    (output_dir / "identity-files.txt").write_text(
        identity_list, encoding="utf-8", newline="\n"
    )
    summary = [
        "# Geometric identity discovery corpus",
        "",
        f"- source identities: {len(sources)}",
        f"- generated finite-field statements: {len(entries)}",
        f"- primes: {', '.join(str(prime) for prime in primes)}",
        f"- deterministic prechecks per statement: {args.precheck_assignments}",
        f"- formal polynomial identities: {sum(1 for entry in entries if entry['polynomial_zero'])}",
        f"- formal nonzero differences: {sum(1 for entry in entries if not entry['polynomial_zero'])}",
        f"- precheck counterexamples: {sum(1 for entry in entries if entry['precheck_found_counterexample'])}",
        "",
        "A zero canonical polynomial is an exact identity for the declared dimension,",
        "signature, and variable grade supports. Cross-prime search remains useful as",
        "an independent backend and witness-validation check.",
        "",
    ]
    (output_dir / "corpus-summary.md").write_text(
        "\n".join(summary), encoding="utf-8", newline="\n"
    )
    print(f"generated {len(entries)} statements in {corpus_dir}")
    print(f"manifest: {output_dir / 'corpus-manifest.json'}")
    return 0


def variables_to_signed(
    variables: Sequence[Sequence[int]], prime: int
) -> list[list[int]]:
    return [[signed_value(value, prime) for value in variable] for variable in variables]


def variables_to_mod(
    variables: Sequence[Sequence[int]], prime: int
) -> list[list[int]]:
    return [[value % prime for value in variable] for variable in variables]


def witness_score(variables_signed: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    nonzero = 0
    magnitude = 0
    highest_blade = 0
    for variable in variables_signed:
        for blade, coefficient in enumerate(variable):
            if coefficient:
                nonzero += 1
                magnitude += abs(coefficient)
                highest_blade = max(highest_blade, blade)
    return nonzero, magnitude, highest_blade


def minimize_witness(
    spec: dict[str, Any],
    assignment: int,
) -> tuple[list[list[int]], Mismatch, tuple[int, int, int]]:
    prime = spec.get("prime", 65521)
    signed_variables = variables_to_signed(generate_assignment(spec, assignment), prime)
    initial = evaluate_with_variables(spec, variables_to_mod(signed_variables, prime))
    if initial is None:
        raise DiscoveryError(
            f"{spec['name']}: assignment {assignment} is not a counterexample"
        )
    positions = [
        (variable_index, blade)
        for variable_index, variable in enumerate(signed_variables)
        for blade, coefficient in enumerate(variable)
        if coefficient
    ]
    changed = True
    while changed:
        changed = False
        for variable_index, blade in list(positions):
            current = signed_variables[variable_index][blade]
            if current == 0:
                continue
            signed_variables[variable_index][blade] = 0
            mismatch = evaluate_with_variables(
                spec, variables_to_mod(signed_variables, prime)
            )
            if mismatch is not None:
                changed = True
            else:
                signed_variables[variable_index][blade] = current
        positions = [
            (variable_index, blade)
            for variable_index, variable in enumerate(signed_variables)
            for blade, coefficient in enumerate(variable)
            if coefficient
        ]
    for variable_index, blade in positions:
        current = signed_variables[variable_index][blade]
        sign = 1 if current > 0 else -1
        for magnitude in range(1, abs(current)):
            signed_variables[variable_index][blade] = sign * magnitude
            mismatch = evaluate_with_variables(
                spec, variables_to_mod(signed_variables, prime)
            )
            if mismatch is not None:
                current = sign * magnitude
                break
        signed_variables[variable_index][blade] = current
    final_mismatch = evaluate_with_variables(
        spec, variables_to_mod(signed_variables, prime)
    )
    if final_mismatch is None:
        raise AssertionError("witness minimization destroyed the counterexample")
    return signed_variables, final_mismatch, witness_score(signed_variables)


def witness_variables_json(
    spec: dict[str, Any], variables: Sequence[Sequence[int]]
) -> list[dict[str, Any]]:
    dimension = spec["dimension"]
    output: list[dict[str, Any]] = []
    for variable_spec, coefficients in zip(spec["variables"], variables):
        terms = [
            {
                "blade": blade,
                "blade_label": blade_label(blade, dimension),
                "coefficient": coefficient,
            }
            for blade, coefficient in enumerate(coefficients)
            if coefficient
        ]
        output.append({"name": variable_spec["name"], "terms": terms})
    return output


def reduce_results(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry_by_name = {entry["name"]: entry for entry in manifest["entries"]}
    spec_by_name: dict[str, dict[str, Any]] = {}
    base_dir = args.manifest.parent
    for entry in manifest["entries"]:
        spec_by_name[entry["name"]] = load_spec(base_dir / entry["file"])
    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8-sig")))
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        name = row["identity"]
        if name not in entry_by_name:
            failures.append(f"CSV identity not found in manifest: {name}")
            continue
        entry = entry_by_name[name]
        found = row["found_counterexample"].strip().lower() == "true"
        record: dict[str, Any] = {
            "name": name,
            "variant_id": entry["variant_id"],
            "source_identity": entry["source_identity"],
            "prime": int(row["prime"]),
            "expected": row["expected"],
            "polynomial_zero": entry["polynomial_zero"],
            "polynomial_hash": entry["polynomial_hash"],
            "kernel_us": float(row["kernel_us"]),
            "assignments_per_second": float(row["assignments_per_second"]),
            "found_counterexample": found,
            "gpu_result": row["result"],
        }
        if found:
            assignment = int(row["witness_assignment"])
            spec = spec_by_name[name]
            mismatch = evaluate_assignment(spec, assignment)
            if mismatch is None:
                failures.append(f"{name}: GPU witness does not reproduce")
            else:
                gpu_blade = int(row["witness_blade"])
                gpu_lhs = int(row["witness_lhs"])
                gpu_rhs = int(row["witness_rhs"])
                if (
                    mismatch.blade != gpu_blade
                    or mismatch.lhs != gpu_lhs
                    or mismatch.rhs != gpu_rhs
                ):
                    failures.append(
                        f"{name}: host witness differs from GPU row "
                        f"({mismatch} vs {gpu_blade},{gpu_lhs},{gpu_rhs})"
                    )
                reduced_variables, reduced_mismatch, score = minimize_witness(
                    spec, assignment
                )
                record["gpu_witness"] = {
                    "assignment": assignment,
                    "blade": mismatch.blade,
                    "blade_label": blade_label(mismatch.blade, spec["dimension"]),
                    "lhs": mismatch.lhs,
                    "rhs": mismatch.rhs,
                }
                record["reduced_witness"] = {
                    "blade": reduced_mismatch.blade,
                    "blade_label": blade_label(
                        reduced_mismatch.blade, spec["dimension"]
                    ),
                    "lhs": reduced_mismatch.lhs,
                    "rhs": reduced_mismatch.rhs,
                    "score": {
                        "nonzero_coefficients": score[0],
                        "l1_magnitude": score[1],
                        "highest_blade": score[2],
                    },
                    "variables": witness_variables_json(spec, reduced_variables),
                }
        records.append(record)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["variant_id"], []).append(record)
    variants: list[dict[str, Any]] = []
    for variant_id in sorted(grouped):
        group = sorted(grouped[variant_id], key=lambda item: item["prime"])
        variants.append(
            {
                "variant_id": variant_id,
                "source_identity": group[0]["source_identity"],
                "polynomial_zero": group[0]["polynomial_zero"],
                "polynomial_hash": group[0]["polynomial_hash"],
                "primes_tested": [item["prime"] for item in group],
                "counterexample_primes": [
                    item["prime"] for item in group if item["found_counterexample"]
                ],
                "surviving_primes": [
                    item["prime"] for item in group if not item["found_counterexample"]
                ],
                "all_gpu_rows_passed": all(
                    item["gpu_result"] == "pass" for item in group
                ),
                "mean_assignments_per_second": sum(
                    item["assignments_per_second"] for item in group
                )
                / len(group),
            }
        )
    report = {
        "schema_version": 1,
        "manifest": args.manifest.as_posix(),
        "csv": args.csv.as_posix(),
        "records": records,
        "variants": variants,
        "validation": "pass" if not failures else "fail",
        "failures": failures,
    }
    write_json(args.output_json, report)
    markdown = [
        "# Geometric identity discovery result",
        "",
        f"- finite-field rows: {len(records)}",
        f"- structural variants: {len(variants)}",
        f"- exact GPU witnesses: {sum(1 for record in records if record['found_counterexample'])}",
        f"- formal polynomial identities: {sum(1 for variant in variants if variant['polynomial_zero'])}",
        f"- validation: **{report['validation'].upper()}**",
        "",
        "| variant | source | polynomial | primes | counterexample primes | survivor primes | mean assignments/s |",
        "|---|---|---|---|---|---|---:|",
    ]
    for variant in variants:
        markdown.append(
            f"| `{variant['variant_id']}` | `{variant['source_identity']}` | "
            f"{'zero' if variant['polynomial_zero'] else 'nonzero'} | "
            f"{','.join(map(str, variant['primes_tested']))} | "
            f"{','.join(map(str, variant['counterexample_primes'])) or 'none'} | "
            f"{','.join(map(str, variant['surviving_primes'])) or 'none'} | "
            f"{variant['mean_assignments_per_second']:.3f} |"
        )
    if failures:
        markdown.extend(["", "## Failures", ""])
        markdown.extend(f"- {failure}" for failure in failures)
    args.markdown_out.write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )
    if failures:
        print("VALIDATION: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("VALIDATION: PASS")
    return 0


def analyze_identity(args: argparse.Namespace) -> int:
    spec = load_spec(args.identity)
    polynomial = extract_polynomial(spec, args.term_limit)
    first_assignment, mismatch = precheck(spec, args.precheck_assignments)
    result = {
        "identity": args.identity.as_posix(),
        "name": spec["name"],
        "polynomial": polynomial,
        "precheck_assignments": args.precheck_assignments,
        "precheck_witness": (
            None
            if mismatch is None
            else {
                "assignment": first_assignment,
                "blade": mismatch.blade,
                "blade_label": blade_label(mismatch.blade, spec["dimension"]),
                "lhs": mismatch.lhs,
                "rhs": mismatch.rhs,
            }
        ),
    }
    if args.output:
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "name": spec["name"],
                "polynomial_zero": polynomial["zero"],
                "polynomial_terms": polynomial["total_terms"],
                "polynomial_hash": polynomial["canonical_hash"],
                "precheck_witness": result["precheck_witness"],
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-prime mutation, polynomial, and witness-reduction layer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-corpus")
    build.add_argument("--source", type=Path, action="append", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--prime", type=int, action="append")
    build.add_argument("--precheck-assignments", type=int, default=4096)
    build.add_argument("--max-mutations", type=int, default=12)
    build.add_argument("--term-limit", type=int, default=200_000)
    build.add_argument("--mutate-counterexamples", action="store_true")
    build.add_argument("--clean", action="store_true")
    build.set_defaults(function=build_corpus)

    reduce_parser = subparsers.add_parser("reduce-results")
    reduce_parser.add_argument("--manifest", type=Path, required=True)
    reduce_parser.add_argument("--csv", type=Path, required=True)
    reduce_parser.add_argument("--output-json", type=Path, required=True)
    reduce_parser.add_argument("--markdown-out", type=Path, required=True)
    reduce_parser.set_defaults(function=reduce_results)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--identity", type=Path, required=True)
    analyze.add_argument("--precheck-assignments", type=int, default=4096)
    analyze.add_argument("--term-limit", type=int, default=200_000)
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(function=analyze_identity)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if hasattr(args, "precheck_assignments") and args.precheck_assignments < 0:
            raise DiscoveryError("precheck assignments cannot be negative")
        if hasattr(args, "max_mutations") and args.max_mutations < 0:
            raise DiscoveryError("max mutations cannot be negative")
        if hasattr(args, "term_limit") and args.term_limit < 1:
            raise DiscoveryError("term limit must be positive")
        return int(args.function(args))
    except (DiscoveryError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
