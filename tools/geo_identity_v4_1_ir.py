#!/usr/bin/env python3
"""V4.1 fixed-blade extension over the accepted exact identity backends.

This module intentionally leaves the accepted V1-V4 modules unchanged. It patches their
public expression hooks at import time, adding one strictly checked node:

    {"fixed_blade": {"blade": <int>, "coefficient": <int>}}

The coefficient is an exact characteristic-zero integer in the symbolic backend and is
normalized modulo the statement prime in finite-field host/CUDA execution.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

try:
    import geo_identity_discovery as exact
    import geo_identity_compiler as compiler
except ModuleNotFoundError:
    from tools import geo_identity_discovery as exact
    from tools import geo_identity_compiler as compiler


_EXACT_VALIDATE = exact._validate_expression
_EXACT_EVALUATE = exact.evaluate_expression
_EXACT_POLYNOMIAL = exact.polynomial_expression
_COMPILER_PARSE = compiler.Builder.parse
_COMPILER_EMIT_IDENTITY = compiler._emit_identity


def parse_fixed_blade(raw: object, dimension: int, context: str) -> tuple[int, int] | None:
    if not isinstance(raw, dict) or "fixed_blade" not in raw:
        return None
    if set(raw) != {"fixed_blade"}:
        raise ValueError(f"{context}: fixed_blade expression has extra fields")
    payload = raw["fixed_blade"]
    if not isinstance(payload, dict) or set(payload) != {"blade", "coefficient"}:
        raise ValueError(
            f"{context}.fixed_blade must contain exactly blade and coefficient"
        )
    blade = payload["blade"]
    coefficient = payload["coefficient"]
    if not isinstance(blade, int) or isinstance(blade, bool):
        raise ValueError(f"{context}.fixed_blade.blade must be an integer")
    if not 0 <= blade < (1 << dimension):
        raise ValueError(
            f"{context}.fixed_blade.blade must be in [0,{(1 << dimension) - 1}]"
        )
    if not isinstance(coefficient, int) or isinstance(coefficient, bool):
        raise ValueError(f"{context}.fixed_blade.coefficient must be an integer")
    return blade, coefficient


def _validate_expression_extended(
    expression: object,
    variables: set[str],
    dimension: int,
    context: str,
) -> None:
    try:
        parsed = parse_fixed_blade(expression, dimension, context)
    except ValueError as exc:
        raise exact.DiscoveryError(str(exc)) from exc
    if parsed is not None:
        return
    _EXACT_VALIDATE(expression, variables, dimension, context)


def _evaluate_expression_extended(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variables: Sequence[Sequence[int]],
    variable_index: dict[str, int],
    memo: dict[str, list[int]] | None = None,
) -> list[int]:
    parsed = parse_fixed_blade(expression, spec["dimension"], "expression")
    if parsed is None:
        return _EXACT_EVALUATE(spec, expression, variables, variable_index, memo)
    blade, coefficient = parsed
    value = [0] * (1 << spec["dimension"])
    value[blade] = coefficient % spec.get("prime", 65521)
    return value


def _polynomial_expression_extended(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variable_values: dict[str, exact.PolyMV],
    term_limit: int,
    memo: dict[str, exact.PolyMV] | None = None,
) -> exact.PolyMV:
    parsed = parse_fixed_blade(expression, spec["dimension"], "expression")
    if parsed is None:
        return _EXACT_POLYNOMIAL(spec, expression, variable_values, term_limit, memo)
    blade, coefficient = parsed
    value: exact.PolyMV = [{} for _ in range(1 << spec["dimension"])]
    if coefficient:
        value[blade] = {(): coefficient}
    return value


def _builder_fixed_blade(self: compiler.Builder, blade: int, coefficient: int) -> int:
    if not 0 <= blade < self.blade_count:
        raise compiler.IdentityError(
            f"fixed blade {blade} is outside dimension {self.dimension}"
        )
    normalized = coefficient % self.prime
    if normalized == 0:
        return self.scalar(0)
    return self.intern(
        op="fixed_blade",
        value=normalized,
        grade=blade,
        support_mask=1 << blade,
        key=("fixed_blade", blade, normalized),
    )


def _builder_parse_extended(
    self: compiler.Builder, raw: object, context: str
) -> int:
    try:
        parsed = parse_fixed_blade(raw, self.dimension, context)
    except ValueError as exc:
        raise compiler.IdentityError(str(exc)) from exc
    if parsed is not None:
        return _builder_fixed_blade(self, *parsed)
    return _COMPILER_PARSE(self, raw, context)


def evaluate_identity(identity: compiler.Identity, assignment: int) -> tuple[bool, int, int, int]:
    blade_count = 1 << identity.dimension
    variables = compiler.generate_assignment(identity, assignment)
    values: list[list[int]] = []
    for node in identity.nodes:
        if node.op == "fixed_blade":
            assert node.grade is not None and node.value is not None
            value = [0] * blade_count
            value[node.grade] = node.value
        elif node.op == "var":
            assert node.variable is not None
            value = list(variables[node.variable])
        elif node.op == "scalar":
            assert node.value is not None
            value = [0] * blade_count
            value[0] = node.value
        elif node.op == "add":
            value = [0] * blade_count
            for argument in node.args:
                for blade in range(blade_count):
                    value[blade] = (value[blade] + values[argument][blade]) % identity.prime
        elif node.op == "neg":
            value = [(-coefficient) % identity.prime for coefficient in values[node.args[0]]]
        elif node.op == "scale":
            assert node.value is not None
            value = [
                (node.value * coefficient) % identity.prime
                for coefficient in values[node.args[0]]
            ]
        elif node.op == "gp":
            value = compiler._gp(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "wedge":
            value = compiler._wedge(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "reverse":
            value = [0] * blade_count
            for blade, coefficient in enumerate(values[node.args[0]]):
                grade = blade.bit_count()
                sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                value[blade] = sign * coefficient % identity.prime
        elif node.op == "grade":
            assert node.grade is not None
            value = [
                coefficient if blade.bit_count() == node.grade else 0
                for blade, coefficient in enumerate(values[node.args[0]])
            ]
        else:
            raise AssertionError(node.op)
        values.append(value)
    lhs = values[identity.lhs]
    rhs = values[identity.rhs]
    for blade, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return False, blade, left, right
    return True, 0, 0, 0


def _emit_identity_extended(identity: compiler.Identity, index: int) -> list[str]:
    fixed: dict[int, int] = {}
    shadow_nodes = []
    for node_index, node in enumerate(identity.nodes):
        if node.op == "fixed_blade":
            assert node.grade is not None
            fixed[node_index] = node.grade
            shadow_nodes.append(replace(node, op="scalar"))
        else:
            shadow_nodes.append(node)
    shadow = replace(identity, nodes=tuple(shadow_nodes))
    lines = _COMPILER_EMIT_IDENTITY(shadow, index)
    for node_index, blade in fixed.items():
        old = f"        nodes[{node_index}].c[0] = {identity.nodes[node_index].value};"
        new = f"        nodes[{node_index}].c[{blade}] = {identity.nodes[node_index].value};"
        matches = [line_index for line_index, line in enumerate(lines) if line == old]
        if len(matches) != 1:
            raise compiler.IdentityError(
                f"unable to rewrite fixed-blade node {node_index}; scalar marker count={len(matches)}"
            )
        lines[matches[0]] = new
    return lines


def install() -> None:
    exact._validate_expression = _validate_expression_extended
    exact.evaluate_expression = _evaluate_expression_extended
    exact.polynomial_expression = _polynomial_expression_extended
    compiler.Builder.parse = _builder_parse_extended
    compiler._emit_identity = _emit_identity_extended


install()

validate_spec = exact.validate_spec
extract_polynomial = exact.extract_polynomial
load_identity = compiler.load_identity
load_corpus = compiler.load_corpus
emit_header = compiler.emit_header
IdentityError = compiler.IdentityError
DiscoveryError = exact.DiscoveryError
