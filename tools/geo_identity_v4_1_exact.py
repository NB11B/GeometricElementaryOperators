#!/usr/bin/env python3
"""Fixed-blade extension for the exact geometric identity backend.

This module installs a narrowly scoped ``fixed_blade`` expression node into the
existing exact evaluator and polynomial extractor without changing the accepted
V1-V4 modules in place.

Node contract::

    {"fixed_blade": {"blade": 15, "coefficient": 1}}

The coefficient is a nonzero characteristic-zero integer. Finite-field
execution reduces it modulo the statement prime. The blade index must satisfy
``0 <= blade < 2**dimension``. No additional fields are accepted.
"""
from __future__ import annotations

from typing import Any, Sequence

import geo_identity_discovery as exact

_ORIGINAL_VALIDATE_EXPRESSION = exact._validate_expression
_ORIGINAL_EVALUATE_EXPRESSION = exact.evaluate_expression
_ORIGINAL_POLYNOMIAL_EXPRESSION = exact.polynomial_expression
_INSTALLED = False


def parse_fixed_blade(expression: object, dimension: int, context: str) -> tuple[int, int]:
    if not isinstance(expression, dict) or set(expression) != {"fixed_blade"}:
        raise exact.DiscoveryError(f"{context} must contain only fixed_blade")
    payload = expression["fixed_blade"]
    if not isinstance(payload, dict) or set(payload) != {"blade", "coefficient"}:
        raise exact.DiscoveryError(
            f"{context}.fixed_blade must contain exactly blade and coefficient"
        )
    blade = payload["blade"]
    coefficient = payload["coefficient"]
    if not isinstance(blade, int) or isinstance(blade, bool):
        raise exact.DiscoveryError(f"{context}.fixed_blade.blade must be an integer")
    if not 0 <= blade < (1 << dimension):
        raise exact.DiscoveryError(
            f"{context}.fixed_blade.blade must be in [0,{(1 << dimension) - 1}]"
        )
    if not isinstance(coefficient, int) or isinstance(coefficient, bool):
        raise exact.DiscoveryError(
            f"{context}.fixed_blade.coefficient must be an integer"
        )
    if coefficient == 0:
        raise exact.DiscoveryError(
            f"{context}.fixed_blade.coefficient must be nonzero"
        )
    return blade, coefficient


def _validate_expression(
    expression: object,
    variables: set[str],
    dimension: int,
    context: str,
) -> None:
    if isinstance(expression, dict) and "fixed_blade" in expression:
        parse_fixed_blade(expression, dimension, context)
        return
    _ORIGINAL_VALIDATE_EXPRESSION(expression, variables, dimension, context)


def evaluate_expression(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variables: Sequence[Sequence[int]],
    variable_index: dict[str, int],
    memo: dict[str, list[int]] | None = None,
) -> list[int]:
    if memo is None:
        memo = {}
    key = exact.canonical_json(expression)
    cached = memo.get(key)
    if cached is not None:
        return list(cached)
    if "fixed_blade" not in expression:
        return _ORIGINAL_EVALUATE_EXPRESSION(
            spec, expression, variables, variable_index, memo
        )
    blade, coefficient = parse_fixed_blade(
        expression, int(spec["dimension"]), "expression"
    )
    value = [0] * (1 << int(spec["dimension"]))
    value[blade] = coefficient % int(spec.get("prime", 65521))
    memo[key] = list(value)
    return value


def polynomial_expression(
    spec: dict[str, Any],
    expression: dict[str, Any],
    variable_values: dict[str, Any],
    term_limit: int,
    memo: dict[str, Any] | None = None,
):
    if memo is None:
        memo = {}
    key = exact.canonical_json(expression)
    cached = memo.get(key)
    if cached is not None:
        import copy

        return copy.deepcopy(cached)
    if "fixed_blade" not in expression:
        return _ORIGINAL_POLYNOMIAL_EXPRESSION(
            spec, expression, variable_values, term_limit, memo
        )
    blade, coefficient = parse_fixed_blade(
        expression, int(spec["dimension"]), "expression"
    )
    value = [{} for _ in range(1 << int(spec["dimension"]))]
    value[blade] = {(): coefficient}
    memo[key] = value
    import copy

    return copy.deepcopy(value)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    exact._validate_expression = _validate_expression
    exact.evaluate_expression = evaluate_expression
    exact.polynomial_expression = polynomial_expression
    _INSTALLED = True


install()

# Public aliases used by the V4.1 tools.
DiscoveryError = exact.DiscoveryError
validate_spec = exact.validate_spec
load_spec = exact.load_spec
extract_polynomial = exact.extract_polynomial
evaluate_assignment = exact.evaluate_assignment
evaluate_with_variables = exact.evaluate_with_variables
precheck = exact.precheck
canonical_json = exact.canonical_json
write_json = exact.write_json
is_prime = exact.is_prime
active_blades = exact.active_blades
blade_label = exact.blade_label
gp_sign = exact.gp_sign
wedge_sign = exact.wedge_sign
poly_add = exact.poly_add
poly_scale = exact.poly_scale
poly_mul = exact.poly_mul
polymv_add = exact.polymv_add
polymv_scale = exact.polymv_scale
polymv_gp = exact.polymv_gp
polymv_wedge = exact.polymv_wedge
