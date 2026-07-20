#!/usr/bin/env python3
"""Grammar-bounded exact identity discovery for the geometric engine.

V3 generates Clifford expressions from bounded grammars, classifies every
expression by an exact blade-wise integer polynomial, groups exact and scalar-
multiple equivalence classes, ranks nontrivial relations, and emits a v1-
compatible finite-field corpus for independent host/CUDA validation.

The symbolic result is scoped to the declared dimension, diagonal signature,
and variable grade supports. A generated finite-field witness is exact in its
field; absence of a witness is not used as the proof mechanism because selected
identity relations are required to have a zero integer difference polynomial.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


def _load_exact_module():
    try:
        import geo_identity_discovery as module  # type: ignore
        return module
    except ModuleNotFoundError:
        path = Path(__file__).with_name("geo_identity_discovery.py")
        spec = importlib.util.spec_from_file_location("geo_identity_discovery", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


exact = _load_exact_module()


class GrammarError(ValueError):
    """Raised when a grammar or generated relation violates the v3 contract."""


COMMUTATIVE_OPS = {"add"}
BINARY_OPS = {"add", "sub", "gp", "wedge", "commutator"}
UNARY_OPS = {"neg", "reverse"}
SUPPORTED_FAMILIES = {
    "involution",
    "reverse_product",
    "vector_product_decomposition",
    "commutator_antisymmetry",
    "jacobi_cycle",
}
DEFAULT_PRIMES = (65521, 65519, 65497, 32749)


@dataclass(frozen=True)
class GrammarConfig:
    name: str
    description: str
    dimension: int
    signature: tuple[int, ...]
    prime: int
    coefficient_bound: int
    seed: int
    variables: tuple[dict[str, Any], ...]
    max_cost: int
    max_expressions: int
    max_per_cost: int
    max_representatives_per_polynomial: int
    candidate_multiplier: int
    unary_ops: tuple[str, ...]
    binary_ops: tuple[str, ...]
    scales: tuple[int, ...]
    grades: tuple[int, ...]
    constants: tuple[int, ...]
    families: tuple[dict[str, Any], ...]
    max_relations: int
    max_controls: int
    max_pairs_per_class: int
    max_relations_per_primitive_class: int
    min_relation_variables: int
    require_geometric_operator: bool
    term_limit: int


@dataclass
class ExpressionRecord:
    expression: dict[str, Any]
    key: str
    label: str
    cost: int
    top_op: str
    operators: tuple[tuple[str, int], ...]
    variables: tuple[str, ...]
    exact_hash: str
    primitive_hash: str
    zero: bool
    total_terms: int
    maximum_degree: int
    content_gcd: int
    exact_blades: list[dict[str, Any]]
    primitive_blades: list[dict[str, Any]]


@dataclass
class Relation:
    grammar: str
    relation_id: str
    kind: str
    score: float
    lhs: ExpressionRecord
    rhs: ExpressionRecord | None
    lhs_scale: int
    rhs_scale: int
    polynomial_hash: str
    primitive_hash: str
    certificate_note: str


@dataclass
class Control:
    relation_id: str
    control_id: str
    mutation: dict[str, Any]
    specification: dict[str, Any]
    polynomial: dict[str, Any]
    witness_assignment: int
    witness: Any


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sanitize_identifier(value: str) -> str:
    output = "".join(character if character.isalnum() else "_" for character in value)
    output = output.strip("_") or "generated"
    if output[0].isdigit():
        output = "g_" + output
    return output


def expression_cost(expression: dict[str, Any]) -> int:
    if "var" in expression or "scalar" in expression:
        return 1
    if "arg" in expression:
        return 1 + expression_cost(expression["arg"])
    if "args" in expression:
        return 1 + sum(expression_cost(argument) for argument in expression["args"])
    raise GrammarError(f"invalid expression: {expression!r}")


def expression_top_op(expression: dict[str, Any]) -> str:
    if "var" in expression:
        return "var"
    if "scalar" in expression:
        return "scalar"
    return str(expression.get("op", "unknown"))


def expression_operators(expression: dict[str, Any]) -> Counter[str]:
    output: Counter[str] = Counter()
    if "op" not in expression:
        return output
    output[str(expression["op"])] += 1
    if "arg" in expression:
        output.update(expression_operators(expression["arg"]))
    for argument in expression.get("args", []):
        output.update(expression_operators(argument))
    return output


def expression_variables(expression: dict[str, Any]) -> set[str]:
    if "var" in expression:
        return {str(expression["var"])}
    output: set[str] = set()
    if "arg" in expression:
        output.update(expression_variables(expression["arg"]))
    for argument in expression.get("args", []):
        output.update(expression_variables(argument))
    return output


def canonicalize_expression(expression: dict[str, Any]) -> dict[str, Any]:
    if "var" in expression:
        return {"var": expression["var"]}
    if "scalar" in expression:
        return {"scalar": int(expression["scalar"])}
    op = expression.get("op")
    if op in UNARY_OPS:
        return {"op": op, "arg": canonicalize_expression(expression["arg"])}
    if op == "scale":
        return {
            "op": "scale",
            "value": int(expression["value"]),
            "arg": canonicalize_expression(expression["arg"]),
        }
    if op == "grade":
        return {
            "op": "grade",
            "grade": int(expression["grade"]),
            "arg": canonicalize_expression(expression["arg"]),
        }
    if op in BINARY_OPS:
        arguments = [canonicalize_expression(argument) for argument in expression["args"]]
        if op in COMMUTATIVE_OPS:
            arguments.sort(key=canonical_json)
        return {"op": op, "args": arguments}
    raise GrammarError(f"unsupported expression operation: {op!r}")


def expression_label(expression: dict[str, Any]) -> str:
    if "var" in expression:
        return str(expression["var"])
    if "scalar" in expression:
        return str(expression["scalar"])
    op = expression["op"]
    if op == "neg":
        return f"-({expression_label(expression['arg'])})"
    if op == "reverse":
        return f"~({expression_label(expression['arg'])})"
    if op == "scale":
        return f"{expression['value']}*({expression_label(expression['arg'])})"
    if op == "grade":
        return f"<{expression_label(expression['arg'])}>_{expression['grade']}"
    if op == "add" and len(expression["args"]) > 2:
        return "(" + " + ".join(expression_label(argument) for argument in expression["args"]) + ")"
    left = expression_label(expression["args"][0])
    right = expression_label(expression["args"][1])
    symbol = {
        "add": "+",
        "sub": "-",
        "gp": "*",
        "wedge": "^",
        "commutator": "[,]",
    }[op]
    if op == "commutator":
        return f"[{left},{right}]"
    return f"({left} {symbol} {right})"


def load_grammar(path: Path) -> GrammarConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GrammarError(f"unable to read {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise GrammarError(f"{path}: schema_version must be 1")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise GrammarError(f"{path}: name must be non-empty")
    dimension = raw.get("dimension")
    signature = raw.get("signature")
    variables = raw.get("variables")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or not 1 <= dimension <= 6:
        raise GrammarError(f"{path}: dimension must be in [1,6]")
    if (
        not isinstance(signature, list)
        or len(signature) != dimension
        or any(value not in (-1, 1) for value in signature)
    ):
        raise GrammarError(f"{path}: signature must contain {dimension} values from -1,+1")
    if not isinstance(variables, list) or not variables:
        raise GrammarError(f"{path}: variables must be non-empty")
    variable_names: set[str] = set()
    normalized_variables: list[dict[str, Any]] = []
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            raise GrammarError(f"{path}: variable {index} must be an object")
        variable_name = variable.get("name")
        grades = variable.get("grades")
        if not isinstance(variable_name, str) or not variable_name or variable_name in variable_names:
            raise GrammarError(f"{path}: invalid or duplicate variable name at {index}")
        if (
            not isinstance(grades, list)
            or not grades
            or any(
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or not 0 <= grade <= dimension
                for grade in grades
            )
        ):
            raise GrammarError(f"{path}: invalid grades for variable {variable_name}")
        variable_names.add(variable_name)
        normalized_variables.append({"name": variable_name, "grades": sorted(set(grades))})
    grammar = raw.get("grammar", {})
    if not isinstance(grammar, dict):
        raise GrammarError(f"{path}: grammar must be an object")
    operators = grammar.get("operators", {})
    if not isinstance(operators, dict):
        raise GrammarError(f"{path}: grammar.operators must be an object")
    unary = tuple(operators.get("unary", ("neg", "reverse")))
    binary = tuple(operators.get("binary", ("add", "sub", "gp", "wedge", "commutator")))
    if any(op not in UNARY_OPS for op in unary):
        raise GrammarError(f"{path}: unsupported unary operator")
    if any(op not in BINARY_OPS for op in binary):
        raise GrammarError(f"{path}: unsupported binary operator")
    scales = tuple(int(value) for value in operators.get("scales", (-2, 2)))
    grades = tuple(int(value) for value in operators.get("grades", tuple(range(dimension + 1))))
    if any(value == 0 for value in scales):
        raise GrammarError(f"{path}: scale factors cannot contain zero")
    if any(not 0 <= grade <= dimension for grade in grades):
        raise GrammarError(f"{path}: grade operator is outside dimension")
    constants = tuple(int(value) for value in grammar.get("constants", (0,)))
    families_raw = grammar.get("families", [])
    if not isinstance(families_raw, list):
        raise GrammarError(f"{path}: families must be a list")
    families: list[dict[str, Any]] = []
    for family in families_raw:
        if isinstance(family, str):
            family = {"kind": family}
        if not isinstance(family, dict) or family.get("kind") not in SUPPORTED_FAMILIES:
            raise GrammarError(f"{path}: unsupported family {family!r}")
        families.append(copy.deepcopy(family))
    prime = int(raw.get("prime", 65521))
    if not exact.is_prime(prime):
        raise GrammarError(f"{path}: prime is not prime")

    def positive_int(key: str, default: int) -> int:
        value = grammar.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise GrammarError(f"{path}: grammar.{key} must be positive")
        return value

    max_controls = grammar.get("max_controls", 8)
    min_relation_variables = grammar.get("min_relation_variables", 1)
    if not isinstance(max_controls, int) or isinstance(max_controls, bool) or max_controls < 0:
        raise GrammarError(f"{path}: grammar.max_controls must be non-negative")
    if (
        not isinstance(min_relation_variables, int)
        or isinstance(min_relation_variables, bool)
        or min_relation_variables < 1
        or min_relation_variables > len(normalized_variables)
    ):
        raise GrammarError(f"{path}: grammar.min_relation_variables is invalid")

    return GrammarConfig(
        name=name,
        description=str(raw.get("description", "")),
        dimension=dimension,
        signature=tuple(int(value) for value in signature),
        prime=prime,
        coefficient_bound=int(raw.get("coefficient_bound", 3)),
        seed=int(raw.get("seed", 0x243F6A88)),
        variables=tuple(normalized_variables),
        max_cost=positive_int("max_cost", 6),
        max_expressions=positive_int("max_expressions", 1200),
        max_per_cost=positive_int("max_per_cost", 300),
        max_representatives_per_polynomial=positive_int(
            "max_representatives_per_polynomial", 5
        ),
        candidate_multiplier=positive_int("candidate_multiplier", 12),
        unary_ops=unary,
        binary_ops=binary,
        scales=scales,
        grades=grades,
        constants=constants,
        families=tuple(families),
        max_relations=positive_int("max_relations", 24),
        max_controls=max_controls,
        max_pairs_per_class=positive_int("max_pairs_per_class", 32),
        max_relations_per_primitive_class=positive_int(
            "max_relations_per_primitive_class", 3
        ),
        min_relation_variables=min_relation_variables,
        require_geometric_operator=bool(grammar.get("require_geometric_operator", True)),
        term_limit=positive_int("term_limit", 200_000),
    )


class SymbolicClassifier:
    def __init__(self, grammar: GrammarConfig) -> None:
        self.grammar = grammar
        self.spec = {
            "schema_version": 1,
            "name": f"{grammar.name}_expression",
            "description": grammar.description,
            "expected": "identity",
            "dimension": grammar.dimension,
            "signature": list(grammar.signature),
            "prime": grammar.prime,
            "coefficient_bound": grammar.coefficient_bound,
            "seed": grammar.seed,
            "variables": [copy.deepcopy(variable) for variable in grammar.variables],
            "lhs": {"scalar": 0},
            "rhs": {"scalar": 0},
        }
        exact.validate_spec(self.spec)
        blade_count = 1 << grammar.dimension
        self.symbols: list[dict[str, Any]] = []
        self.variable_values: dict[str, Any] = {}
        symbol_id = 0
        for variable in grammar.variables:
            value = [{} for _ in range(blade_count)]
            for blade in exact.active_blades(grammar.dimension, variable["grades"]):
                value[blade] = {(symbol_id,): 1}
                self.symbols.append(
                    {
                        "id": symbol_id,
                        "variable": variable["name"],
                        "blade": blade,
                        "blade_label": exact.blade_label(blade, grammar.dimension),
                        "name": f"{variable['name']}_{exact.blade_label(blade, grammar.dimension)}",
                    }
                )
                symbol_id += 1
            self.variable_values[variable["name"]] = value
        self.memo: dict[str, Any] = {}
        self.cache: dict[str, ExpressionRecord] = {}

    def classify(self, expression: dict[str, Any]) -> ExpressionRecord:
        expression = canonicalize_expression(expression)
        key = canonical_json(expression)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        exact._validate_expression(
            expression,
            {variable["name"] for variable in self.grammar.variables},
            self.grammar.dimension,
            f"{self.grammar.name}.expression",
        )
        polymv = exact.polynomial_expression(
            self.spec,
            expression,
            self.variable_values,
            self.grammar.term_limit,
            self.memo,
        )
        metadata = polynomial_metadata(
            polymv,
            dimension=self.grammar.dimension,
            signature=self.grammar.signature,
            symbols=self.symbols,
        )
        operators = expression_operators(expression)
        record = ExpressionRecord(
            expression=expression,
            key=key,
            label=expression_label(expression),
            cost=expression_cost(expression),
            top_op=expression_top_op(expression),
            operators=tuple(sorted(operators.items())),
            variables=tuple(sorted(expression_variables(expression))),
            exact_hash=metadata["exact_hash"],
            primitive_hash=metadata["primitive_hash"],
            zero=metadata["zero"],
            total_terms=metadata["total_terms"],
            maximum_degree=metadata["maximum_degree"],
            content_gcd=metadata["content_gcd"],
            exact_blades=metadata["blades"],
            primitive_blades=metadata["primitive_blades"],
        )
        self.cache[key] = record
        return record


def polynomial_metadata(
    polymv: Sequence[dict[tuple[int, ...], int]],
    *,
    dimension: int,
    signature: Sequence[int],
    symbols: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    coefficients = [
        abs(coefficient)
        for polynomial in polymv
        for coefficient in polynomial.values()
        if coefficient
    ]
    content = 0
    for coefficient in coefficients:
        content = math.gcd(content, coefficient)
    primitive = copy.deepcopy(list(polymv))
    if content > 1:
        primitive = [
            {monomial: coefficient // content for monomial, coefficient in polynomial.items()}
            for polynomial in primitive
        ]
    leading: int | None = None
    for polynomial in primitive:
        if polynomial:
            first = sorted(polynomial)[0]
            leading = polynomial[first]
            break
    if leading is not None and leading < 0:
        primitive = [exact.poly_scale(polynomial, -1) for polynomial in primitive]
    blades = polynomial_rows(polymv, dimension)
    primitive_blades = polynomial_rows(primitive, dimension)
    exact_payload = {
        "dimension": dimension,
        "signature": list(signature),
        "symbols": list(symbols),
        "blades": blades,
    }
    primitive_payload = {
        "dimension": dimension,
        "signature": list(signature),
        "symbols": list(symbols),
        "primitive_blades": primitive_blades,
    }
    degrees = [len(monomial) for polynomial in polymv for monomial in polynomial]
    return {
        "zero": not blades,
        "total_terms": sum(len(polynomial) for polynomial in polymv),
        "maximum_degree": max(degrees, default=0),
        "content_gcd": content,
        "exact_hash": hashlib.sha256(canonical_json(exact_payload).encode()).hexdigest(),
        "primitive_hash": hashlib.sha256(canonical_json(primitive_payload).encode()).hexdigest(),
        "blades": blades,
        "primitive_blades": primitive_blades,
    }


def polynomial_rows(
    polymv: Sequence[dict[tuple[int, ...], int]], dimension: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for blade, polynomial in enumerate(polymv):
        if not polynomial:
            continue
        rows.append(
            {
                "blade": blade,
                "blade_label": exact.blade_label(blade, dimension),
                "terms": [
                    {"monomial": list(monomial), "coefficient": polynomial[monomial]}
                    for monomial in sorted(polynomial)
                ],
            }
        )
    return rows


def flat_coefficients(record: ExpressionRecord) -> dict[tuple[int, tuple[int, ...]], int]:
    output: dict[tuple[int, tuple[int, ...]], int] = {}
    for blade in record.exact_blades:
        for term in blade["terms"]:
            output[(int(blade["blade"]), tuple(int(value) for value in term["monomial"]))] = int(
                term["coefficient"]
            )
    return output


def relation_multipliers(left: ExpressionRecord, right: ExpressionRecord) -> tuple[int, int] | None:
    left_coefficients = flat_coefficients(left)
    right_coefficients = flat_coefficients(right)
    if not left_coefficients and not right_coefficients:
        return 1, 1
    if set(left_coefficients) != set(right_coefficients):
        return None
    first_key = sorted(left_coefficients)[0]
    ratio = Fraction(left_coefficients[first_key], right_coefficients[first_key])
    for key in left_coefficients:
        if Fraction(left_coefficients[key], right_coefficients[key]) != ratio:
            return None
    lhs_scale = ratio.denominator
    rhs_scale = ratio.numerator
    common = math.gcd(abs(lhs_scale), abs(rhs_scale))
    if common > 1:
        lhs_scale //= common
        rhs_scale //= common
    if lhs_scale < 0:
        lhs_scale = -lhs_scale
        rhs_scale = -rhs_scale
    return lhs_scale, rhs_scale


def scaled_expression(expression: dict[str, Any], factor: int) -> dict[str, Any]:
    if factor == 1:
        return copy.deepcopy(expression)
    return {"op": "scale", "value": factor, "arg": copy.deepcopy(expression)}


def family_expressions(grammar: GrammarConfig) -> list[dict[str, Any]]:
    names = [variable["name"] for variable in grammar.variables]
    output: list[dict[str, Any]] = []
    for family in grammar.families:
        kind = family["kind"]
        selected = family.get("variables", names)
        if not isinstance(selected, list) or any(name not in names for name in selected):
            raise GrammarError(f"{grammar.name}: invalid variables for family {kind}")
        variables = [{"var": name} for name in selected]
        if kind == "involution":
            for variable in variables:
                output.append({"op": "reverse", "arg": {"op": "reverse", "arg": variable}})
        elif kind == "reverse_product":
            if len(variables) < 2:
                raise GrammarError(f"{grammar.name}: reverse_product requires two variables")
            left, right = variables[:2]
            output.extend(
                [
                    {"op": "reverse", "arg": {"op": "gp", "args": [left, right]}},
                    {
                        "op": "gp",
                        "args": [
                            {"op": "reverse", "arg": right},
                            {"op": "reverse", "arg": left},
                        ],
                    },
                ]
            )
        elif kind == "vector_product_decomposition":
            if len(variables) < 2:
                raise GrammarError(f"{grammar.name}: vector_product_decomposition requires two variables")
            left, right = variables[:2]
            ab = {"op": "gp", "args": [left, right]}
            ba = {"op": "gp", "args": [right, left]}
            wedge = {"op": "wedge", "args": [left, right]}
            commutator = {"op": "commutator", "args": [left, right]}
            output.extend(
                [
                    ab,
                    ba,
                    wedge,
                    commutator,
                    {"op": "scale", "value": 2, "arg": wedge},
                    {"op": "sub", "args": [ab, ba]},
                    {"op": "add", "args": [ab, ba]},
                    {"op": "scale", "value": 2, "arg": {"op": "grade", "grade": 0, "arg": ab}},
                    {"op": "scale", "value": 2, "arg": {"op": "grade", "grade": 2, "arg": ab}},
                ]
            )
        elif kind == "commutator_antisymmetry":
            if len(variables) < 2:
                raise GrammarError(f"{grammar.name}: commutator_antisymmetry requires two variables")
            left, right = variables[:2]
            output.extend(
                [
                    {"op": "commutator", "args": [left, right]},
                    {"op": "neg", "arg": {"op": "commutator", "args": [right, left]}},
                ]
            )
        elif kind == "jacobi_cycle":
            if len(variables) < 3:
                raise GrammarError(f"{grammar.name}: jacobi_cycle requires three variables")
            a, b, c = variables[:3]
            first = {"op": "commutator", "args": [a, {"op": "commutator", "args": [b, c]}]}
            second = {"op": "commutator", "args": [b, {"op": "commutator", "args": [c, a]}]}
            third = {"op": "commutator", "args": [c, {"op": "commutator", "args": [a, b]}]}
            output.extend([first, second, third, {"op": "add", "args": [first, second, third]}])
        else:
            raise AssertionError(kind)
    return [canonicalize_expression(expression) for expression in output]


def enumerate_expressions(grammar: GrammarConfig) -> tuple[list[ExpressionRecord], dict[str, Any]]:
    classifier = SymbolicClassifier(grammar)
    by_cost: dict[int, list[ExpressionRecord]] = defaultdict(list)
    all_records: list[ExpressionRecord] = []
    seen_syntax: set[str] = set()
    representatives: dict[str, list[ExpressionRecord]] = defaultdict(list)
    generated_candidates = 0
    rejected_semantic = 0
    rejected_limits = 0

    def accept(expression: dict[str, Any], *, forced: bool = False) -> ExpressionRecord | None:
        nonlocal rejected_semantic, rejected_limits
        canonical = canonicalize_expression(expression)
        key = canonical_json(canonical)
        if key in seen_syntax:
            return classifier.cache.get(key)
        cost = expression_cost(canonical)
        if not forced and cost > grammar.max_cost:
            return None
        if len(all_records) >= grammar.max_expressions and not forced:
            rejected_limits += 1
            return None
        record = classifier.classify(canonical)
        existing = representatives[record.exact_hash]
        if len(existing) >= grammar.max_representatives_per_polynomial and not forced:
            rejected_semantic += 1
            return None
        seen_syntax.add(key)
        existing.append(record)
        all_records.append(record)
        by_cost[cost].append(record)
        return record

    for variable in grammar.variables:
        accept({"var": variable["name"]}, forced=True)
    for constant in grammar.constants:
        accept({"scalar": constant}, forced=True)
    for expression in family_expressions(grammar):
        accept(expression, forced=True)

    for cost in range(2, grammar.max_cost + 1):
        candidates: dict[str, dict[str, Any]] = {}
        candidate_limit = grammar.max_per_cost * grammar.candidate_multiplier

        def add_candidate(expression: dict[str, Any]) -> bool:
            nonlocal generated_candidates
            canonical = canonicalize_expression(expression)
            key = canonical_json(canonical)
            if key in seen_syntax or key in candidates:
                return True
            candidates[key] = canonical
            generated_candidates += 1
            return len(candidates) < candidate_limit

        previous = sorted(by_cost.get(cost - 1, []), key=lambda record: record.key)
        for record in previous:
            for op in grammar.unary_ops:
                if not add_candidate({"op": op, "arg": record.expression}):
                    break
            if len(candidates) >= candidate_limit:
                break
            for factor in grammar.scales:
                if not add_candidate({"op": "scale", "value": factor, "arg": record.expression}):
                    break
            if len(candidates) >= candidate_limit:
                break
            for grade in grammar.grades:
                if not add_candidate({"op": "grade", "grade": grade, "arg": record.expression}):
                    break
            if len(candidates) >= candidate_limit:
                break

        if len(candidates) < candidate_limit:
            stop = False
            for left_cost in range(1, cost - 1):
                right_cost = cost - 1 - left_cost
                left_records = sorted(by_cost.get(left_cost, []), key=lambda record: record.key)
                right_records = sorted(by_cost.get(right_cost, []), key=lambda record: record.key)
                for op in grammar.binary_ops:
                    for left in left_records:
                        for right in right_records:
                            if op in COMMUTATIVE_OPS and left.key > right.key:
                                continue
                            if not add_candidate(
                                {"op": op, "args": [left.expression, right.expression]}
                            ):
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break

        accepted_at_cost = 0
        for key in sorted(candidates):
            if accepted_at_cost >= grammar.max_per_cost:
                rejected_limits += len(candidates) - accepted_at_cost
                break
            record = accept(candidates[key])
            if record is not None:
                accepted_at_cost += 1
        by_cost[cost].sort(key=lambda record: record.key)

    all_records.sort(key=lambda record: (record.cost, record.key))
    stats = {
        "grammar": grammar.name,
        "expressions": len(all_records),
        "cost_histogram": {
            str(cost): len(records) for cost, records in sorted(by_cost.items())
        },
        "exact_polynomial_classes": len({record.exact_hash for record in all_records}),
        "primitive_polynomial_classes": len(
            {record.primitive_hash for record in all_records}
        ),
        "zero_expressions": sum(record.zero for record in all_records),
        "generated_candidates": generated_candidates,
        "rejected_semantic_cap": rejected_semantic,
        "rejected_expression_limits": rejected_limits,
    }
    return all_records, stats


def direct_wrapper(left: ExpressionRecord, right: ExpressionRecord) -> bool:
    for wrapper, other in ((left, right), (right, left)):
        expression = wrapper.expression
        if expression.get("op") in {"neg", "scale", "grade", "reverse"}:
            if canonical_json(expression.get("arg")) == other.key:
                return True
    return False


def relation_score(
    left: ExpressionRecord,
    right: ExpressionRecord | None,
    lhs_scale: int,
    rhs_scale: int,
    kind: str,
) -> float:
    if right is None:
        operator_distance = sum(count for _, count in left.operators)
        top_difference = 1
        total_cost = left.cost + 1
        wrapper_penalty = 0
    else:
        left_ops = Counter(dict(left.operators))
        right_ops = Counter(dict(right.operators))
        operator_distance = sum((left_ops - right_ops).values()) + sum(
            (right_ops - left_ops).values()
        )
        top_difference = int(left.top_op != right.top_op)
        total_cost = left.cost + right.cost
        wrapper_penalty = 30 if direct_wrapper(left, right) else 0
    variables = set(left.variables)
    if right is not None:
        variables.update(right.variables)
    binary_count = sum(count for op, count in left.operators if op in BINARY_OPS)
    if right is not None:
        binary_count += sum(count for op, count in right.operators if op in BINARY_OPS)
    wrapper_operation_count = sum(
        count for op, count in left.operators if op in {"neg", "scale"}
    )
    if right is not None:
        wrapper_operation_count += sum(
            count for op, count in right.operators if op in {"neg", "scale"}
        )
    kind_bonus = {"zero": 22, "exact": 30, "integer_multiple": 15}[kind]
    scale_penalty = abs(lhs_scale) + abs(rhs_scale) - 2
    return (
        120
        + 12 * len(variables)
        + 3 * binary_count
        + kind_bonus
        + 12 * top_difference
        + 3 * operator_distance
        - 4 * total_cost
        - 4 * scale_penalty
        - 5 * wrapper_operation_count
        - wrapper_penalty
    )


def discover_relations(
    grammar: GrammarConfig, records: Sequence[ExpressionRecord]
) -> list[Relation]:
    relations: list[Relation] = []
    zero_literal = next(
        (record for record in records if record.expression == {"scalar": 0}),
        None,
    )
    for record in records:
        if not record.zero or record.expression == {"scalar": 0}:
            continue
        if len(record.variables) < grammar.min_relation_variables:
            continue
        if grammar.require_geometric_operator and not any(
            op in {"gp", "wedge", "commutator"} for op, _ in record.operators
        ):
            continue
        score = relation_score(record, zero_literal, 1, 1, "zero")
        stable = hashlib.sha256(
            f"{grammar.name}|zero|{record.key}".encode("utf-8")
        ).hexdigest()[:16]
        relations.append(
            Relation(
                grammar=grammar.name,
                relation_id=f"{grammar.name}:z:{stable}",
                kind="zero",
                score=score,
                lhs=record,
                rhs=zero_literal,
                lhs_scale=1,
                rhs_scale=1,
                polynomial_hash=record.exact_hash,
                primitive_hash=record.primitive_hash,
                certificate_note="expression has an exact zero blade-wise polynomial",
            )
        )

    groups: dict[str, list[ExpressionRecord]] = defaultdict(list)
    for record in records:
        if not record.zero:
            groups[record.primitive_hash].append(record)
    for primitive_hash, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda record: (record.cost, record.key))
        pair_count = 0
        for left_index, right_index in itertools.combinations(range(len(ordered)), 2):
            if pair_count >= grammar.max_pairs_per_class:
                break
            left = ordered[left_index]
            right = ordered[right_index]
            if len(set(left.variables) | set(right.variables)) < grammar.min_relation_variables:
                continue
            if grammar.require_geometric_operator and not any(
                op in {"gp", "wedge", "commutator"}
                for op, _ in tuple(left.operators) + tuple(right.operators)
            ):
                continue
            multipliers = relation_multipliers(left, right)
            if multipliers is None:
                continue
            lhs_scale, rhs_scale = multipliers
            kind = (
                "exact"
                if lhs_scale == 1 and rhs_scale == 1
                else "integer_multiple"
            )
            score = relation_score(left, right, lhs_scale, rhs_scale, kind)
            stable = hashlib.sha256(
                (
                    f"{grammar.name}|{primitive_hash}|{left.key}|{right.key}|"
                    f"{lhs_scale}|{rhs_scale}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            relations.append(
                Relation(
                    grammar=grammar.name,
                    relation_id=f"{grammar.name}:r:{stable}",
                    kind=kind,
                    score=score,
                    lhs=left,
                    rhs=right,
                    lhs_scale=lhs_scale,
                    rhs_scale=rhs_scale,
                    polynomial_hash=(
                        left.exact_hash if kind == "exact" else primitive_hash
                    ),
                    primitive_hash=primitive_hash,
                    certificate_note=(
                        "expressions have identical exact blade-wise polynomials"
                        if kind == "exact"
                        else "expressions have proportional primitive blade-wise polynomials"
                    ),
                )
            )
            pair_count += 1

    unique: dict[tuple[str, str, int, int], Relation] = {}
    for relation in relations:
        right_key = relation.rhs.key if relation.rhs is not None else "0"
        key = (
            relation.lhs.key,
            right_key,
            relation.lhs_scale,
            relation.rhs_scale,
        )
        previous = unique.get(key)
        if previous is None or relation.score > previous.score:
            unique[key] = relation
    ordered_relations = sorted(
        unique.values(),
        key=lambda relation: (-relation.score, relation.relation_id),
    )
    selected: list[Relation] = []
    per_class: Counter[str] = Counter()
    for relation in ordered_relations:
        class_key = relation.primitive_hash
        if per_class[class_key] >= grammar.max_relations_per_primitive_class:
            continue
        selected.append(relation)
        per_class[class_key] += 1
        if len(selected) >= grammar.max_relations:
            break
    return selected


def relation_specification(
    grammar: GrammarConfig,
    relation: Relation,
    *,
    prime: int,
    name: str,
) -> dict[str, Any]:
    rhs_expression = (
        {"scalar": 0} if relation.rhs is None else relation.rhs.expression
    )
    spec = {
        "schema_version": 1,
        "name": name,
        "description": (
            f"V3 grammar relation from {grammar.name}: "
            f"{relation.lhs_scale}*({relation.lhs.label}) = "
            f"{relation.rhs_scale}*({expression_label(rhs_expression)}); "
            f"relation={relation.relation_id}; kind={relation.kind}."
        ),
        "expected": "identity",
        "dimension": grammar.dimension,
        "signature": list(grammar.signature),
        "prime": prime,
        "coefficient_bound": grammar.coefficient_bound,
        "seed": grammar.seed,
        "variables": [copy.deepcopy(variable) for variable in grammar.variables],
        "lhs": scaled_expression(relation.lhs.expression, relation.lhs_scale),
        "rhs": scaled_expression(rhs_expression, relation.rhs_scale),
    }
    exact.validate_spec(spec, name)
    return spec


def generate_controls(
    grammar: GrammarConfig,
    relations: Sequence[Relation],
    *,
    precheck_assignments: int,
    term_limit: int,
) -> list[Control]:
    controls: list[Control] = []
    for relation in relations:
        if len(controls) >= grammar.max_controls:
            break
        base_name = sanitize_identifier(relation.relation_id)[:80]
        base_spec = relation_specification(
            grammar,
            relation,
            prime=grammar.prime,
            name=f"{base_name}_base",
        )
        for mutation, candidate in exact.generate_mutations(base_spec, 16):
            candidate = copy.deepcopy(candidate)
            candidate["name"] = f"{base_name}_control_{mutation.stable_id}"
            polynomial = exact.extract_polynomial(candidate, term_limit)
            if polynomial["zero"]:
                continue
            assignment, mismatch = exact.precheck(candidate, precheck_assignments)
            if assignment is None or mismatch is None:
                continue
            controls.append(
                Control(
                    relation_id=relation.relation_id,
                    control_id=f"{relation.relation_id}:control:{mutation.stable_id}",
                    mutation={
                        "stable_id": mutation.stable_id,
                        "kind": mutation.kind,
                        "side": mutation.side,
                        "path": list(mutation.path),
                    },
                    specification=candidate,
                    polynomial=polynomial,
                    witness_assignment=assignment,
                    witness=mismatch,
                )
            )
            break
    return controls


def relation_certificate(grammar: GrammarConfig, relation: Relation) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine": "geometric_identity_grammar_discovery_v3",
        "grammar": grammar.name,
        "relation_id": relation.relation_id,
        "relation_kind": relation.kind,
        "score": relation.score,
        "lhs_scale": relation.lhs_scale,
        "rhs_scale": relation.rhs_scale,
        "lhs": {
            "expression": relation.lhs.expression,
            "label": relation.lhs.label,
            "cost": relation.lhs.cost,
            "exact_hash": relation.lhs.exact_hash,
            "primitive_hash": relation.lhs.primitive_hash,
            "blades": relation.lhs.exact_blades,
        },
        "rhs": (
            {"expression": {"scalar": 0}, "label": "0", "cost": 1}
            if relation.rhs is None
            else {
                "expression": relation.rhs.expression,
                "label": relation.rhs.label,
                "cost": relation.rhs.cost,
                "exact_hash": relation.rhs.exact_hash,
                "primitive_hash": relation.rhs.primitive_hash,
                "blades": relation.rhs.exact_blades,
            }
        ),
        "polynomial_hash": relation.polynomial_hash,
        "primitive_hash": relation.primitive_hash,
        "certificate_note": relation.certificate_note,
        "scope": {
            "dimension": grammar.dimension,
            "signature": list(grammar.signature),
            "variables": [copy.deepcopy(variable) for variable in grammar.variables],
        },
    }


def replace_grammar_limit(
    grammar: GrammarConfig,
    *,
    max_relations: int | None,
    max_controls: int | None,
) -> GrammarConfig:
    values = asdict(grammar)
    if max_relations is not None:
        values["max_relations"] = min(grammar.max_relations, max_relations)
    if max_controls is not None:
        values["max_controls"] = min(grammar.max_controls, max_controls)
    return GrammarConfig(**values)


def build_corpus(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    corpus_dir = output_dir / "corpus"
    certificate_dir = output_dir / "certificates"
    polynomial_dir = output_dir / "polynomials"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    certificate_dir.mkdir(parents=True, exist_ok=True)
    polynomial_dir.mkdir(parents=True, exist_ok=True)

    primes = tuple(dict.fromkeys(args.prime or DEFAULT_PRIMES))
    for prime in primes:
        if not exact.is_prime(prime) or not 3 <= prime <= 1_000_003:
            raise GrammarError(f"invalid configured prime: {prime}")

    entries: list[dict[str, Any]] = []
    grammar_reports: list[dict[str, Any]] = []
    relation_reports: list[dict[str, Any]] = []
    file_index = 1
    total_relations = 0
    total_controls = 0

    for grammar_path in args.grammar:
        grammar = replace_grammar_limit(
            load_grammar(grammar_path),
            max_relations=args.max_relations,
            max_controls=args.max_controls,
        )
        records, stats = enumerate_expressions(grammar)
        relations = discover_relations(grammar, records)
        controls = generate_controls(
            grammar,
            relations,
            precheck_assignments=args.precheck_assignments,
            term_limit=args.term_limit,
        )
        total_relations += len(relations)
        total_controls += len(controls)
        grammar_reports.append(
            {
                "path": grammar_path.as_posix(),
                "name": grammar.name,
                "description": grammar.description,
                "statistics": stats,
                "selected_relations": len(relations),
                "generated_controls": len(controls),
            }
        )

        control_by_relation: dict[str, list[Control]] = defaultdict(list)
        for control in controls:
            control_by_relation[control.relation_id].append(control)

        for relation_index, relation in enumerate(relations, 1):
            certificate = relation_certificate(grammar, relation)
            certificate_filename = (
                f"{sanitize_identifier(grammar.name)}_r{relation_index:03d}_"
                f"{relation.relation_id.rsplit(':', 1)[-1]}.json"
            )
            write_json(certificate_dir / certificate_filename, certificate)
            relation_reports.append(
                {
                    "grammar": grammar.name,
                    "relation_id": relation.relation_id,
                    "kind": relation.kind,
                    "score": relation.score,
                    "lhs": relation.lhs.label,
                    "rhs": "0" if relation.rhs is None else relation.rhs.label,
                    "lhs_scale": relation.lhs_scale,
                    "rhs_scale": relation.rhs_scale,
                    "certificate": f"certificates/{certificate_filename}",
                }
            )
            for prime in primes:
                stable = relation.relation_id.rsplit(":", 1)[-1]
                name = (
                    f"{sanitize_identifier(grammar.name)}__r{relation_index:03d}_"
                    f"{stable}__p{prime}"
                )
                spec = relation_specification(
                    grammar, relation, prime=prime, name=name
                )
                polynomial = exact.extract_polynomial(spec, args.term_limit)
                if not polynomial["zero"]:
                    raise GrammarError(
                        f"selected relation {relation.relation_id} has a nonzero difference polynomial"
                    )
                assignment, mismatch = exact.precheck(spec, args.precheck_assignments)
                if mismatch is not None:
                    raise GrammarError(
                        f"selected relation {relation.relation_id} failed precheck at {assignment}"
                    )
                filename = f"{file_index:04d}_{name}.json"
                file_index += 1
                write_json(corpus_dir / filename, spec)
                polynomial_filename = f"{hashlib.sha256((relation.relation_id + str(prime)).encode()).hexdigest()[:16]}.json"
                polynomial_record = copy.deepcopy(polynomial)
                polynomial_record.update(
                    {
                        "grammar": grammar.name,
                        "relation_id": relation.relation_id,
                        "relation_kind": relation.kind,
                        "certificate": f"certificates/{certificate_filename}",
                    }
                )
                write_json(polynomial_dir / polynomial_filename, polynomial_record)
                entries.append(
                    {
                        "file": f"corpus/{filename}",
                        "name": name,
                        "variant_id": relation.relation_id,
                        "source_identity": grammar.name,
                        "source_expected": "generated_relation",
                        "prime": prime,
                        "expected": "identity",
                        "mutation": None,
                        "relation_kind": relation.kind,
                        "relation_score": relation.score,
                        "certificate": f"certificates/{certificate_filename}",
                        "polynomial_file": f"polynomials/{polynomial_filename}",
                        "polynomial_zero": True,
                        "polynomial_hash": polynomial["canonical_hash"],
                        "polynomial_terms": 0,
                        "maximum_degree": polynomial["maximum_degree"],
                        "precheck_assignments": args.precheck_assignments,
                        "precheck_found_counterexample": False,
                        "precheck_witness": None,
                    }
                )

            for control_index, control in enumerate(control_by_relation.get(relation.relation_id, []), 1):
                for prime in primes:
                    stable = control.control_id.rsplit(":", 1)[-1]
                    name = (
                        f"{sanitize_identifier(grammar.name)}__r{relation_index:03d}_"
                        f"c{control_index:02d}_{stable}__p{prime}"
                    )
                    spec = copy.deepcopy(control.specification)
                    spec["name"] = name
                    spec["prime"] = prime
                    spec["expected"] = "counterexample"
                    spec["description"] = (
                        f"V3 near-miss control for {relation.relation_id}; "
                        f"mutation={control.mutation['kind']}; prime={prime}."
                    )
                    exact.validate_spec(spec, name)
                    polynomial = exact.extract_polynomial(spec, args.term_limit)
                    if polynomial["zero"] or not exact.polynomial_nonzero_mod_prime(
                        polynomial, prime
                    ):
                        continue
                    assignment, mismatch = exact.precheck(
                        spec, args.precheck_assignments
                    )
                    if assignment is None or mismatch is None:
                        continue
                    filename = f"{file_index:04d}_{name}.json"
                    file_index += 1
                    write_json(corpus_dir / filename, spec)
                    polynomial_filename = f"{hashlib.sha256((control.control_id + str(prime)).encode()).hexdigest()[:16]}.json"
                    write_json(polynomial_dir / polynomial_filename, polynomial)
                    entries.append(
                        {
                            "file": f"corpus/{filename}",
                            "name": name,
                            "variant_id": control.control_id,
                            "source_identity": grammar.name,
                            "source_expected": "generated_control",
                            "prime": prime,
                            "expected": "counterexample",
                            "mutation": control.mutation,
                            "relation_kind": "near_miss_control",
                            "relation_score": relation.score,
                            "certificate": f"certificates/{certificate_filename}",
                            "polynomial_file": f"polynomials/{polynomial_filename}",
                            "polynomial_zero": False,
                            "polynomial_hash": polynomial["canonical_hash"],
                            "polynomial_terms": polynomial["total_terms"],
                            "maximum_degree": polynomial["maximum_degree"],
                            "precheck_assignments": args.precheck_assignments,
                            "precheck_found_counterexample": True,
                            "precheck_witness": {
                                "assignment": assignment,
                                "blade": mismatch.blade,
                                "blade_label": exact.blade_label(
                                    mismatch.blade, grammar.dimension
                                ),
                                "lhs": mismatch.lhs,
                                "rhs": mismatch.rhs,
                            },
                        }
                    )

    if not entries:
        raise GrammarError("grammar discovery produced no finite-field statements")
    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_grammar_discovery_v3",
        "primes": list(primes),
        "precheck_assignments": args.precheck_assignments,
        "term_limit": args.term_limit,
        "grammars": grammar_reports,
        "relations": relation_reports,
        "entries": entries,
    }
    write_json(output_dir / "corpus-manifest.json", manifest)
    (output_dir / "identity-files.txt").write_text(
        "".join(f"{entry['file']}\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        output_dir / "grammar-report.json",
        {
            "schema_version": 1,
            "engine": "geometric_identity_grammar_discovery_v3",
            "grammars": grammar_reports,
            "relations": relation_reports,
            "finite_field_statements": len(entries),
            "selected_relations": total_relations,
            "near_miss_controls": total_controls,
        },
    )
    summary = [
        "# Grammar-bounded geometric identity discovery corpus",
        "",
        f"- grammars: {len(grammar_reports)}",
        f"- selected exact relations: {total_relations}",
        f"- generated near-miss controls: {total_controls}",
        f"- finite-field statements: {len(entries)}",
        f"- primes: {', '.join(str(prime) for prime in primes)}",
        f"- deterministic prechecks per statement: {args.precheck_assignments}",
        "",
        "| grammar | expressions | exact classes | primitive classes | zero expressions | relations | controls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in grammar_reports:
        stats = report["statistics"]
        summary.append(
            f"| {report['name']} | {stats['expressions']} | "
            f"{stats['exact_polynomial_classes']} | "
            f"{stats['primitive_polynomial_classes']} | "
            f"{stats['zero_expressions']} | {report['selected_relations']} | "
            f"{report['generated_controls']} |"
        )
    summary.extend(
        [
            "",
            "Selected identity rows have exact zero difference polynomials for their",
            "declared dimension, signature, and grade supports. Near-miss rows have",
            "nonzero difference polynomials and deterministic exact precheck witnesses.",
            "",
        ]
    )
    (output_dir / "corpus-summary.md").write_text(
        "\n".join(summary), encoding="utf-8", newline="\n"
    )
    print(f"generated {len(entries)} finite-field statements in {corpus_dir}")
    print(f"selected relations: {total_relations}")
    print(f"near-miss controls: {total_controls}")
    print(f"manifest: {output_dir / 'corpus-manifest.json'}")
    return 0


def analyze_grammar(args: argparse.Namespace) -> int:
    grammar = load_grammar(args.grammar)
    records, stats = enumerate_expressions(grammar)
    relations = discover_relations(grammar, records)
    report = {
        "grammar": asdict(grammar),
        "statistics": stats,
        "relations": [relation_certificate(grammar, relation) for relation in relations],
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact Clifford identity candidates from bounded grammars"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="enumerate and classify one grammar")
    analyze.add_argument("--grammar", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=analyze_grammar)

    build = subparsers.add_parser(
        "build-corpus", help="emit selected relations and controls across prime fields"
    )
    build.add_argument("--grammar", type=Path, action="append", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--prime", type=int, action="append")
    build.add_argument("--precheck-assignments", type=int, default=2048)
    build.add_argument("--term-limit", type=int, default=200_000)
    build.add_argument("--max-relations", type=int)
    build.add_argument("--max-controls", type=int)
    build.add_argument("--clean", action="store_true")
    build.set_defaults(handler=build_corpus)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if getattr(args, "precheck_assignments", 0) < 0:
            raise GrammarError("--precheck-assignments must be non-negative")
        if getattr(args, "term_limit", 1) < 1:
            raise GrammarError("--term-limit must be positive")
        if getattr(args, "max_relations", None) is not None and args.max_relations < 1:
            raise GrammarError("--max-relations must be positive")
        if getattr(args, "max_controls", None) is not None and args.max_controls < 0:
            raise GrammarError("--max-controls must be non-negative")
        return int(args.handler(args))
    except (OSError, json.JSONDecodeError, GrammarError, exact.DiscoveryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
