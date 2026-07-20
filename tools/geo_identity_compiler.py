#!/usr/bin/env python3
"""Compile checked Clifford identities into exact CPU/CUDA search kernels.

The v1 IR intentionally targets bounded, exact experimental mathematics:
- non-degenerate signatures with entries +1 or -1;
- dimensions 1 through 6, so basis support fits in a uint64 mask;
- coefficients evaluated modulo an odd prime;
- variables constrained by grade;
- expression DAGs with exact common-subexpression elimination;
- geometric product, wedge, reverse, grade projection, and additive structure.

The generated header is usable from both host C++ and CUDA device code.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class IdentityError(ValueError):
    """Raised when an identity specification violates the v1 contract."""


@dataclass(frozen=True)
class Variable:
    name: str
    grades: tuple[int, ...]


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple[int, ...]
    value: int | None
    grade: int | None
    variable: int | None
    support_mask: int
    key: tuple[Any, ...]


@dataclass(frozen=True)
class Identity:
    name: str
    description: str
    expected: str
    dimension: int
    signature: tuple[int, ...]
    prime: int
    coefficient_bound: int
    seed: int
    variables: tuple[Variable, ...]
    nodes: tuple[Node, ...]
    lhs: int
    rhs: int


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityError(f"{context} must be a non-empty string")
    if not (value[0].isalpha() or value[0] == "_"):
        raise IdentityError(f"{context} is not a valid identifier: {value!r}")
    if not all(character.isalnum() or character == "_" for character in value):
        raise IdentityError(f"{context} is not a valid identifier: {value!r}")
    return value


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IdentityError(f"{context} must be an integer")
    return value


def _popcount(value: int) -> int:
    return value.bit_count()


def _blade_mask_for_grades(dimension: int, grades: Iterable[int]) -> int:
    grade_set = set(grades)
    mask = 0
    for blade in range(1 << dimension):
        if _popcount(blade) in grade_set:
            mask |= 1 << blade
    return mask


def _active_blades(mask: int, blade_count: int) -> Iterable[int]:
    for blade in range(blade_count):
        if (mask >> blade) & 1:
            yield blade


def _gp_support(left: int, right: int, blade_count: int) -> int:
    support = 0
    for blade_left in _active_blades(left, blade_count):
        for blade_right in _active_blades(right, blade_count):
            support |= 1 << (blade_left ^ blade_right)
    return support


def _wedge_support(left: int, right: int, blade_count: int) -> int:
    support = 0
    for blade_left in _active_blades(left, blade_count):
        for blade_right in _active_blades(right, blade_count):
            if blade_left & blade_right == 0:
                support |= 1 << (blade_left | blade_right)
    return support


class Builder:
    def __init__(
        self,
        *,
        dimension: int,
        prime: int,
        variables: Sequence[Variable],
    ) -> None:
        self.dimension = dimension
        self.prime = prime
        self.blade_count = 1 << dimension
        self.variables = tuple(variables)
        self.variable_index = {
            variable.name: index for index, variable in enumerate(variables)
        }
        self.nodes: list[Node] = []
        self.interned: dict[tuple[Any, ...], int] = {}

    def intern(
        self,
        *,
        op: str,
        args: tuple[int, ...] = (),
        value: int | None = None,
        grade: int | None = None,
        variable: int | None = None,
        support_mask: int,
        key: tuple[Any, ...],
    ) -> int:
        existing = self.interned.get(key)
        if existing is not None:
            return existing
        index = len(self.nodes)
        self.nodes.append(Node(op, args, value, grade, variable, support_mask, key))
        self.interned[key] = index
        return index

    def scalar(self, value: int) -> int:
        normalized = value % self.prime
        return self.intern(
            op="scalar",
            value=normalized,
            support_mask=1,
            key=("scalar", normalized),
        )

    def variable(self, name: str) -> int:
        try:
            index = self.variable_index[name]
        except KeyError as exc:
            raise IdentityError(
                f"expression references unknown variable {name!r}"
            ) from exc
        variable = self.variables[index]
        support = _blade_mask_for_grades(self.dimension, variable.grades)
        return self.intern(
            op="var",
            variable=index,
            support_mask=support,
            key=("var", name),
        )

    def neg(self, argument: int) -> int:
        node = self.nodes[argument]
        if node.op == "scalar":
            assert node.value is not None
            return self.scalar(-node.value)
        if node.op == "neg":
            return node.args[0]
        return self.intern(
            op="neg",
            args=(argument,),
            support_mask=node.support_mask,
            key=("neg", node.key),
        )

    def scale(self, value: int, argument: int) -> int:
        normalized = value % self.prime
        if normalized == 0:
            return self.scalar(0)
        if normalized == 1:
            return argument
        node = self.nodes[argument]
        if node.op == "scalar":
            assert node.value is not None
            return self.scalar(normalized * node.value)
        return self.intern(
            op="scale",
            args=(argument,),
            value=normalized,
            support_mask=node.support_mask,
            key=("scale", normalized, node.key),
        )

    def add(self, arguments: Sequence[int]) -> int:
        flattened: list[int] = []
        for argument in arguments:
            node = self.nodes[argument]
            if node.op == "add":
                flattened.extend(node.args)
            elif node.op == "scalar" and node.value == 0:
                continue
            else:
                flattened.append(argument)
        if not flattened:
            return self.scalar(0)
        if len(flattened) == 1:
            return flattened[0]
        flattened.sort(key=lambda index: repr(self.nodes[index].key))
        support = 0
        for argument in flattened:
            support |= self.nodes[argument].support_mask
        keys = tuple(self.nodes[index].key for index in flattened)
        return self.intern(
            op="add",
            args=tuple(flattened),
            support_mask=support,
            key=("add", keys),
        )

    def gp(self, left: int, right: int) -> int:
        support = _gp_support(
            self.nodes[left].support_mask,
            self.nodes[right].support_mask,
            self.blade_count,
        )
        return self.intern(
            op="gp",
            args=(left, right),
            support_mask=support,
            key=("gp", self.nodes[left].key, self.nodes[right].key),
        )

    def wedge(self, left: int, right: int) -> int:
        support = _wedge_support(
            self.nodes[left].support_mask,
            self.nodes[right].support_mask,
            self.blade_count,
        )
        return self.intern(
            op="wedge",
            args=(left, right),
            support_mask=support,
            key=("wedge", self.nodes[left].key, self.nodes[right].key),
        )

    def reverse(self, argument: int) -> int:
        return self.intern(
            op="reverse",
            args=(argument,),
            support_mask=self.nodes[argument].support_mask,
            key=("reverse", self.nodes[argument].key),
        )

    def grade(self, grade: int, argument: int) -> int:
        if grade < 0 or grade > self.dimension:
            raise IdentityError(
                f"grade projection {grade} is outside dimension {self.dimension}"
            )
        support = self.nodes[argument].support_mask & _blade_mask_for_grades(
            self.dimension, (grade,)
        )
        if support == 0:
            return self.scalar(0)
        return self.intern(
            op="grade",
            args=(argument,),
            grade=grade,
            support_mask=support,
            key=("grade", grade, self.nodes[argument].key),
        )

    def parse(self, raw: object, context: str) -> int:
        if not isinstance(raw, dict):
            raise IdentityError(f"{context} must be an expression object")

        if "var" in raw:
            if len(raw) != 1:
                raise IdentityError(f"{context}: variable expression has extra fields")
            return self.variable(_identifier(raw["var"], f"{context}.var"))

        if "scalar" in raw:
            if len(raw) != 1:
                raise IdentityError(f"{context}: scalar expression has extra fields")
            return self.scalar(_integer(raw["scalar"], f"{context}.scalar"))

        op = raw.get("op")
        if not isinstance(op, str):
            raise IdentityError(f"{context}.op must be a string")

        if op == "add":
            arguments = raw.get("args")
            if not isinstance(arguments, list) or len(arguments) < 2:
                raise IdentityError(
                    f"{context}.args must contain at least two expressions"
                )
            return self.add(
                [
                    self.parse(argument, f"{context}.args[{index}]")
                    for index, argument in enumerate(arguments)
                ]
            )

        if op == "sub":
            arguments = raw.get("args")
            if not isinstance(arguments, list) or len(arguments) != 2:
                raise IdentityError(
                    f"{context}.args must contain exactly two expressions"
                )
            left = self.parse(arguments[0], f"{context}.args[0]")
            right = self.parse(arguments[1], f"{context}.args[1]")
            return self.add((left, self.neg(right)))

        if op == "neg":
            return self.neg(self.parse(raw.get("arg"), f"{context}.arg"))

        if op == "scale":
            value = _integer(raw.get("value"), f"{context}.value")
            argument = self.parse(raw.get("arg"), f"{context}.arg")
            return self.scale(value, argument)

        if op in {"gp", "wedge", "commutator"}:
            arguments = raw.get("args")
            if not isinstance(arguments, list) or len(arguments) != 2:
                raise IdentityError(
                    f"{context}.args must contain exactly two expressions"
                )
            left = self.parse(arguments[0], f"{context}.args[0]")
            right = self.parse(arguments[1], f"{context}.args[1]")
            if op == "gp":
                return self.gp(left, right)
            if op == "wedge":
                return self.wedge(left, right)
            return self.add((self.gp(left, right), self.neg(self.gp(right, left))))

        if op == "reverse":
            return self.reverse(self.parse(raw.get("arg"), f"{context}.arg"))

        if op == "grade":
            grade = _integer(raw.get("grade"), f"{context}.grade")
            return self.grade(grade, self.parse(raw.get("arg"), f"{context}.arg"))

        raise IdentityError(f"{context}: unsupported operation {op!r}")


def load_identity(path: Path) -> Identity:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"unable to read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise IdentityError(f"{path}: root must be an object")
    if raw.get("schema_version") != 1:
        raise IdentityError(
            f"{path}: unsupported schema_version {raw.get('schema_version')!r}"
        )

    name = _identifier(raw.get("name"), f"{path}.name")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise IdentityError(f"{path}.description must be a string")
    expected = raw.get("expected")
    if expected not in {"identity", "counterexample"}:
        raise IdentityError(f"{path}.expected must be identity or counterexample")

    dimension = _integer(raw.get("dimension"), f"{path}.dimension")
    if dimension < 1 or dimension > 6:
        raise IdentityError(f"{path}.dimension must be between 1 and 6")

    signature_raw = raw.get("signature")
    if (
        not isinstance(signature_raw, list)
        or len(signature_raw) != dimension
        or any(entry not in (-1, 1) for entry in signature_raw)
    ):
        raise IdentityError(
            f"{path}.signature must contain {dimension} entries, each +1 or -1"
        )
    signature = tuple(int(entry) for entry in signature_raw)

    prime = _integer(raw.get("prime", 65521), f"{path}.prime")
    if prime < 3 or prime % 2 == 0 or prime > 1_000_003:
        raise IdentityError(
            f"{path}.prime must be an odd integer in [3, 1000003]"
        )

    coefficient_bound = _integer(
        raw.get("coefficient_bound", 3), f"{path}.coefficient_bound"
    )
    if coefficient_bound < 1 or 2 * coefficient_bound + 1 >= prime:
        raise IdentityError(
            f"{path}.coefficient_bound must be positive and smaller than half the prime"
        )

    seed = _integer(raw.get("seed", 0x243F6A88), f"{path}.seed")
    if seed < 0 or seed > (1 << 64) - 1:
        raise IdentityError(f"{path}.seed must fit uint64")

    variables_raw = raw.get("variables")
    if not isinstance(variables_raw, list) or not variables_raw:
        raise IdentityError(f"{path}.variables must be a non-empty list")
    variables: list[Variable] = []
    for index, variable_raw in enumerate(variables_raw):
        context = f"{path}.variables[{index}]"
        if not isinstance(variable_raw, dict):
            raise IdentityError(f"{context} must be an object")
        variable_name = _identifier(variable_raw.get("name"), f"{context}.name")
        grades_raw = variable_raw.get("grades")
        if not isinstance(grades_raw, list) or not grades_raw:
            raise IdentityError(f"{context}.grades must be a non-empty list")
        grades = tuple(
            sorted(
                {
                    _integer(value, f"{context}.grades")
                    for value in grades_raw
                }
            )
        )
        if any(grade < 0 or grade > dimension for grade in grades):
            raise IdentityError(
                f"{context}.grades contains a grade outside the dimension"
            )
        variables.append(Variable(variable_name, grades))
    names = [variable.name for variable in variables]
    if len(names) != len(set(names)):
        raise IdentityError(f"{path}.variables contains duplicate names")

    builder = Builder(dimension=dimension, prime=prime, variables=variables)
    lhs = builder.parse(raw.get("lhs"), f"{path}.lhs")
    rhs = builder.parse(raw.get("rhs"), f"{path}.rhs")

    return Identity(
        name=name,
        description=description,
        expected=expected,
        dimension=dimension,
        signature=signature,
        prime=prime,
        coefficient_bound=coefficient_bound,
        seed=seed,
        variables=tuple(variables),
        nodes=tuple(builder.nodes),
        lhs=lhs,
        rhs=rhs,
    )


def _gp_sign(blade_left: int, blade_right: int, signature: Sequence[int]) -> int:
    sign = 1
    for index, metric in enumerate(signature):
        bit = 1 << index
        if blade_left & bit:
            lower = blade_right & (bit - 1)
            if lower.bit_count() & 1:
                sign = -sign
            if blade_right & bit:
                sign *= metric
    return sign


def _wedge_sign(blade_left: int, blade_right: int) -> int:
    if blade_left & blade_right:
        return 0
    sign = 1
    value = blade_left
    while value:
        lowest = value & -value
        lower = blade_right & (lowest - 1)
        if lower.bit_count() & 1:
            sign = -sign
        value ^= lowest
    return sign


def _mod(value: int, prime: int) -> int:
    return value % prime


def _gp(
    left: Sequence[int],
    right: Sequence[int],
    identity: Identity,
) -> list[int]:
    count = 1 << identity.dimension
    output = [0] * count
    for blade_left, coefficient_left in enumerate(left):
        if coefficient_left == 0:
            continue
        for blade_right, coefficient_right in enumerate(right):
            if coefficient_right == 0:
                continue
            blade = blade_left ^ blade_right
            sign = _gp_sign(blade_left, blade_right, identity.signature)
            output[blade] = _mod(
                output[blade] + sign * coefficient_left * coefficient_right,
                identity.prime,
            )
    return output


def _wedge(
    left: Sequence[int],
    right: Sequence[int],
    identity: Identity,
) -> list[int]:
    count = 1 << identity.dimension
    output = [0] * count
    for blade_left, coefficient_left in enumerate(left):
        if coefficient_left == 0:
            continue
        for blade_right, coefficient_right in enumerate(right):
            if coefficient_right == 0:
                continue
            sign = _wedge_sign(blade_left, blade_right)
            if sign == 0:
                continue
            blade = blade_left | blade_right
            output[blade] = _mod(
                output[blade] + sign * coefficient_left * coefficient_right,
                identity.prime,
            )
    return output


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def generate_assignment(identity: Identity, assignment: int) -> list[list[int]]:
    count = 1 << identity.dimension
    span = 2 * identity.coefficient_bound + 1
    variables: list[list[int]] = []
    for variable_index, variable in enumerate(identity.variables):
        coefficients = [0] * count
        grade_set = set(variable.grades)
        active = [
            blade for blade in range(count) if blade.bit_count() in grade_set
        ]
        nonzero = False
        for blade in active:
            mixed = (
                identity.seed
                ^ ((assignment + 1) * 0xD1B54A32D192ED03)
                ^ ((variable_index + 1) * 0x94D049BB133111EB)
                ^ ((blade + 1) * 0x9E3779B97F4A7C15)
            ) & ((1 << 64) - 1)
            signed = int(_splitmix64(mixed) % span) - identity.coefficient_bound
            coefficients[blade] = signed % identity.prime
            nonzero = nonzero or signed != 0
        if not nonzero:
            coefficients[active[0]] = 1
        variables.append(coefficients)
    return variables


def evaluate_identity(
    identity: Identity,
    assignment: int,
) -> tuple[bool, int, int, int]:
    blade_count = 1 << identity.dimension
    variables = generate_assignment(identity, assignment)
    values: list[list[int]] = []

    for node in identity.nodes:
        if node.op == "var":
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
                    value[blade] = (
                        value[blade] + values[argument][blade]
                    ) % identity.prime
        elif node.op == "neg":
            value = [
                (-coefficient) % identity.prime
                for coefficient in values[node.args[0]]
            ]
        elif node.op == "scale":
            assert node.value is not None
            value = [
                (node.value * coefficient) % identity.prime
                for coefficient in values[node.args[0]]
            ]
        elif node.op == "gp":
            value = _gp(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "wedge":
            value = _wedge(values[node.args[0]], values[node.args[1]], identity)
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


def _mask_literal(mask: int) -> str:
    return f"UINT64_C(0x{mask:016x})"


def _emit_identity(identity: Identity, index: int) -> list[str]:
    blade_count = 1 << identity.dimension
    lines: list[str] = [
        f"template <> struct identity<{index}> {{",
        f"    static constexpr int ID = {index};",
        f'    static constexpr const char *NAME = "{identity.name}";',
        f'    static constexpr const char *DESCRIPTION = {json.dumps(identity.description)};',
        f"    static constexpr bool EXPECT_COUNTEREXAMPLE = {'true' if identity.expected == 'counterexample' else 'false'};",
        f"    static constexpr int DIMENSION = {identity.dimension};",
        f"    static constexpr int BLADE_COUNT = {blade_count};",
        f"    static constexpr int PRIME = {identity.prime};",
        f"    static constexpr int COEFFICIENT_BOUND = {identity.coefficient_bound};",
        f"    static constexpr uint64_t SEED = UINT64_C(0x{identity.seed:016x});",
        f"    static constexpr int VARIABLE_COUNT = {len(identity.variables)};",
        f"    static constexpr int NODE_COUNT = {len(identity.nodes)};",
        "",
        "    GEO_ID_HD static int signature_at(int index) {",
        "        switch (index) {",
    ]
    for signature_index, signature_value in enumerate(identity.signature):
        lines.append(
            f"            case {signature_index}: return {signature_value};"
        )
    lines.extend(
        [
            "            default: return 0;",
            "        }",
            "    }",
            "",
            "    struct mv_t {",
            "        int32_t c[BLADE_COUNT];",
            "    };",
            "",
            "    GEO_ID_HD static int popcount_u32(uint32_t value) {",
            "        int count = 0;",
            "        while (value != 0U) {",
            "            value &= value - 1U;",
            "            ++count;",
            "        }",
            "        return count;",
            "    }",
            "",
            "    GEO_ID_HD static int32_t normalize(int64_t value) {",
            "        value %= static_cast<int64_t>(PRIME);",
            "        if (value < 0) value += PRIME;",
            "        return static_cast<int32_t>(value);",
            "    }",
            "",
            "    GEO_ID_HD static int32_t signed_value(int32_t value) {",
            "        return value > PRIME / 2 ? value - PRIME : value;",
            "    }",
            "",
            "    GEO_ID_HD static uint64_t splitmix64(uint64_t value) {",
            "        value += UINT64_C(0x9e3779b97f4a7c15);",
            "        value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);",
            "        value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);",
            "        return value ^ (value >> 31U);",
            "    }",
            "",
            "    GEO_ID_HD static void zero(mv_t *value) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) value->c[blade] = 0;",
            "    }",
            "",
            "    GEO_ID_HD static int gp_sign(uint32_t left, uint32_t right) {",
            "        int sign = 1;",
            "        GEO_ID_UNROLL",
            "        for (int index = 0; index < DIMENSION; ++index) {",
            "            const uint32_t bit = UINT32_C(1) << index;",
            "            if ((left & bit) != 0U) {",
            "                if ((popcount_u32(right & (bit - 1U)) & 1) != 0) sign = -sign;",
            "                if ((right & bit) != 0U) sign *= signature_at(index);",
            "            }",
            "        }",
            "        return sign;",
            "    }",
            "",
            "    GEO_ID_HD static int wedge_sign(uint32_t left, uint32_t right) {",
            "        if ((left & right) != 0U) return 0;",
            "        int sign = 1;",
            "        uint32_t value = left;",
            "        while (value != 0U) {",
            "            const uint32_t lowest = value & (~value + 1U);",
            "            if ((popcount_u32(right & (lowest - 1U)) & 1) != 0) sign = -sign;",
            "            value ^= lowest;",
            "        }",
            "        return sign;",
            "    }",
            "",
            "    template <uint64_t LEFT_MASK, uint64_t RIGHT_MASK>",
            "    GEO_ID_HD static void add(mv_t *output, const mv_t &left, const mv_t &right) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            const uint64_t bit = UINT64_C(1) << blade;",
            "            output->c[blade] = (LEFT_MASK & bit) != 0U || (RIGHT_MASK & bit) != 0U",
            "                ? normalize(static_cast<int64_t>(left.c[blade]) + right.c[blade])",
            "                : 0;",
            "        }",
            "    }",
            "",
            "    template <uint64_t MASK>",
            "    GEO_ID_HD static void negate(mv_t *output, const mv_t &input) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            const uint64_t bit = UINT64_C(1) << blade;",
            "            output->c[blade] = (MASK & bit) != 0U ? normalize(-input.c[blade]) : 0;",
            "        }",
            "    }",
            "",
            "    template <uint64_t MASK>",
            "    GEO_ID_HD static void scale(mv_t *output, const mv_t &input, int32_t factor) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            const uint64_t bit = UINT64_C(1) << blade;",
            "            output->c[blade] = (MASK & bit) != 0U",
            "                ? normalize(static_cast<int64_t>(factor) * input.c[blade])",
            "                : 0;",
            "        }",
            "    }",
            "",
            "    template <uint64_t LEFT_MASK, uint64_t RIGHT_MASK>",
            "    GEO_ID_HD static void geometric_product(",
            "        mv_t *output, const mv_t &left, const mv_t &right",
            "    ) {",
            "        zero(output);",
            "        GEO_ID_UNROLL",
            "        for (int blade_left = 0; blade_left < BLADE_COUNT; ++blade_left) {",
            "            const uint64_t left_bit = UINT64_C(1) << blade_left;",
            "            if ((LEFT_MASK & left_bit) == 0U || left.c[blade_left] == 0) continue;",
            "            GEO_ID_UNROLL",
            "            for (int blade_right = 0; blade_right < BLADE_COUNT; ++blade_right) {",
            "                const uint64_t right_bit = UINT64_C(1) << blade_right;",
            "                if ((RIGHT_MASK & right_bit) == 0U || right.c[blade_right] == 0) continue;",
            "                const int blade = blade_left ^ blade_right;",
            "                const int sign = gp_sign(",
            "                    static_cast<uint32_t>(blade_left),",
            "                    static_cast<uint32_t>(blade_right)",
            "                );",
            "                output->c[blade] = normalize(",
            "                    static_cast<int64_t>(output->c[blade]) +",
            "                    static_cast<int64_t>(sign) * left.c[blade_left] * right.c[blade_right]",
            "                );",
            "            }",
            "        }",
            "    }",
            "",
            "    template <uint64_t LEFT_MASK, uint64_t RIGHT_MASK>",
            "    GEO_ID_HD static void wedge_product(",
            "        mv_t *output, const mv_t &left, const mv_t &right",
            "    ) {",
            "        zero(output);",
            "        GEO_ID_UNROLL",
            "        for (int blade_left = 0; blade_left < BLADE_COUNT; ++blade_left) {",
            "            const uint64_t left_bit = UINT64_C(1) << blade_left;",
            "            if ((LEFT_MASK & left_bit) == 0U || left.c[blade_left] == 0) continue;",
            "            GEO_ID_UNROLL",
            "            for (int blade_right = 0; blade_right < BLADE_COUNT; ++blade_right) {",
            "                const uint64_t right_bit = UINT64_C(1) << blade_right;",
            "                if ((RIGHT_MASK & right_bit) == 0U || right.c[blade_right] == 0) continue;",
            "                const int sign = wedge_sign(",
            "                    static_cast<uint32_t>(blade_left),",
            "                    static_cast<uint32_t>(blade_right)",
            "                );",
            "                if (sign == 0) continue;",
            "                const int blade = blade_left | blade_right;",
            "                output->c[blade] = normalize(",
            "                    static_cast<int64_t>(output->c[blade]) +",
            "                    static_cast<int64_t>(sign) * left.c[blade_left] * right.c[blade_right]",
            "                );",
            "            }",
            "        }",
            "    }",
            "",
            "    template <uint64_t MASK>",
            "    GEO_ID_HD static void reverse(mv_t *output, const mv_t &input) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            const uint64_t bit = UINT64_C(1) << blade;",
            "            if ((MASK & bit) == 0U) {",
            "                output->c[blade] = 0;",
            "                continue;",
            "            }",
            "            const int grade = popcount_u32(static_cast<uint32_t>(blade));",
            "            const int sign = (((grade * (grade - 1)) / 2) & 1) != 0 ? -1 : 1;",
            "            output->c[blade] = normalize(static_cast<int64_t>(sign) * input.c[blade]);",
            "        }",
            "    }",
            "",
            "    template <int GRADE, uint64_t MASK>",
            "    GEO_ID_HD static void project_grade(mv_t *output, const mv_t &input) {",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            const uint64_t bit = UINT64_C(1) << blade;",
            "            output->c[blade] =",
            "                (MASK & bit) != 0U && popcount_u32(static_cast<uint32_t>(blade)) == GRADE",
            "                ? input.c[blade] : 0;",
            "        }",
            "    }",
            "",
            "    GEO_ID_HD static void make_variables(uint64_t assignment, mv_t *variables) {",
        ]
    )

    span = 2 * identity.coefficient_bound + 1
    for variable_index, variable in enumerate(identity.variables):
        support = _blade_mask_for_grades(identity.dimension, variable.grades)
        active = list(_active_blades(support, blade_count))
        lines.extend(
            [
                f"        zero(&variables[{variable_index}]);",
                f"        bool nonzero_{variable_index} = false;",
            ]
        )
        for blade in active:
            lines.extend(
                [
                    "        {",
                    "            const uint64_t mixed = SEED ^",
                    "                ((assignment + UINT64_C(1)) * UINT64_C(0xd1b54a32d192ed03)) ^",
                    f"                (UINT64_C({variable_index + 1}) * UINT64_C(0x94d049bb133111eb)) ^",
                    f"                (UINT64_C({blade + 1}) * UINT64_C(0x9e3779b97f4a7c15));",
                    f"            const int32_t signed_coefficient = static_cast<int32_t>(splitmix64(mixed) % UINT64_C({span})) - COEFFICIENT_BOUND;",
                    f"            variables[{variable_index}].c[{blade}] = normalize(signed_coefficient);",
                    f"            nonzero_{variable_index} = nonzero_{variable_index} || signed_coefficient != 0;",
                    "        }",
                ]
            )
        lines.extend(
            [
                f"        if (!nonzero_{variable_index}) variables[{variable_index}].c[{active[0]}] = 1;",
            ]
        )
    lines.extend(
        [
            "    }",
            "",
            "    GEO_ID_HD static witness_t evaluate(uint64_t assignment) {",
            "        mv_t variables[VARIABLE_COUNT];",
            "        make_variables(assignment, variables);",
            "        mv_t nodes[NODE_COUNT];",
        ]
    )

    for node_index, node in enumerate(identity.nodes):
        if node.op == "var":
            assert node.variable is not None
            lines.append(
                f"        nodes[{node_index}] = variables[{node.variable}];"
            )
        elif node.op == "scalar":
            assert node.value is not None
            lines.extend(
                [
                    f"        zero(&nodes[{node_index}]);",
                    f"        nodes[{node_index}].c[0] = {node.value};",
                ]
            )
        elif node.op == "add":
            first = node.args[0]
            lines.append(f"        nodes[{node_index}] = nodes[{first}];")
            current_mask = identity.nodes[first].support_mask
            for argument in node.args[1:]:
                next_mask = identity.nodes[argument].support_mask
                lines.append(
                    f"        add<{_mask_literal(current_mask)}, {_mask_literal(next_mask)}>("
                    f"&nodes[{node_index}], nodes[{node_index}], nodes[{argument}]);"
                )
                current_mask |= next_mask
        elif node.op == "neg":
            argument = node.args[0]
            lines.append(
                f"        negate<{_mask_literal(identity.nodes[argument].support_mask)}>("
                f"&nodes[{node_index}], nodes[{argument}]);"
            )
        elif node.op == "scale":
            argument = node.args[0]
            assert node.value is not None
            lines.append(
                f"        scale<{_mask_literal(identity.nodes[argument].support_mask)}>("
                f"&nodes[{node_index}], nodes[{argument}], {node.value});"
            )
        elif node.op == "gp":
            left, right = node.args
            lines.append(
                f"        geometric_product<{_mask_literal(identity.nodes[left].support_mask)}, "
                f"{_mask_literal(identity.nodes[right].support_mask)}>("
                f"&nodes[{node_index}], nodes[{left}], nodes[{right}]);"
            )
        elif node.op == "wedge":
            left, right = node.args
            lines.append(
                f"        wedge_product<{_mask_literal(identity.nodes[left].support_mask)}, "
                f"{_mask_literal(identity.nodes[right].support_mask)}>("
                f"&nodes[{node_index}], nodes[{left}], nodes[{right}]);"
            )
        elif node.op == "reverse":
            argument = node.args[0]
            lines.append(
                f"        reverse<{_mask_literal(identity.nodes[argument].support_mask)}>("
                f"&nodes[{node_index}], nodes[{argument}]);"
            )
        elif node.op == "grade":
            argument = node.args[0]
            assert node.grade is not None
            lines.append(
                f"        project_grade<{node.grade}, {_mask_literal(identity.nodes[argument].support_mask)}>("
                f"&nodes[{node_index}], nodes[{argument}]);"
            )
        else:
            raise AssertionError(node.op)

    lines.extend(
        [
            f"        const mv_t &lhs = nodes[{identity.lhs}];",
            f"        const mv_t &rhs = nodes[{identity.rhs}];",
            "        GEO_ID_UNROLL",
            "        for (int blade = 0; blade < BLADE_COUNT; ++blade) {",
            "            if (lhs.c[blade] != rhs.c[blade]) {",
            "                witness_t witness{};",
            "                witness.equal = false;",
            "                witness.blade = static_cast<uint16_t>(blade);",
            "                witness.lhs = signed_value(lhs.c[blade]);",
            "                witness.rhs = signed_value(rhs.c[blade]);",
            "                return witness;",
            "            }",
            "        }",
            "        witness_t witness{};",
            "        witness.equal = true;",
            "        return witness;",
            "    }",
            "};",
            "",
        ]
    )
    return lines


def emit_header(identities: Sequence[Identity], source_paths: Sequence[str]) -> str:
    lines = [
        "/*",
        " * Generated file. Do not edit by hand.",
        " * Sources:",
    ]
    lines.extend(f" *   - {path}" for path in source_paths)
    lines.extend(
        [
            " */",
            "#ifndef GEO_GENERATED_IDENTITY_CORPUS_CUH",
            "#define GEO_GENERATED_IDENTITY_CORPUS_CUH",
            "",
            "#include <stdint.h>",
            "",
            "#if defined(__CUDACC__)",
            "#define GEO_ID_HD __host__ __device__ __forceinline__",
            "#define GEO_ID_UNROLL _Pragma(\"unroll\")",
            "#else",
            "#define GEO_ID_HD inline",
            "#define GEO_ID_UNROLL",
            "#endif",
            "",
            "namespace geo_identity_generated {",
            "",
            "struct witness_t {",
            "    bool equal;",
            "    uint16_t blade;",
            "    int32_t lhs;",
            "    int32_t rhs;",
            "};",
            "",
            "template <int INDEX> struct identity;",
            "",
        ]
    )
    for index, identity in enumerate(identities):
        lines.extend(_emit_identity(identity, index))
    lines.extend(
        [
            f"inline constexpr int IDENTITY_COUNT = {len(identities)};",
            "",
            "#define GEO_IDENTITY_FOR_EACH(MACRO) \\",
        ]
    )
    for index in range(len(identities)):
        suffix = " \\" if index + 1 < len(identities) else ""
        lines.append(f"    MACRO({index}){suffix}")
    lines.extend(
        [
            "",
            "}  // namespace geo_identity_generated",
            "",
            "#undef GEO_ID_UNROLL",
            "#undef GEO_ID_HD",
            "",
            "#endif  // GEO_GENERATED_IDENTITY_CORPUS_CUH",
            "",
        ]
    )
    return "\n".join(lines)


def load_corpus(paths: Sequence[Path]) -> list[Identity]:
    identities = [load_identity(path) for path in paths]
    names = [identity.name for identity in identities]
    if len(names) != len(set(names)):
        raise IdentityError("corpus contains duplicate identity names")
    return identities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile exact Clifford identity IR into a CPU/CUDA header"
    )
    parser.add_argument("--identity", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in generated header is stale",
    )
    parser.add_argument(
        "--python-checks",
        type=int,
        default=0,
        help="evaluate each identity over this many exact deterministic assignments",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        identities = load_corpus(args.identity)
        source_paths = [path.as_posix() for path in args.identity]
        generated = emit_header(identities, source_paths)

        if args.python_checks < 0:
            raise IdentityError("--python-checks must be non-negative")
        for identity in identities:
            found_counterexample = False
            for assignment in range(args.python_checks):
                equal, _, _, _ = evaluate_identity(identity, assignment)
                if not equal:
                    found_counterexample = True
                    if identity.expected == "identity":
                        raise IdentityError(
                            f"{identity.name} failed exact Python check at assignment {assignment}"
                        )
                    break
            if (
                args.python_checks > 0
                and identity.expected == "counterexample"
                and not found_counterexample
            ):
                raise IdentityError(
                    f"{identity.name} produced no counterexample in "
                    f"{args.python_checks} Python checks"
                )
    except IdentityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: unable to read {args.output}: {exc}", file=sys.stderr)
            return 2
        if existing != generated:
            print(
                f"ERROR: {args.output} is stale; regenerate from the identity corpus",
                file=sys.stderr,
            )
            return 1
        print(
            f"PASS: {args.output} matches {len(identities)} checked identity specifications"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
