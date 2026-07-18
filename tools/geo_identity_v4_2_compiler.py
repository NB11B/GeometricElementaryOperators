#!/usr/bin/env python3
"""Native V4.2 compiler and host/CUDA emitter for fixed-blade identity IR.

The module subclasses the accepted compiler without mutating it. Fixed basis
blades become true DAG nodes with exact one-blade support. Header emission uses
an isolated scalar-shadow lowering so legacy generated host/CUDA code remains
stable for all existing operations.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

try:
    import geo_identity_compiler as legacy
    import geo_identity_v4_2_exact as exact
except ModuleNotFoundError:
    from tools import geo_identity_compiler as legacy
    from tools import geo_identity_v4_2_exact as exact

IdentityError = legacy.IdentityError
Variable = legacy.Variable
Node = legacy.Node
Identity = legacy.Identity


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityError(f"{context} must be a non-empty string")
    if not (value[0].isalpha() or value[0] == "_"):
        raise IdentityError(f"{context} is not a valid identifier: {value!r}")
    if not all(character.isalnum() or character == "_" for character in value):
        raise IdentityError(f"{context} is not a valid identifier: {value!r}")
    return value


def integer(value: object, context: str) -> int:
    if not is_integer(value):
        raise IdentityError(f"{context} must be an integer")
    return int(value)


class Builder(legacy.Builder):
    def fixed_blade(self, blade: int, coefficient: int) -> int:
        if not 0 <= blade < self.blade_count:
            raise IdentityError(f"fixed blade {blade} is outside dimension {self.dimension}")
        normalized = coefficient % self.prime
        if normalized == 0:
            raise IdentityError("fixed-blade coefficient is zero modulo the statement prime")
        return self.intern(
            op="fixed_blade",
            value=normalized,
            grade=blade,
            support_mask=1 << blade,
            key=("fixed_blade", blade, normalized),
        )

    def parse(self, raw: object, context: str) -> int:
        try:
            parsed = exact.parse_fixed_blade(raw, self.dimension, context)
        except ValueError as exc:
            raise IdentityError(str(exc)) from exc
        if parsed is not None:
            return self.fixed_blade(*parsed)
        return super().parse(raw, context)


def load_identity(path: Path) -> Identity:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"unable to read {path}: {exc}") from exc
    try:
        exact.validate_spec(raw)
    except exact.DiscoveryError as exc:
        raise IdentityError(str(exc)) from exc

    name = identifier(raw.get("name"), f"{path}.name")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise IdentityError(f"{path}.description must be a string")
    expected = raw.get("expected")
    dimension = integer(raw.get("dimension"), f"{path}.dimension")
    signature = tuple(int(value) for value in raw["signature"])
    prime = integer(raw.get("prime", 65521), f"{path}.prime")
    coefficient_bound = integer(raw.get("coefficient_bound", 3), f"{path}.coefficient_bound")
    if coefficient_bound < 1 or 2 * coefficient_bound + 1 >= prime:
        raise IdentityError(f"{path}.coefficient_bound is invalid")
    seed = integer(raw.get("seed", 0x243F6A88), f"{path}.seed")
    if seed < 0 or seed > (1 << 64) - 1:
        raise IdentityError(f"{path}.seed must fit uint64")

    variables = tuple(
        Variable(
            identifier(variable["name"], f"{path}.variables.name"),
            tuple(sorted(set(int(grade) for grade in variable["grades"]))),
        )
        for variable in raw["variables"]
    )
    builder = Builder(dimension=dimension, prime=prime, variables=variables)
    lhs = builder.parse(raw["lhs"], f"{path}.lhs")
    rhs = builder.parse(raw["rhs"], f"{path}.rhs")
    return Identity(
        name=name,
        description=description,
        expected=expected,
        dimension=dimension,
        signature=signature,
        prime=prime,
        coefficient_bound=coefficient_bound,
        seed=seed,
        variables=variables,
        nodes=tuple(builder.nodes),
        lhs=lhs,
        rhs=rhs,
    )


def load_corpus(paths: Sequence[Path]) -> list[Identity]:
    identities = [load_identity(path) for path in paths]
    names = [identity.name for identity in identities]
    if len(names) != len(set(names)):
        raise IdentityError("corpus contains duplicate identity names")
    return identities


def evaluate_identity(identity: Identity, assignment: int) -> tuple[bool, int, int, int]:
    blade_count = 1 << identity.dimension
    variables = legacy.generate_assignment(identity, assignment)
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
            value = [(node.value * coefficient) % identity.prime for coefficient in values[node.args[0]]]
        elif node.op == "gp":
            value = legacy._gp(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "wedge":
            value = legacy._wedge(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "reverse":
            value = [0] * blade_count
            for blade, coefficient in enumerate(values[node.args[0]]):
                grade = blade.bit_count()
                sign = -1 if ((grade * (grade - 1) // 2) & 1) else 1
                value[blade] = sign * coefficient % identity.prime
        elif node.op == "grade":
            assert node.grade is not None
            value = [coefficient if blade.bit_count() == node.grade else 0 for blade, coefficient in enumerate(values[node.args[0]])]
        else:
            raise AssertionError(node.op)
        values.append(value)

    lhs = values[identity.lhs]
    rhs = values[identity.rhs]
    for blade, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return False, blade, left, right
    return True, 0, 0, 0


def emit_identity(identity: Identity, index: int) -> list[str]:
    fixed_nodes: dict[int, int] = {}
    shadow_nodes: list[Node] = []
    for node_index, node in enumerate(identity.nodes):
        if node.op == "fixed_blade":
            assert node.grade is not None
            fixed_nodes[node_index] = node.grade
            shadow_nodes.append(replace(node, op="scalar"))
        else:
            shadow_nodes.append(node)
    shadow = replace(identity, nodes=tuple(shadow_nodes))
    lines = legacy._emit_identity(shadow, index)
    for node_index, blade in fixed_nodes.items():
        value = identity.nodes[node_index].value
        marker = f"        nodes[{node_index}].c[0] = {value};"
        replacement = f"        nodes[{node_index}].c[{blade}] = {value};"
        matches = [line_index for line_index, line in enumerate(lines) if line == marker]
        if len(matches) != 1:
            raise IdentityError(f"unable to lower fixed-blade node {node_index}; scalar marker count={len(matches)}")
        lines[matches[0]] = replacement
    return lines


def emit_header(identities: Sequence[Identity], source_paths: Sequence[str]) -> str:
    lines = ["/*", " * Generated file. Do not edit by hand.", " * Sources:"]
    lines.extend(f" *   - {path}" for path in source_paths)
    lines.extend([
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
    ])
    for index, identity in enumerate(identities):
        lines.extend(emit_identity(identity, index))
    lines.extend([
        f"inline constexpr int IDENTITY_COUNT = {len(identities)};",
        "",
        "#define GEO_IDENTITY_FOR_EACH(MACRO) \\",
    ])
    for index in range(len(identities)):
        suffix = " \\" if index + 1 < len(identities) else ""
        lines.append(f"    MACRO({index}){suffix}")
    lines.extend([
        "",
        "}  // namespace geo_identity_generated",
        "",
        "#undef GEO_ID_UNROLL",
        "#undef GEO_ID_HD",
        "",
        "#endif  // GEO_GENERATED_IDENTITY_CORPUS_CUH",
        "",
    ])
    return "\n".join(lines)
