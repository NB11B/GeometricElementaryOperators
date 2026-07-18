#!/usr/bin/env python3
"""V4.1 compiler extension with exact fixed basis-blade constants.

The accepted V1-V4 compiler remains unchanged. This module layers a strict
``fixed_blade`` node onto its Builder, Python evaluator, and generated host/CUDA
header emitter. All other operations delegate to the established compiler.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Sequence

import geo_identity_compiler as base

IdentityError = base.IdentityError
Identity = base.Identity
Node = base.Node
Variable = base.Variable

_BASE_BUILDER = base.Builder
_BASE_EMIT_IDENTITY = base._emit_identity


def _strict_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IdentityError(f"{context} must be an integer")
    return value


class FixedBladeBuilder(_BASE_BUILDER):
    """Established DAG builder plus a single fixed-blade constant node."""

    def fixed_blade(self, blade: int, coefficient: int) -> int:
        if not 0 <= blade < self.blade_count:
            raise IdentityError(
                f"fixed blade {blade} is outside [0,{self.blade_count - 1}]"
            )
        if coefficient == 0:
            raise IdentityError("fixed blade coefficient must be nonzero")
        normalized = coefficient % self.prime
        if normalized == 0:
            raise IdentityError(
                "fixed blade coefficient must be nonzero modulo the statement prime"
            )
        return self.intern(
            op="fixed_blade",
            value=normalized,
            grade=blade,
            support_mask=1 << blade,
            key=("fixed_blade", blade, normalized),
        )

    def parse(self, raw: object, context: str) -> int:
        if isinstance(raw, dict) and "fixed_blade" in raw:
            if set(raw) != {"fixed_blade"}:
                raise IdentityError(f"{context}: fixed_blade expression has extra fields")
            payload = raw["fixed_blade"]
            if not isinstance(payload, dict) or set(payload) != {
                "blade",
                "coefficient",
            }:
                raise IdentityError(
                    f"{context}.fixed_blade must contain exactly blade and coefficient"
                )
            blade = _strict_integer(payload["blade"], f"{context}.fixed_blade.blade")
            coefficient = _strict_integer(
                payload["coefficient"], f"{context}.fixed_blade.coefficient"
            )
            return self.fixed_blade(blade, coefficient)
        return super().parse(raw, context)


def load_identity(path: Path) -> Identity:
    original = base.Builder
    base.Builder = FixedBladeBuilder
    try:
        return base.load_identity(path)
    finally:
        base.Builder = original


def load_corpus(paths: Sequence[Path]) -> list[Identity]:
    identities = [load_identity(path) for path in paths]
    names = [identity.name for identity in identities]
    if len(names) != len(set(names)):
        raise IdentityError("corpus contains duplicate identity names")
    return identities


def evaluate_identity(
    identity: Identity,
    assignment: int,
) -> tuple[bool, int, int, int]:
    blade_count = 1 << identity.dimension
    variables = base.generate_assignment(identity, assignment)
    values: list[list[int]] = []

    for node in identity.nodes:
        if node.op == "var":
            assert node.variable is not None
            value = list(variables[node.variable])
        elif node.op == "scalar":
            assert node.value is not None
            value = [0] * blade_count
            value[0] = node.value
        elif node.op == "fixed_blade":
            assert node.value is not None and node.grade is not None
            value = [0] * blade_count
            value[node.grade] = node.value
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
            value = base._gp(values[node.args[0]], values[node.args[1]], identity)
        elif node.op == "wedge":
            value = base._wedge(values[node.args[0]], values[node.args[1]], identity)
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


def _emit_identity(identity: Identity, index: int) -> list[str]:
    """Delegate established emission, then retarget constant blade slots."""
    fixed_nodes: dict[int, tuple[int, int]] = {}
    transformed_nodes: list[Node] = []
    for node_index, node in enumerate(identity.nodes):
        if node.op != "fixed_blade":
            transformed_nodes.append(node)
            continue
        assert node.value is not None and node.grade is not None
        fixed_nodes[node_index] = (node.grade, node.value)
        transformed_nodes.append(dataclasses.replace(node, op="scalar"))
    transformed = dataclasses.replace(identity, nodes=tuple(transformed_nodes))
    lines = _BASE_EMIT_IDENTITY(transformed, index)
    for node_index, (blade, coefficient) in fixed_nodes.items():
        needle = f"        nodes[{node_index}].c[0] = {coefficient};"
        replacement = f"        nodes[{node_index}].c[{blade}] = {coefficient};"
        matches = [line_index for line_index, line in enumerate(lines) if line == needle]
        if len(matches) != 1:
            raise IdentityError(
                f"unable to locate unique generated constant assignment for node {node_index}"
            )
        lines[matches[0]] = replacement
    return lines


def emit_header(identities: Sequence[Identity], source_paths: Sequence[str]) -> str:
    original = base._emit_identity
    base._emit_identity = _emit_identity
    try:
        return base.emit_header(identities, source_paths)
    finally:
        base._emit_identity = original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile V4.1 exact identity IR with fixed basis-blade constants"
    )
    parser.add_argument("--identity", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--python-checks", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.python_checks < 0:
            raise IdentityError("--python-checks must be non-negative")
        identities = load_corpus(args.identity)
        generated = emit_header(identities, [path.as_posix() for path in args.identity])
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
    except (OSError, IdentityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: unable to read {args.output}: {exc}", file=sys.stderr)
            return 2
        if existing != generated:
            print(f"ERROR: {args.output} is stale", file=sys.stderr)
            return 1
        print(f"PASS: {args.output} matches {len(identities)} V4.1 identities")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"WROTE: {args.output}")
    print(f"IDENTITIES: {len(identities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
