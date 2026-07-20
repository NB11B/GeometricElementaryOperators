#!/usr/bin/env python3
"""Compile arbitrary GEO V8 JSON graphs into deterministic runner IR."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import tempfile
from dataclasses import dataclass
from typing import Any

VERSION = 8
INVALID = 2**32 - 1
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

KINDS = {
    "input": 1,
    "target": 2,
    "parameter": 3,
    "constant": 4,
    "state": 5,
    "add": 6,
    "scale": 7,
    "gp": 8,
    "reverse": 9,
    "grade_project": 10,
    "grade_involution": 11,
    "clifford_conjugate": 12,
    "hadamard": 13,
    "tanh": 14,
    "sigmoid": 15,
    "normalize": 16,
    "squared_norm": 17,
}

LEAFS = {"input", "target", "parameter", "constant", "state"}
UNARY = {
    "scale",
    "reverse",
    "grade_project",
    "grade_involution",
    "clifford_conjugate",
    "tanh",
    "sigmoid",
    "normalize",
    "squared_norm",
}
BINARY = {"add", "gp", "hadamard"}
CONSTRAINTS = {
    "none": 0,
    "unit_euclidean": 1,
    "unit_vector_metric": 2,
    "even_versor": 3,
}
OPTIMIZERS = {"sgd": 1, "adam": 2}


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledNode:
    name: str
    op: str
    left: int
    right: int
    scalar: float
    grade: int
    constraint: int
    requires_grad: int
    value: tuple[float, ...]


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelError(f"{label} must be finite")
    return result


def positive_number(value: Any, label: str) -> float:
    result = finite_number(value, label)
    if result <= 0.0:
        raise ModelError(f"{label} must be positive")
    return result


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def parse_initializer(spec: Any, blade_count: int, seed: int, name: str) -> tuple[float, ...]:
    if isinstance(spec, list):
        if len(spec) != blade_count:
            raise ModelError(f"initializer for {name} must contain {blade_count} coefficients")
        return tuple(finite_number(v, f"initializer {name}[{i}]") for i, v in enumerate(spec))
    if spec is None or spec == "zeros":
        return tuple(0.0 for _ in range(blade_count))
    if spec == "ones":
        return tuple(1.0 for _ in range(blade_count))
    if not isinstance(spec, str):
        raise ModelError(f"invalid initializer for {name}")
    parts = spec.split(":")
    if parts[0] == "scalar" and len(parts) == 2:
        values = [0.0] * blade_count
        values[0] = finite_number(float(parts[1]), f"initializer {name}")
        return tuple(values)
    if parts[0] == "basis" and len(parts) == 3:
        index = int(parts[1])
        if index < 0 or index >= blade_count:
            raise ModelError(f"basis index for {name} is out of range")
        values = [0.0] * blade_count
        values[index] = finite_number(float(parts[2]), f"initializer {name}")
        return tuple(values)
    if parts[0] == "random" and len(parts) in (1, 2, 3):
        local_seed = seed
        scale = 0.1
        if len(parts) >= 2 and parts[1]:
            local_seed = int(parts[1])
        if len(parts) == 3:
            scale = positive_number(float(parts[2]), f"random scale for {name}")
        rng = random.Random((local_seed << 32) ^ sum(ord(ch) for ch in name))
        return tuple(rng.uniform(-scale, scale) for _ in range(blade_count))
    raise ModelError(f"unsupported initializer '{spec}' for {name}")


def validate_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ModelError("model root must be an object")
    allowed_root = {
        "version",
        "dimension",
        "signature",
        "seed",
        "nodes",
        "loss",
        "output",
        "state_updates",
        "optimizer",
        "training",
        "dataset",
        "metadata",
    }
    unknown = sorted(set(document) - allowed_root)
    if unknown:
        raise ModelError(f"unknown root keys: {', '.join(unknown)}")
    if document.get("version") != VERSION:
        raise ModelError("version must be 8")
    dimension = document.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 6:
        raise ModelError("dimension must be an integer from 1 through 6")
    signature = document.get("signature")
    if not isinstance(signature, list) or len(signature) != dimension or any(v not in (-1, 1) for v in signature):
        raise ModelError("signature must contain exactly dimension entries, each +1 or -1")
    seed = document.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ModelError("seed must be a nonnegative integer")
    nodes = document.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ModelError("nodes must be a nonempty array")
    optimizer = document.get("optimizer", {})
    if not isinstance(optimizer, dict):
        raise ModelError("optimizer must be an object")
    training = document.get("training", {})
    if not isinstance(training, dict):
        raise ModelError("training must be an object")
    dataset = document.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ModelError("dataset must be an object")
    return {
        "dimension": dimension,
        "signature": signature,
        "seed": seed,
        "nodes": nodes,
        "loss": document.get("loss"),
        "output": document.get("output"),
        "state_updates": document.get("state_updates", {}),
        "optimizer": optimizer,
        "training": training,
        "dataset": dataset,
    }


def compile_model(document: Any) -> tuple[list[str], dict[str, int]]:
    validated = validate_document(document)
    dimension = validated["dimension"]
    signature = validated["signature"]
    seed = validated["seed"]
    blade_count = 1 << dimension
    names: dict[str, int] = {}
    compiled: list[CompiledNode] = []

    for index, raw in enumerate(validated["nodes"]):
        if not isinstance(raw, dict):
            raise ModelError(f"node {index} must be an object")
        allowed = {"name", "op", "inputs", "input", "scalar", "grade", "constraint", "requires_grad", "init"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ModelError(f"node {index} has unknown keys: {', '.join(unknown)}")
        name = raw.get("name")
        op = raw.get("op")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ModelError(f"node {index} has an invalid name")
        if name in names:
            raise ModelError(f"duplicate node name '{name}'")
        if op not in KINDS:
            raise ModelError(f"node {name} has unsupported op '{op}'")
        left = right = INVALID
        scalar = finite_number(raw.get("scalar", 0.0), f"node {name} scalar")
        grade = raw.get("grade", 0)
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0 or grade > dimension:
            raise ModelError(f"node {name} grade is invalid")
        constraint_name = raw.get("constraint", "none")
        if constraint_name not in CONSTRAINTS:
            raise ModelError(f"node {name} has invalid constraint")
        constraint = CONSTRAINTS[constraint_name]
        requires_grad = raw.get("requires_grad", op == "parameter")
        if not isinstance(requires_grad, bool):
            raise ModelError(f"node {name} requires_grad must be boolean")
        value = parse_initializer(raw.get("init", "zeros"), blade_count, seed, name)

        if op in LEAFS:
            if raw.get("inputs") is not None or raw.get("input") is not None:
                raise ModelError(f"leaf node {name} cannot have inputs")
            if op != "parameter" and constraint != 0:
                raise ModelError(f"constraint is valid only on parameter node {name}")
        elif op in UNARY:
            input_name = raw.get("input")
            if not isinstance(input_name, str) or input_name not in names:
                raise ModelError(f"node {name} must reference one earlier input")
            left = names[input_name]
            value = tuple(0.0 for _ in range(blade_count))
            constraint = 0
            requires_grad = False
        elif op in BINARY:
            inputs = raw.get("inputs")
            if not isinstance(inputs, list) or len(inputs) != 2 or any(not isinstance(v, str) for v in inputs):
                raise ModelError(f"node {name} must have two inputs")
            if any(v not in names for v in inputs):
                raise ModelError(f"node {name} references an unknown or forward input")
            left, right = names[inputs[0]], names[inputs[1]]
            value = tuple(0.0 for _ in range(blade_count))
            constraint = 0
            requires_grad = False
        else:
            raise AssertionError(op)

        if op == "normalize" and scalar == 0.0:
            scalar = 1e-12
        if op != "grade_project" and "grade" in raw:
            raise ModelError(f"grade is only valid for grade_project node {name}")
        if op != "scale" and op != "normalize" and "scalar" in raw:
            raise ModelError(f"scalar is not valid for node {name}")

        names[name] = index
        compiled.append(
            CompiledNode(name, op, left, right, scalar, grade, constraint, int(requires_grad), value)
        )

    loss_name = validated["loss"]
    output_name = validated["output"]
    if not isinstance(loss_name, str) or loss_name not in names:
        raise ModelError("loss must name an existing node")
    if not isinstance(output_name, str) or output_name not in names:
        raise ModelError("output must name an existing node")

    state_updates = validated["state_updates"]
    if not isinstance(state_updates, dict):
        raise ModelError("state_updates must be an object")
    state_bindings: list[tuple[int, int]] = []
    for state_name, source_name in sorted(state_updates.items()):
        if state_name not in names or source_name not in names:
            raise ModelError("state update references an unknown node")
        if compiled[names[state_name]].op != "state":
            raise ModelError(f"state update key '{state_name}' is not a state")
        state_bindings.append((names[state_name], names[source_name]))
    declared_states = {i for i, node in enumerate(compiled) if node.op == "state"}
    if declared_states != {state for state, _ in state_bindings}:
        missing = [compiled[i].name for i in sorted(declared_states - {s for s, _ in state_bindings})]
        raise ModelError(f"every state requires an update binding: {', '.join(missing)}")

    optimizer = validated["optimizer"]
    optimizer_name = optimizer.get("type", "adam")
    if optimizer_name not in OPTIMIZERS:
        raise ModelError("optimizer.type must be sgd or adam")
    learning_rate = positive_number(optimizer.get("learning_rate", 0.01), "optimizer.learning_rate")
    beta1 = finite_number(optimizer.get("beta1", 0.9), "optimizer.beta1")
    beta2 = finite_number(optimizer.get("beta2", 0.999), "optimizer.beta2")
    epsilon = positive_number(optimizer.get("epsilon", 1e-8), "optimizer.epsilon")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ModelError("optimizer beta values must be in [0,1)")

    training = validated["training"]
    epochs = positive_int(training.get("epochs", 1), "training.epochs")
    batch_size = positive_int(training.get("batch_size", 1), "training.batch_size")
    reset_state_each_epoch = training.get("reset_state_each_epoch", True)
    if not isinstance(reset_state_each_epoch, bool):
        raise ModelError("training.reset_state_each_epoch must be boolean")

    dataset = validated["dataset"]
    reset_column = dataset.get("reset_column", bool(declared_states))
    if not isinstance(reset_column, bool):
        raise ModelError("dataset.reset_column must be boolean")

    lines = [
        "geo_model_ir=8",
        f"dimension={dimension}",
        "signature=" + ",".join(str(v) for v in signature),
        f"optimizer={OPTIMIZERS[optimizer_name]}",
        f"learning_rate={learning_rate:.17g}",
        f"beta1={beta1:.17g}",
        f"beta2={beta2:.17g}",
        f"epsilon={epsilon:.17g}",
        f"epochs={epochs}",
        f"batch_size={batch_size}",
        f"reset_state_each_epoch={int(reset_state_each_epoch)}",
        f"reset_column={int(reset_column)}",
        f"node_count={len(compiled)}",
        f"loss={names[loss_name]}",
        f"output={names[output_name]}",
    ]
    for index, node in enumerate(compiled):
        lines.append(
            "node=" + ",".join(
                [
                    str(index),
                    str(KINDS[node.op]),
                    node.name,
                    str(node.left),
                    str(node.right),
                    f"{node.scalar:.17g}",
                    str(node.grade),
                    str(node.constraint),
                    str(node.requires_grad),
                ]
            )
        )
        lines.append("value=" + ",".join(f"{v:.17g}" for v in node.value))
    for state, source in state_bindings:
        lines.append(f"state_update={state},{source}")
    lines.append("end=1")

    summary = {
        "nodes": len(compiled),
        "parameters": sum(node.op == "parameter" for node in compiled),
        "inputs": sum(node.op == "input" for node in compiled),
        "targets": sum(node.op == "target" for node in compiled),
        "states": len(declared_states),
        "blade_count": blade_count,
        "batch_size": batch_size,
    }
    return lines, summary


def load_and_compile(path: pathlib.Path) -> tuple[list[str], dict[str, int]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(str(exc)) from exc
    return compile_model(document)


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("model", type=pathlib.Path)
    compile_command = sub.add_parser("compile")
    compile_command.add_argument("model", type=pathlib.Path)
    compile_command.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        lines, summary = load_and_compile(args.model)
        if args.command == "compile":
            atomic_write(args.output, "\n".join(lines) + "\n")
            print(f"GEO_MODEL_COMPILE: PASS output={args.output} " + " ".join(f"{k}={v}" for k, v in summary.items()))
        else:
            print("GEO_MODEL_VALIDATE: PASS " + " ".join(f"{k}={v}" for k, v in summary.items()))
        return 0
    except (ModelError, OSError, ValueError) as exc:
        print(f"geo_model: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
