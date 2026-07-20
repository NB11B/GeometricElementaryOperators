#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


def run(command: list[str], ok: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ok and completed.returncode != 0:
        raise AssertionError(
            f"command failed: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not ok and completed.returncode == 0:
        raise AssertionError(
            f"command unexpectedly succeeded: {command}\nstdout:\n{completed.stdout}"
        )
    return completed


def gp_sign(a: int, b: int, signature: list[int]) -> int:
    sign = 1
    for axis, metric in enumerate(signature):
        if (a >> axis) & 1:
            if (b & ((1 << axis) - 1)).bit_count() & 1:
                sign = -sign
            if (b >> axis) & 1:
                sign *= metric
    return sign


def gp(a: list[float], b: list[float], signature: list[int]) -> list[float]:
    result = [0.0] * len(a)
    for left, av in enumerate(a):
        for right, bv in enumerate(b):
            result[left ^ right] += av * bv * gp_sign(left, right, signature)
    return result


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def tanh_mv(value: list[float]) -> list[float]:
    return [math.tanh(v) for v in value]


def write_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compile_model(
    compiler: pathlib.Path,
    model: pathlib.Path,
    ir: pathlib.Path,
    validate_first: bool = True,
) -> None:
    if validate_first:
        result = run([sys.executable, str(compiler), "validate", str(model)])
        if "GEO_MODEL_VALIDATE: PASS" not in result.stdout:
            raise AssertionError(result.stdout)
    result = run([sys.executable, str(compiler), "compile", str(model), str(ir)])
    if "GEO_MODEL_COMPILE: PASS" not in result.stdout:
        raise AssertionError(result.stdout)


def write_rows(path: pathlib.Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(",".join(format(value, ".17g") for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_checkpoint(path: pathlib.Path) -> dict[str, dict[str, list[float] | int]]:
    parameters: dict[str, dict[str, list[float] | int]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("parameter="):
            name, constraint = line.split("=", 1)[1].split(",", 1)
            current = name
            parameters[name] = {"constraint": int(constraint)}
        elif current is not None and line.startswith("value="):
            parameters[current]["value"] = [float(value) for value in line.split("=", 1)[1].split(",")]
        elif current is not None and line.startswith("first="):
            parameters[current]["first"] = [float(value) for value in line.split("=", 1)[1].split(",")]
        elif current is not None and line.startswith("second="):
            parameters[current]["second"] = [float(value) for value in line.split("=", 1)[1].split(",")]
        elif line == "end_parameter=1":
            current = None
    return parameters


def read_vectors(path: pathlib.Path) -> list[list[float]]:
    return [
        [float(value) for value in line.split(",")]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_close(actual: list[float], expected: list[float], tolerance: float = 1e-9) -> None:
    if len(actual) != len(expected):
        raise AssertionError((len(actual), len(expected)))
    for index, (left, right) in enumerate(zip(actual, expected)):
        if abs(left - right) > tolerance * (1.0 + abs(left) + abs(right)):
            raise AssertionError(f"coefficient {index}: {left} != {right}")


def residual_tail(prediction: str, target: str) -> list[dict[str, Any]]:
    return [
        {"name": "negative_target", "op": "scale", "input": target, "scalar": -1.0},
        {"name": "residual", "op": "add", "inputs": [prediction, "negative_target"]},
        {"name": "loss", "op": "squared_norm", "input": "residual"},
    ]


def simple_gp_document(
    epochs: int = 1,
    batch_size: int = 4,
    optimizer: str = "sgd",
    learning_rate: float = 1.0,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"name": "x", "op": "input"},
        {"name": "target", "op": "target"},
        {"name": "w", "op": "parameter", "init": "zeros"},
        {"name": "prediction", "op": "gp", "inputs": ["x", "w"]},
    ]
    nodes.extend(residual_tail("prediction", "target"))
    return {
        "version": 8,
        "dimension": 2,
        "signature": [1, 1],
        "nodes": nodes,
        "loss": "loss",
        "output": "prediction",
        "optimizer": {"type": optimizer, "learning_rate": learning_rate},
        "training": {"epochs": epochs, "batch_size": batch_size},
        "dataset": {"reset_column": False},
    }


def test_exact_batched_gp(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model = root / "batched_gp.json"
    ir = root / "batched_gp.geoir"
    data = root / "batched_gp.csv"
    checkpoint = root / "batched_gp.checkpoint"
    prediction_input = root / "batched_gp_input.csv"
    prediction_output = root / "batched_gp_output.csv"
    header = root / "batched_gp.h"
    truth = [0.75, -1.25, 0.5, 1.0]
    signature = [1, 1]
    write_json(model, simple_gp_document())
    rows: list[list[float]] = []
    inputs: list[list[float]] = []
    for blade in range(4):
        value = [0.0] * 4
        value[blade] = 1.0
        inputs.append(value)
        rows.append(value + gp(value, truth, signature))
    write_rows(data, rows)
    write_rows(prediction_input, inputs)
    compile_model(compiler, model, ir)
    if "GEO_CYCLE_CHECK: PASS" not in run([str(runner), "check", str(ir)]).stdout:
        raise AssertionError("check marker missing")
    if "GEO_CYCLE_TRAIN: PASS" not in run(
        [str(runner), "train", str(ir), str(data), str(checkpoint)]
    ).stdout:
        raise AssertionError("train marker missing")
    assert_close(parse_checkpoint(checkpoint)["w"]["value"], truth, 1e-11)  # type: ignore[arg-type]
    run([str(runner), "predict", str(ir), str(checkpoint), str(prediction_input), str(prediction_output)])
    expected = [gp(value, truth, signature) for value in inputs]
    for actual, wanted in zip(read_vectors(prediction_output), expected):
        assert_close(actual, wanted, 1e-11)
    run([str(runner), "export-c", str(ir), str(checkpoint), str(header), "batched_gp"])
    text = header.read_text(encoding="utf-8")
    if "batched_gp_w" not in text or "BLADE_COUNT 4u" not in text:
        raise AssertionError("exported C header is incomplete")


def test_multiple_parameters(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model = root / "multiple_parameters.json"
    ir = root / "multiple_parameters.geoir"
    data = root / "multiple_parameters.csv"
    checkpoint = root / "multiple_parameters.checkpoint"
    signature = [1, 1]
    truth1 = [0.4, -0.2, 0.7, 0.1]
    truth2 = [-0.3, 0.8, -0.1, 0.5]
    nodes: list[dict[str, Any]] = [
        {"name": "x1", "op": "input"},
        {"name": "x2", "op": "input"},
        {"name": "target", "op": "target"},
        {"name": "w1", "op": "parameter", "init": "zeros"},
        {"name": "w2", "op": "parameter", "init": "zeros"},
        {"name": "p1", "op": "gp", "inputs": ["x1", "w1"]},
        {"name": "p2", "op": "gp", "inputs": ["x2", "w2"]},
        {"name": "prediction", "op": "add", "inputs": ["p1", "p2"]},
    ]
    nodes.extend(residual_tail("prediction", "target"))
    document = {
        "version": 8,
        "dimension": 2,
        "signature": signature,
        "nodes": nodes,
        "loss": "loss",
        "output": "prediction",
        "optimizer": {"type": "sgd", "learning_rate": 2.0},
        "training": {"epochs": 1, "batch_size": 8},
        "dataset": {"reset_column": False},
    }
    write_json(model, document)
    rows = []
    zero = [0.0] * 4
    for blade in range(4):
        basis = [0.0] * 4
        basis[blade] = 1.0
        rows.append(basis + zero + gp(basis, truth1, signature))
    for blade in range(4):
        basis = [0.0] * 4
        basis[blade] = 1.0
        rows.append(zero + basis + gp(basis, truth2, signature))
    write_rows(data, rows)
    compile_model(compiler, model, ir)
    run([str(runner), "train", str(ir), str(data), str(checkpoint)])
    parameters = parse_checkpoint(checkpoint)
    assert_close(parameters["w1"]["value"], truth1, 1e-11)  # type: ignore[arg-type]
    assert_close(parameters["w2"]["value"], truth2, 1e-11)  # type: ignore[arg-type]


def test_nonlinear_multilayer(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model = root / "nonlinear.json"
    ir = root / "nonlinear.geoir"
    data = root / "nonlinear.csv"
    checkpoint = root / "nonlinear.checkpoint"
    input_file = root / "nonlinear_input.csv"
    output_file = root / "nonlinear_output.csv"
    truth = [0.45, -0.3]
    signature = [1]
    nodes: list[dict[str, Any]] = [
        {"name": "x", "op": "input"},
        {"name": "target", "op": "target"},
        {"name": "w", "op": "parameter", "init": "zeros"},
        {"name": "linear", "op": "gp", "inputs": ["x", "w"]},
        {"name": "prediction", "op": "tanh", "input": "linear"},
    ]
    nodes.extend(residual_tail("prediction", "target"))
    document = {
        "version": 8,
        "dimension": 1,
        "signature": signature,
        "nodes": nodes,
        "loss": "loss",
        "output": "prediction",
        "optimizer": {"type": "adam", "learning_rate": 0.04},
        "training": {"epochs": 240, "batch_size": 4},
        "dataset": {"reset_column": False},
    }
    samples = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, -0.25],
        [-0.75, 0.4],
        [0.2, 0.8],
        [-0.3, -0.6],
        [0.9, 0.2],
        [-0.4, 0.9],
    ]
    targets = [tanh_mv(gp(value, truth, signature)) for value in samples]
    write_json(model, document)
    write_rows(data, [value + target for value, target in zip(samples, targets)])
    write_rows(input_file, samples)
    compile_model(compiler, model, ir)
    train_output = run([str(runner), "train", str(ir), str(data), str(checkpoint)]).stdout
    if "GEO_CYCLE_TRAIN: PASS" not in train_output:
        raise AssertionError(train_output)
    run([str(runner), "predict", str(ir), str(checkpoint), str(input_file), str(output_file)])
    predictions = read_vectors(output_file)
    mse = sum(
        (actual - wanted) ** 2
        for row, target in zip(predictions, targets)
        for actual, wanted in zip(row, target)
    ) / (len(samples) * 2)
    if mse > 2e-5:
        raise AssertionError(f"nonlinear model did not converge: mse={mse}")


def test_recurrent_cell(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model = root / "recurrent.json"
    ir = root / "recurrent.geoir"
    data = root / "recurrent.csv"
    checkpoint = root / "recurrent.checkpoint"
    input_file = root / "recurrent_input.csv"
    output_file = root / "recurrent_output.csv"
    signature = [1]
    truth = [0.3, -0.15]
    nodes: list[dict[str, Any]] = [
        {"name": "x", "op": "input"},
        {"name": "target", "op": "target"},
        {"name": "memory", "op": "state", "init": "zeros"},
        {"name": "weight", "op": "parameter", "init": "zeros"},
        {"name": "transformed", "op": "gp", "inputs": ["x", "weight"]},
        {"name": "preactivation", "op": "add", "inputs": ["transformed", "memory"]},
        {"name": "prediction", "op": "tanh", "input": "preactivation"},
    ]
    nodes.extend(residual_tail("prediction", "target"))
    document = {
        "version": 8,
        "dimension": 1,
        "signature": signature,
        "nodes": nodes,
        "loss": "loss",
        "output": "prediction",
        "state_updates": {"memory": "prediction"},
        "optimizer": {"type": "adam", "learning_rate": 0.025},
        "training": {"epochs": 220, "batch_size": 1, "reset_state_each_epoch": True},
        "dataset": {"reset_column": True},
    }
    sequence = [
        [0.4, -0.2],
        [-0.1, 0.5],
        [0.8, 0.1],
        [-0.3, -0.4],
        [0.2, 0.7],
        [-0.6, 0.3],
    ]
    memory = [0.0, 0.0]
    targets: list[list[float]] = []
    for value in sequence:
        memory = tanh_mv(add(gp(value, truth, signature), memory))
        targets.append(memory)
    write_json(model, document)
    write_rows(
        data,
        [([1.0 if index == 0 else 0.0] + value + target) for index, (value, target) in enumerate(zip(sequence, targets))],
    )
    write_rows(
        input_file,
        [([1.0 if index == 0 else 0.0] + value) for index, value in enumerate(sequence)],
    )
    compile_model(compiler, model, ir)
    run([str(runner), "train", str(ir), str(data), str(checkpoint)])
    run([str(runner), "predict", str(ir), str(checkpoint), str(input_file), str(output_file)])
    predictions = read_vectors(output_file)
    mse = sum(
        (actual - wanted) ** 2
        for row, target in zip(predictions, targets)
        for actual, wanted in zip(row, target)
    ) / (len(sequence) * 2)
    if mse > 1e-4:
        raise AssertionError(f"recurrent cell did not converge: mse={mse}")


def test_constraint(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model = root / "constraint.json"
    ir = root / "constraint.geoir"
    data = root / "constraint.csv"
    checkpoint = root / "constraint.checkpoint"
    nodes: list[dict[str, Any]] = [
        {"name": "x", "op": "input"},
        {"name": "target", "op": "target"},
        {
            "name": "direction",
            "op": "parameter",
            "init": [0.6, 0.8, 0.0, 0.0],
            "constraint": "unit_euclidean",
        },
    ]
    nodes.extend(residual_tail("direction", "target"))
    document = {
        "version": 8,
        "dimension": 2,
        "signature": [1, 1],
        "nodes": nodes,
        "loss": "loss",
        "output": "direction",
        "optimizer": {"type": "sgd", "learning_rate": 0.3},
        "training": {"epochs": 30, "batch_size": 1},
        "dataset": {"reset_column": False},
    }
    write_json(model, document)
    write_rows(data, [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    compile_model(compiler, model, ir)
    run([str(runner), "train", str(ir), str(data), str(checkpoint)])
    value = parse_checkpoint(checkpoint)["direction"]["value"]  # type: ignore[assignment]
    assert isinstance(value, list)
    norm = math.sqrt(sum(component * component for component in value))
    if abs(norm - 1.0) > 1e-12 or value[0] < 0.999:
        raise AssertionError((norm, value))


def test_large_graph(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> int:
    model = root / "large_graph.json"
    ir = root / "large_graph.geoir"
    nodes: list[dict[str, Any]] = [
        {"name": "x", "op": "input", "init": [1.0, 0.0]},
        {"name": "target", "op": "target"},
        {"name": "w", "op": "parameter", "init": [0.2, -0.1]},
        {"name": "layer_0", "op": "gp", "inputs": ["x", "w"]},
    ]
    previous = "layer_0"
    for index in range(1, 321):
        name = f"layer_{index}"
        nodes.append({"name": name, "op": "scale", "input": previous, "scalar": 0.9999})
        previous = name
    nodes.extend(residual_tail(previous, "target"))
    document = {
        "version": 8,
        "dimension": 1,
        "signature": [1],
        "nodes": nodes,
        "loss": "loss",
        "output": previous,
        "optimizer": {"type": "sgd", "learning_rate": 0.01},
        "training": {"epochs": 1, "batch_size": 1},
    }
    write_json(model, document)
    compile_model(compiler, model, ir)
    result = run([str(runner), "check", str(ir)])
    if "GEO_CYCLE_CHECK: PASS" not in result.stdout:
        raise AssertionError(result.stdout)
    return len(nodes)


def test_streaming_dataset(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> int:
    model = root / "stream.json"
    ir = root / "stream.geoir"
    data = root / "stream.csv"
    checkpoint = root / "stream.checkpoint"
    document = simple_gp_document(epochs=1, batch_size=64, optimizer="sgd", learning_rate=0.5)
    write_json(model, document)
    truth = [0.2, -0.3, 0.1, 0.4]
    row = [1.0, 0.0, 0.0, 0.0] + truth
    rows = [row for _ in range(2048)]
    write_rows(data, rows)
    compile_model(compiler, model, ir)
    run([str(runner), "train", str(ir), str(data), str(checkpoint)])
    value = parse_checkpoint(checkpoint)["w"]["value"]  # type: ignore[assignment]
    assert isinstance(value, list)
    assert_close(value, truth, 1e-8)
    return len(rows)


def test_resume_equivalence(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> None:
    model10 = root / "resume10.json"
    model20 = root / "resume20.json"
    ir10 = root / "resume10.geoir"
    ir20 = root / "resume20.geoir"
    data = root / "resume.csv"
    checkpoint10 = root / "checkpoint10"
    resumed = root / "resumed"
    fresh20 = root / "fresh20"
    signature = [1, 1]
    truth = [0.3, -0.2, 0.4, 0.1]
    rows = []
    for blade in range(4):
        basis = [0.0] * 4
        basis[blade] = 1.0
        rows.append(basis + gp(basis, truth, signature))
    write_rows(data, rows)
    write_json(model10, simple_gp_document(epochs=10, batch_size=4, optimizer="adam", learning_rate=0.03))
    write_json(model20, simple_gp_document(epochs=20, batch_size=4, optimizer="adam", learning_rate=0.03))
    compile_model(compiler, model10, ir10)
    compile_model(compiler, model20, ir20)
    run([str(runner), "train", str(ir10), str(data), str(checkpoint10)])
    run([str(runner), "train", str(ir10), str(data), str(resumed), str(checkpoint10)])
    run([str(runner), "train", str(ir20), str(data), str(fresh20)])
    if resumed.read_bytes() != fresh20.read_bytes():
        raise AssertionError("checkpoint resume is not byte-identical to continuous training")


def test_adversarial(
    compiler: pathlib.Path,
    runner: pathlib.Path,
    root: pathlib.Path,
) -> int:
    base = simple_gp_document()
    cases: list[dict[str, Any]] = []
    duplicate = json.loads(json.dumps(base))
    duplicate["nodes"][1]["name"] = "x"
    cases.append(duplicate)
    forward = json.loads(json.dumps(base))
    forward["nodes"][0] = {"name": "bad", "op": "add", "inputs": ["x", "w"]}
    cases.append(forward)
    bad_dimension = json.loads(json.dumps(base))
    bad_dimension["dimension"] = 7
    bad_dimension["signature"] = [1] * 7
    cases.append(bad_dimension)
    bad_op = json.loads(json.dumps(base))
    bad_op["nodes"][3]["op"] = "magic"
    cases.append(bad_op)
    unknown = json.loads(json.dumps(base))
    unknown["unknown"] = True
    cases.append(unknown)
    missing_state_update = json.loads(json.dumps(base))
    missing_state_update["nodes"].insert(2, {"name": "memory", "op": "state"})
    cases.append(missing_state_update)
    bad_constraint = json.loads(json.dumps(base))
    bad_constraint["nodes"][2]["constraint"] = "hyperbolic_magic"
    cases.append(bad_constraint)
    for index, document in enumerate(cases):
        path = root / f"bad_{index}.json"
        write_json(path, document)
        run([sys.executable, str(compiler), "validate", str(path)], ok=False)
    corrupt = root / "corrupt.geoir"
    corrupt.write_text("geo_model_ir=8\ndimension=2\n", encoding="utf-8")
    run([str(runner), "check", str(corrupt)], ok=False)
    return len(cases) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="geo-cycle-v8-") as temporary:
        root = pathlib.Path(temporary)
        test_exact_batched_gp(args.compiler, args.runner, root)
        test_multiple_parameters(args.compiler, args.runner, root)
        test_nonlinear_multilayer(args.compiler, args.runner, root)
        test_recurrent_cell(args.compiler, args.runner, root)
        test_constraint(args.compiler, args.runner, root)
        dynamic_nodes = test_large_graph(args.compiler, args.runner, root)
        streaming_rows = test_streaming_dataset(args.compiler, args.runner, root)
        test_resume_equivalence(args.compiler, args.runner, root)
        adversarial = test_adversarial(args.compiler, args.runner, root)
    print(
        "GEO_CYCLE_V8_CLI_TEST: PASS arbitrary_json=PASS multiple_parameters=PASS "
        "multilayer=PASS batching=PASS streaming_rows="
        f"{streaming_rows} recurrent=PASS checkpoint_resume=BYTE_IDENTICAL "
        f"export=PASS constraints=PASS dynamic_nodes={dynamic_nodes} "
        f"adversarial_cases={adversarial} external_autograd=NONE cuda=DEFERRED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
