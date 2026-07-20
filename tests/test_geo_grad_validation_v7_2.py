#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile


def run(exe: pathlib.Path, *args: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run([str(exe), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if ok and cp.returncode != 0:
        raise AssertionError(f"command failed: {args}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    if not ok and cp.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {args}\nstdout:\n{cp.stdout}")
    return cp


def gp_sign(a: int, b: int, sig: list[int]) -> int:
    sign = 1
    for i, metric in enumerate(sig):
        if (a >> i) & 1:
            if (b & ((1 << i) - 1)).bit_count() & 1:
                sign = -sign
            if (b >> i) & 1:
                sign *= metric
    return sign


def gp(a: list[float], b: list[float], sig: list[int]) -> list[float]:
    out = [0.0] * len(a)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i ^ j] += av * bv * gp_sign(i, j, sig)
    return out


def write_model(path: pathlib.Path, n: int, q: int, side: str, optimizer: str = "sgd", epochs: int = 1) -> None:
    sig = [1] * (n - q) + [-1] * q
    path.write_text("\n".join([
        "version=7.1", "model=multivector_gp", f"dimension={n}",
        "signature=" + ",".join(map(str, sig)), f"side={side}",
        f"optimizer={optimizer}", "learning_rate=1.0" if optimizer == "sgd" else "learning_rate=0.03",
        f"epochs={epochs}", "beta1=0.9", "beta2=0.999", "epsilon=1e-8", "seed=17", ""
    ]), encoding="utf-8")


def truth_vector(n: int) -> list[float]:
    return [((i * 17 + n * 11) % 23 - 11) / 13.0 for i in range(1 << n)]


def write_dataset(path: pathlib.Path, n: int, q: int, side: str) -> list[float]:
    sig = [1] * (n - q) + [-1] * q
    count = 1 << n
    truth = truth_vector(n)
    rows = []
    for basis in range(count):
        x = [0.0] * count
        x[basis] = 1.0
        y = gp(x, truth, sig) if side == "right" else gp(truth, x, sig)
        rows.append(",".join(format(v, ".17g") for v in x + y))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return truth


def parse_checkpoint(path: pathlib.Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = value
    return result


def assert_close(actual: list[float], expected: list[float], tol: float = 2e-11) -> None:
    if len(actual) != len(expected):
        raise AssertionError((len(actual), len(expected)))
    for i, (a, b) in enumerate(zip(actual, expected)):
        if abs(a - b) > tol * (1 + abs(a) + abs(b)):
            raise AssertionError(f"coefficient {i}: {a} != {b}")


def exhaustive_matrix(exe: pathlib.Path, root: pathlib.Path) -> int:
    cases = 0
    for n in range(1, 7):
        for q in range(n + 1):
            for side in ("left", "right"):
                stem = f"cl_{n-q}_{q}_{side}"
                model, data = root / f"{stem}.geo", root / f"{stem}.csv"
                checkpoint, predictions, header = root / f"{stem}.checkpoint", root / f"{stem}.pred.csv", root / f"{stem}.h"
                write_model(model, n, q, side)
                truth = write_dataset(data, n, q, side)
                if "GEO_GRAD_CHECK: PASS" not in run(exe, "check", str(model)).stdout:
                    raise AssertionError(stem)
                if "GEO_GRAD_TRAIN: PASS" not in run(exe, "train", str(model), str(data), str(checkpoint)).stdout:
                    raise AssertionError(stem)
                coeffs = [float(v) for v in parse_checkpoint(checkpoint)["coefficients"].split(",")]
                assert_close(coeffs, truth)
                run(exe, "predict", str(model), str(checkpoint), str(data), str(predictions))
                if len(predictions.read_text(encoding="utf-8").strip().splitlines()) != 1 << n:
                    raise AssertionError(stem)
                run(exe, "export-c", str(model), str(checkpoint), str(header), f"geo_{n}_{q}_{side}")
                text = header.read_text(encoding="utf-8")
                if "static const double" not in text or f"BLADE_COUNT {1 << n}u" not in text:
                    raise AssertionError(stem)
                cases += 1
    return cases


def adversarial_cases(exe: pathlib.Path, root: pathlib.Path) -> int:
    valid = root / "valid.geo"
    data = root / "valid.csv"
    checkpoint = root / "valid.checkpoint"
    write_model(valid, 2, 0, "right")
    write_dataset(data, 2, 0, "right")
    run(exe, "train", str(valid), str(data), str(checkpoint))
    bad_models = {
        "empty": "", "missing_dimension": "version=7.1\nmodel=multivector_gp\nsignature=1,1\n",
        "dimension_zero": "version=7.1\nmodel=multivector_gp\ndimension=0\nsignature=\n",
        "dimension_seven": "version=7.1\nmodel=multivector_gp\ndimension=7\nsignature=1,1,1,1,1,1,1\n",
        "bad_signature_zero": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,0\n",
        "short_signature": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1\n",
        "unknown_key": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,1\nwat=1\n",
        "bad_version": "version=99\nmodel=multivector_gp\ndimension=2\nsignature=1,1\n",
        "bad_model": "version=7.1\nmodel=tensor\ndimension=2\nsignature=1,1\n",
        "bad_side": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,1\nside=center\n",
        "bad_optimizer": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,1\noptimizer=magic\n",
        "nan_lr": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,1\nlearning_rate=nan\n",
        "zero_epochs": "version=7.1\nmodel=multivector_gp\ndimension=2\nsignature=1,1\nepochs=0\n",
        "missing_equals": "version=7.1\nmodel=multivector_gp\ndimension 2\nsignature=1,1\n",
    }
    count = 0
    for name, text in bad_models.items():
        path = root / f"bad_{name}.geo"
        path.write_text(text, encoding="utf-8")
        run(exe, "check", str(path), ok=False)
        count += 1
    bad_datasets = {
        "empty": "", "short": "1,0,0\n", "long": "1,0,0,0,0,0,0,0,9\n",
        "nan": "1,0,nan,0,0,0,0,0\n", "inf": "1,0,inf,0,0,0,0,0\n",
        "text": "hello,world\n", "trailing_comma": "1,0,0,0,0,0,0,0,\n",
    }
    for name, text in bad_datasets.items():
        path = root / f"bad_{name}.csv"
        path.write_text(text, encoding="utf-8")
        run(exe, "train", str(valid), str(path), str(root / f"bad_{name}.checkpoint"), ok=False)
        count += 1
    base = checkpoint.read_text(encoding="utf-8")
    corrupted = {
        "empty": "", "bad_header": "geo_grad_checkpoint=2\n",
        "wrong_dimension": base.replace("dimension=2", "dimension=3"),
        "wrong_side": base.replace("side=right", "side=left"),
        "nan_coeff": base.replace("coefficients=", "coefficients=nan,"),
        "truncated": "geo_grad_checkpoint=1\ndimension=2\nsignature=1,1\nside=right\n",
        "unknown": base + "unknown=1\n",
    }
    basis_input = root / "input.csv"
    basis_input.write_text("1,0,0,0\n", encoding="utf-8")
    for name, text in corrupted.items():
        path = root / f"corrupt_{name}.checkpoint"
        path.write_text(text, encoding="utf-8")
        run(exe, "predict", str(valid), str(path), str(basis_input), str(root / f"out_{name}.csv"), ok=False)
        count += 1
    run(exe, "export-c", str(valid), str(checkpoint), str(root / "x.h"), "9bad", ok=False)
    return count + 1


def deterministic_replay(exe: pathlib.Path, root: pathlib.Path) -> None:
    model, data = root / "det.geo", root / "det.csv"
    write_model(model, 4, 2, "left", optimizer="adam", epochs=40)
    write_dataset(data, 4, 2, "left")
    a, b = root / "a.checkpoint", root / "b.checkpoint"
    run(exe, "train", str(model), str(data), str(a))
    run(exe, "train", str(model), str(data), str(b))
    if a.read_bytes() != b.read_bytes():
        raise AssertionError("seeded checkpoint replay is not byte-identical")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="geo-grad-v7-2-") as tmp:
        root = pathlib.Path(tmp)
        matrix = exhaustive_matrix(args.exe, root)
        adversarial = adversarial_cases(args.exe, root)
        deterministic_replay(args.exe, root)
    print(f"GEO_GRAD_V7_2_CLI_VALIDATION: PASS matrix_cases={matrix} adversarial_cases={adversarial} checkpoint_replay=BYTE_IDENTICAL prediction=PASS export=PASS external_autograd=NONE cuda=DEFERRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
