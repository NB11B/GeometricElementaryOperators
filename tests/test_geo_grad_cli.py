#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import tempfile


def run(exe: pathlib.Path, *args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(exe), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if expect_success and completed.returncode != 0:
        raise AssertionError(
            f"command failed: {exe} {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(
            f"command unexpectedly succeeded: {exe} {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    options = parser.parse_args()
    exe = pathlib.Path(options.exe).resolve()
    if not exe.is_file():
        raise AssertionError(f"missing geo_grad executable: {exe}")

    version = run(exe, "--version")
    assert "geo_grad 7.1" in version.stdout
    assert "GEO_V7_ABI=0x00070000" in version.stdout

    with tempfile.TemporaryDirectory(prefix="geo-grad-cli-") as temporary:
        root = pathlib.Path(temporary)
        model = root / "model.geo"
        train_csv = root / "train.csv"
        checkpoint = root / "checkpoint.txt"
        predict_input = root / "predict.csv"
        predict_output = root / "predictions.csv"
        header = root / "trained_geo_model.h"

        initialized = run(exe, "init-example", str(model), str(train_csv))
        assert "GEO_GRAD_INIT_EXAMPLE: PASS" in initialized.stdout
        assert model.is_file()
        assert train_csv.is_file()

        checked = run(exe, "check", str(model))
        assert "GEO_GRAD_CHECK: PASS" in checked.stdout
        assert "adjoint_identity=PASS" in checked.stdout
        assert "unsupported_fallbacks=0" in checked.stdout
        assert "no_external_autograd=TRUE" in checked.stdout

        trained = run(exe, "train", str(model), str(train_csv), str(checkpoint))
        assert "GEO_GRAD_TRAIN: PASS" in trained.stdout
        assert "no_external_autograd=TRUE" in trained.stdout
        match = re.search(r"initial_loss=([^ ]+) final_loss=([^ ]+)", trained.stdout)
        if match is None:
            raise AssertionError(f"missing loss report: {trained.stdout}")
        initial_loss = float(match.group(1))
        final_loss = float(match.group(2))
        assert initial_loss > 0.0
        assert final_loss < initial_loss * 1.0e-5, (initial_loss, final_loss)
        assert checkpoint.is_file()
        checkpoint_text = checkpoint.read_text(encoding="utf-8")
        assert "geo_grad_checkpoint=1" in checkpoint_text
        assert "coefficients=" in checkpoint_text
        assert "first_moment=" in checkpoint_text
        assert "second_moment=" in checkpoint_text

        predict_input.write_text(
            "1,0,0,0\n"
            "0,1,0,0\n"
            "0,0,1,0\n"
            "0,0,0,1\n",
            encoding="utf-8",
        )
        predicted = run(
            exe,
            "predict",
            str(model),
            str(checkpoint),
            str(predict_input),
            str(predict_output),
        )
        assert "GEO_GRAD_PREDICT: PASS rows=4" in predicted.stdout
        prediction_rows = [
            [float(value) for value in row.split(",")]
            for row in predict_output.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert len(prediction_rows) == 4
        assert all(len(row) == 4 for row in prediction_rows)
        expected_first = [0.75, -1.25, 0.5, 1.0]
        for actual, expected in zip(prediction_rows[0], expected_first):
            assert abs(actual - expected) < 1.0e-3, (actual, expected)

        exported = run(
            exe,
            "export-c",
            str(model),
            str(checkpoint),
            str(header),
            "demo_model",
        )
        assert "GEO_GRAD_EXPORT_C: PASS" in exported.stdout
        header_text = header.read_text(encoding="utf-8")
        assert "demo_model_coefficients" in header_text
        assert "demo_model_signature" in header_text
        assert "demo_model_SIDE_RIGHT" in header_text

        malformed = root / "malformed.geo"
        malformed.write_text(
            "version=7.1\n"
            "model=multivector_gp\n"
            "dimension=2\n"
            "signature=1,0\n",
            encoding="utf-8",
        )
        rejected = run(exe, "check", str(malformed), expect_success=False)
        assert "invalid signature" in rejected.stderr

    print(
        "GEO_GRAD_CLI_TEST: PASS "
        "init=PASS check=PASS train=PASS predict=PASS export=PASS "
        "invalid_model=REJECTED no_external_autograd=TRUE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
