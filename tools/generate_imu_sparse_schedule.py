#!/usr/bin/env python3
"""Generate deterministic C helpers from the ESP32 IMU sparse schedule IR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class ScheduleError(ValueError):
    """Raised when a schedule violates the generator contract."""


@dataclass(frozen=True)
class Term:
    coefficient: int
    left: str
    right: str


@dataclass(frozen=True)
class Output:
    name: str
    terms: tuple[Term, ...]


@dataclass(frozen=True)
class Function:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[Output, ...]


@dataclass(frozen=True)
class Schedule:
    schema_version: int
    name: str
    description: str
    functions: tuple[Function, ...]


def _require_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleError(f"{context} must be a non-empty string")
    if not (value[0].isalpha() or value[0] == "_"):
        raise ScheduleError(f"{context} is not a valid C identifier: {value!r}")
    if not all(character.isalnum() or character == "_" for character in value):
        raise ScheduleError(f"{context} is not a valid C identifier: {value!r}")
    return value


def _require_unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ScheduleError(f"{context} contains duplicate names: {values}")


def load_schedule(path: Path) -> Schedule:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScheduleError(f"unable to load schedule {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ScheduleError("schedule root must be an object")

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ScheduleError(f"unsupported schema_version: {schema_version!r}")

    name = _require_identifier(raw.get("name"), "schedule name")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ScheduleError("schedule description must be a string")

    raw_functions = raw.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ScheduleError("schedule functions must be a non-empty list")

    functions: list[Function] = []
    for function_index, raw_function in enumerate(raw_functions):
        context = f"functions[{function_index}]"
        if not isinstance(raw_function, dict):
            raise ScheduleError(f"{context} must be an object")

        function_name = _require_identifier(raw_function.get("name"), f"{context}.name")
        raw_inputs = raw_function.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise ScheduleError(f"{context}.inputs must be a non-empty list")
        inputs = tuple(
            _require_identifier(value, f"{context}.inputs[{index}]")
            for index, value in enumerate(raw_inputs)
        )
        _require_unique(inputs, f"{context}.inputs")
        input_set = set(inputs)

        raw_outputs = raw_function.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ScheduleError(f"{context}.outputs must be a non-empty list")

        outputs: list[Output] = []
        for output_index, raw_output in enumerate(raw_outputs):
            output_context = f"{context}.outputs[{output_index}]"
            if not isinstance(raw_output, dict):
                raise ScheduleError(f"{output_context} must be an object")
            output_name = _require_identifier(
                raw_output.get("name"), f"{output_context}.name"
            )
            raw_terms = raw_output.get("terms")
            if not isinstance(raw_terms, list) or not raw_terms:
                raise ScheduleError(f"{output_context}.terms must be non-empty")

            terms: list[Term] = []
            for term_index, raw_term in enumerate(raw_terms):
                term_context = f"{output_context}.terms[{term_index}]"
                if not isinstance(raw_term, dict):
                    raise ScheduleError(f"{term_context} must be an object")
                coefficient = raw_term.get("coefficient")
                if not isinstance(coefficient, int) or isinstance(coefficient, bool):
                    raise ScheduleError(f"{term_context}.coefficient must be an integer")
                if coefficient == 0:
                    raise ScheduleError(f"{term_context}.coefficient must be nonzero")
                factors = raw_term.get("factors")
                if not isinstance(factors, list) or len(factors) != 2:
                    raise ScheduleError(
                        f"{term_context}.factors must contain exactly two inputs"
                    )
                left = _require_identifier(factors[0], f"{term_context}.factors[0]")
                right = _require_identifier(factors[1], f"{term_context}.factors[1]")
                if left not in input_set or right not in input_set:
                    raise ScheduleError(
                        f"{term_context} references variables outside {inputs}"
                    )
                terms.append(Term(coefficient, left, right))

            outputs.append(Output(output_name, tuple(terms)))

        output_names = tuple(output.name for output in outputs)
        _require_unique(output_names, f"{context}.outputs")
        overlap = set(inputs) & set(output_names)
        if overlap:
            raise ScheduleError(
                f"{context} input/output names overlap and would alias: {sorted(overlap)}"
            )

        functions.append(Function(function_name, inputs, tuple(outputs)))

    _require_unique(tuple(function.name for function in functions), "functions")
    return Schedule(schema_version, name, description, tuple(functions))


def evaluate_function(
    function: Function,
    values: Mapping[str, int | float],
) -> dict[str, int | float]:
    missing = [name for name in function.inputs if name not in values]
    if missing:
        raise ScheduleError(f"missing input values for {function.name}: {missing}")

    result: dict[str, int | float] = {}
    for output in function.outputs:
        accumulator: int | float = 0
        for term in output.terms:
            accumulator += (
                term.coefficient * values[term.left] * values[term.right]
            )
        result[output.name] = accumulator
    return result


def _float_term(term: Term, first: bool) -> str:
    magnitude = abs(term.coefficient)
    product = f"{term.left} * {term.right}"
    if magnitude != 1:
        product = f"{magnitude}.0f * ({product})"

    if first:
        return f"-{product}" if term.coefficient < 0 else product
    return f" {'-' if term.coefficient < 0 else '+'} {product}"


def _q32_term(term: Term, first: bool) -> str:
    magnitude = abs(term.coefficient)
    product = f"(int64_t){term.left} * (int64_t){term.right}"
    if magnitude != 1:
        product = f"INT64_C({magnitude}) * ({product})"

    if first:
        return f"-{product}" if term.coefficient < 0 else product
    return f" {'-' if term.coefficient < 0 else '+'} {product}"


def emit_header(schedule: Schedule, source_path: str) -> str:
    guard = f"GEO_GENERATED_{schedule.name.upper()}_H"
    lines: list[str] = [
        "/*",
        " * Generated file. Do not edit by hand.",
        f" * Source: {source_path}",
        f" * Schedule: {schedule.name}",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
    ]

    for function in schedule.functions:
        float_parameters = [f"float {name}" for name in function.inputs]
        float_parameters.extend(f"float *{output.name}" for output in function.outputs)
        lines.append(f"static inline void geo_generated_float_{function.name}(")
        for index, parameter in enumerate(float_parameters):
            suffix = "," if index + 1 < len(float_parameters) else ""
            lines.append(f"    {parameter}{suffix}")
        lines.extend([")", "{"])
        for output in function.outputs:
            expression = "".join(
                _float_term(term, index == 0)
                for index, term in enumerate(output.terms)
            )
            lines.append(f"    *{output.name} = {expression};")
        lines.extend(["}", ""])

        q32_parameters = [f"int32_t {name}" for name in function.inputs]
        q32_parameters.extend(f"int64_t *{output.name}" for output in function.outputs)
        lines.append(f"static inline void geo_generated_q32_{function.name}(")
        for index, parameter in enumerate(q32_parameters):
            suffix = "," if index + 1 < len(q32_parameters) else ""
            lines.append(f"    {parameter}{suffix}")
        lines.extend([")", "{"])
        for output in function.outputs:
            expression = "".join(
                _q32_term(term, index == 0)
                for index, term in enumerate(output.terms)
            )
            lines.append(f"    *{output.name} = {expression};")
        lines.extend(["}", ""])

    lines.extend([f"#endif /* {guard} */", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse float and Q32 accumulator helpers for the IMU benchmark"
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in output differs from deterministic generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        schedule = load_schedule(args.schedule)
        generated = emit_header(schedule, args.schedule.as_posix())
    except ScheduleError as exc:
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
                f"ERROR: {args.output} is stale; regenerate it from {args.schedule}",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: {args.output} matches {args.schedule}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
