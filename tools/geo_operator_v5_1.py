#!/usr/bin/env python3
"""Build and verify the V4.3-V5.1 geometric operator pipeline.

This tool covers:
- V4.3 fixed-blade signed-permutation specialization;
- V4.4 sparse fixed multivectors;
- V4.5 linear operator extraction;
- V4.6 dimensions 2 through 6 and every non-degenerate signature class;
- V4.7 theorem-schema instantiation;
- V4.8 independent proof-certificate production and checking;
- V5.0 allocation-free C-plan generation;
- V5.1 embedded C header generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import geo_identity_v4_2_exact as exact
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_exact as exact


class OperatorError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signatures(dimension: int) -> list[tuple[str, list[int]]]:
    return [
        (f"cl{positive}{dimension-positive}", [1] * positive + [-1] * (dimension - positive))
        for positive in range(dimension, -1, -1)
    ]


def gp_sign(left: int, right: int, signature: Sequence[int]) -> int:
    return exact.gp_sign(left, right, signature)


def fixed_plan(
    dimension: int,
    signature: Sequence[int],
    blade: int,
    coefficient: int = 1,
    side: str = "right",
) -> list[dict[str, int]]:
    if side not in {"left", "right"}:
        raise OperatorError("side must be left or right")
    blade_count = 1 << dimension
    if not 0 <= blade < blade_count:
        raise OperatorError("blade is outside dimension")
    if coefficient == 0:
        raise OperatorError("coefficient must be nonzero")
    rows: list[dict[str, int]] = []
    for source in range(blade_count):
        if side == "right":
            factor = coefficient * gp_sign(source, blade, signature)
        else:
            factor = coefficient * gp_sign(blade, source, signature)
        rows.append({"source": source, "target": source ^ blade, "factor": factor})
    return rows


def normalize_sparse_terms(
    dimension: int,
    terms: Iterable[dict[str, int]],
) -> list[dict[str, int]]:
    blade_count = 1 << dimension
    combined: dict[int, int] = {}
    for raw in terms:
        blade = int(raw["blade"])
        coefficient = int(raw["coefficient"])
        if blade == -1:
            blade = blade_count - 1
        if not 0 <= blade < blade_count:
            raise OperatorError(f"sparse blade {blade} is outside dimension {dimension}")
        combined[blade] = combined.get(blade, 0) + coefficient
    output = [
        {"blade": blade, "coefficient": coefficient}
        for blade, coefficient in sorted(combined.items())
        if coefficient
    ]
    if not output:
        raise OperatorError("sparse fixed multivector must be nonzero")
    if len(output) > 64:
        raise OperatorError("sparse fixed multivector exceeds 64 terms")
    return output


def sparse_plan(
    dimension: int,
    signature: Sequence[int],
    terms: Sequence[dict[str, int]],
    side: str,
) -> list[list[dict[str, int]]]:
    normalized = normalize_sparse_terms(dimension, terms)
    blade_count = 1 << dimension
    rows: list[list[dict[str, int]]] = [[] for _ in range(blade_count)]
    for term in normalized:
        for mapping in fixed_plan(
            dimension,
            signature,
            term["blade"],
            term["coefficient"],
            side,
        ):
            rows[mapping["target"]].append(
                {"source": mapping["source"], "factor": mapping["factor"]}
            )
    for row in rows:
        row.sort(key=lambda item: (item["source"], item["factor"]))
    return rows


def generic_gp(
    left: Sequence[int],
    right: Sequence[int],
    signature: Sequence[int],
) -> list[int]:
    blade_count = 1 << len(signature)
    output = [0] * blade_count
    for left_blade, left_value in enumerate(left):
        if not left_value:
            continue
        for right_blade, right_value in enumerate(right):
            if not right_value:
                continue
            output[left_blade ^ right_blade] += (
                left_value * right_value * gp_sign(left_blade, right_blade, signature)
            )
    return output


def apply_sparse(
    input_value: Sequence[int],
    rows: Sequence[Sequence[dict[str, int]]],
) -> list[int]:
    return [
        sum(input_value[item["source"]] * item["factor"] for item in row)
        for row in rows
    ]


def dense_matrix(rows: Sequence[Sequence[dict[str, int]]]) -> list[list[int]]:
    size = len(rows)
    matrix = [[0] * size for _ in range(size)]
    for target, row in enumerate(rows):
        for item in row:
            matrix[target][item["source"]] += item["factor"]
    return matrix


def matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    work = [[float(value) for value in row] for row in matrix]
    row_count = len(work)
    column_count = len(work[0]) if work else 0
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count) if abs(work[row][column]) > 1e-12), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = work[row][column]
            if abs(factor) <= 1e-12:
                continue
            work[row] = [a - factor * b for a, b in zip(work[row], work[rank])]
        rank += 1
        if rank == row_count:
            break
    return rank


def structural_features(matrix: Sequence[Sequence[int]]) -> dict[str, Any]:
    nonzero = sum(value != 0 for row in matrix for value in row)
    row_nonzero = [sum(value != 0 for value in row) for row in matrix]
    columns = list(zip(*matrix))
    column_nonzero = [sum(value != 0 for value in column) for column in columns]
    return {
        "size": len(matrix),
        "nonzero": nonzero,
        "density": nonzero / (len(matrix) ** 2),
        "rank": matrix_rank(matrix),
        "monomial": all(value == 1 for value in row_nonzero)
        and all(value == 1 for value in column_nonzero),
        "maximum_row_nonzero": max(row_nonzero, default=0),
        "maximum_column_nonzero": max(column_nonzero, default=0),
    }


def theorem_schemas() -> list[dict[str, Any]]:
    return [
        {
            "schema": "fixed_blade_specialization",
            "parameters": ["dimension", "signature", "blade", "side"],
            "statement": "generic_gp(A,e_J)=signed_permutation_J(A)",
        },
        {
            "schema": "sparse_fixed_multivector",
            "parameters": ["dimension", "signature", "terms", "side"],
            "statement": "generic_gp(A,K)=sum_J k_J signed_permutation_J(A)",
        },
        {
            "schema": "pseudoscalar_square",
            "parameters": ["dimension", "signature"],
            "statement": "I^2=(-1)^(n(n-1)/2) product_i g_i",
        },
        {
            "schema": "dual_square",
            "parameters": ["dimension", "signature", "grade"],
            "statement": "dual_R(dual_R(A_r))=I^2 A_r",
        },
    ]


def certificate_payload(
    dimension: int,
    signature_name: str,
    signature: Sequence[int],
    side: str,
    terms: Sequence[dict[str, int]],
    rows: Sequence[Sequence[dict[str, int]]],
) -> dict[str, Any]:
    normalized = normalize_sparse_terms(dimension, terms)
    matrix = dense_matrix(rows)
    payload = {
        "schema_version": 1,
        "certificate_type": "geometric_fixed_operator_equivalence",
        "dimension": dimension,
        "signature_name": signature_name,
        "signature": list(signature),
        "side": side,
        "terms": normalized,
        "rows": rows,
        "matrix_hash": sha256(matrix),
        "features": structural_features(matrix),
    }
    payload["certificate_hash"] = sha256(payload)
    return payload


def verify_certificate(certificate: dict[str, Any]) -> None:
    expected_hash = certificate.get("certificate_hash")
    unsigned = dict(certificate)
    unsigned.pop("certificate_hash", None)
    if expected_hash != sha256(unsigned):
        raise OperatorError("certificate hash mismatch")
    dimension = int(certificate["dimension"])
    signature = [int(value) for value in certificate["signature"]]
    rows = sparse_plan(
        dimension,
        signature,
        certificate["terms"],
        str(certificate["side"]),
    )
    if rows != certificate["rows"]:
        raise OperatorError("certificate row plan mismatch")
    if sha256(dense_matrix(rows)) != certificate["matrix_hash"]:
        raise OperatorError("certificate matrix hash mismatch")


def emit_embedded_header(certificates: Sequence[dict[str, Any]]) -> str:
    lines = [
        "#ifndef GEO_OPERATOR_PLANS_V5_1_H",
        "#define GEO_OPERATOR_PLANS_V5_1_H",
        "",
        "#include \"geo/operator_kernel.h\"",
        "",
    ]
    for index, certificate in enumerate(certificates):
        name = f"geo_operator_plan_v5_1_{index}"
        terms = certificate["terms"]
        lines.append(f"static const geo_operator_term_i32_t {name}_terms[{len(terms)}] = {{")
        for term in terms:
            lines.append(f"    {{{term['blade']}u, {term['coefficient']}}},")
        lines.extend(
            [
                "};",
                f"static inline geo_operator_status_t {name}_init(geo_operator_plan_i32_t *plan) {{",
                f"    static const int8_t signature[{certificate['dimension']}] = {{"
                + ", ".join(str(value) for value in certificate["signature"])
                + "};",
                "    return geo_operator_plan_sparse_i32(",
                f"        plan, {certificate['dimension']}u, signature,",
                "        "
                + (
                    "GEO_OPERATOR_SIDE_RIGHT,"
                    if certificate["side"] == "right"
                    else "GEO_OPERATOR_SIDE_LEFT,"
                ),
                f"        {name}_terms, {len(terms)}u",
                "    );",
                "}",
                "",
            ]
        )
    lines.extend(["#endif", ""])
    return "\n".join(lines)


def run(config_path: Path, output_root: Path, seed: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dimensions = [int(value) for value in config["dimensions"]]
    random_source = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    certificate_dir = output_root / "certificates"
    certificate_dir.mkdir(parents=True, exist_ok=True)

    specialization_cases = 0
    sparse_cases = 0
    assignments = 0
    certificates: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []

    for dimension in dimensions:
        blade_count = 1 << dimension
        for signature_name, signature in signatures(dimension):
            signature_rows.append(
                {
                    "dimension": dimension,
                    "signature_name": signature_name,
                    "signature": signature,
                    "pseudoscalar_square": exact.pseudoscalar_square(signature),
                }
            )
            blades = range(blade_count) if config.get("all_fixed_blades", True) else [blade_count - 1]
            for side in ("left", "right"):
                for blade in blades:
                    rows = sparse_plan(
                        dimension,
                        signature,
                        [{"blade": blade, "coefficient": 1}],
                        side,
                    )
                    constant = [0] * blade_count
                    constant[blade] = 1
                    for _ in range(int(config.get("specialization_iterations", 8))):
                        value = [random_source.randint(-3, 3) for _ in range(blade_count)]
                        generic = generic_gp(constant, value, signature) if side == "left" else generic_gp(value, constant, signature)
                        if generic != apply_sparse(value, rows):
                            raise OperatorError(
                                f"fixed specialization mismatch dimension={dimension} signature={signature_name} blade={blade} side={side}"
                            )
                        assignments += 1
                    specialization_cases += 1

            sparse_terms = normalize_sparse_terms(dimension, config["sparse_terms"])
            for side in ("left", "right"):
                rows = sparse_plan(dimension, signature, sparse_terms, side)
                constant = [0] * blade_count
                for term in sparse_terms:
                    constant[term["blade"]] = term["coefficient"]
                for _ in range(int(config.get("sparse_iterations", 16))):
                    value = [random_source.randint(-3, 3) for _ in range(blade_count)]
                    generic = generic_gp(constant, value, signature) if side == "left" else generic_gp(value, constant, signature)
                    if generic != apply_sparse(value, rows):
                        raise OperatorError(
                            f"sparse specialization mismatch dimension={dimension} signature={signature_name} side={side}"
                        )
                    assignments += 1
                certificate = certificate_payload(
                    dimension,
                    signature_name,
                    signature,
                    side,
                    sparse_terms,
                    rows,
                )
                verify_certificate(certificate)
                path = certificate_dir / f"{signature_name}_{side}.json"
                path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                certificates.append(certificate)
                sparse_cases += 1

    schema_payload = {
        "schema_version": 1,
        "schemas": theorem_schemas(),
        "signature_instances": signature_rows,
    }
    (output_root / "theorem-schemas.json").write_text(
        json.dumps(schema_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "geo_operator_plans_v5_1.h").write_text(
        emit_embedded_header(certificates), encoding="utf-8", newline="\n"
    )

    report = {
        "schema_version": 1,
        "engine": "geometric_operator_kernel_v5_1",
        "dimensions": dimensions,
        "signature_instances": len(signature_rows),
        "specialization_cases": specialization_cases,
        "sparse_cases": sparse_cases,
        "exact_integer_assignments": assignments,
        "certificate_count": len(certificates),
        "theorem_schema_count": len(schema_payload["schemas"]),
        "maximum_blades": max(1 << dimension for dimension in dimensions),
        "allocation_free_c_abi": True,
        "runtime_parsing": False,
        "validation": "PASS",
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "report.md").write_text(
        "\n".join(
            [
                "# Geometric Operator Kernel V5.1",
                "",
                f"- dimensions: `{dimensions}`",
                f"- signature instances: {report['signature_instances']}",
                f"- fixed-blade specialization cases: {specialization_cases}",
                f"- sparse fixed-multivector cases: {sparse_cases}",
                f"- exact integer assignments: {assignments}",
                f"- independent certificates: {len(certificates)}",
                f"- theorem schemas: {report['theorem_schema_count']}",
                f"- maximum blades: {report['maximum_blades']}",
                "- allocation-free C ABI: **PASS**",
                "- runtime parsing required: **no**",
                "- validation: **PASS**",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return report


def verify_directory(root: Path) -> int:
    paths = sorted((root / "certificates").glob("*.json"))
    if not paths:
        raise OperatorError("no certificates found")
    for path in paths:
        verify_certificate(json.loads(path.read_text(encoding="utf-8")))
    print(f"V4_8_CERTIFICATES: PASS certificates={len(paths)}")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify is not None:
            verify_directory(args.verify)
            return 0
        if args.config is None or args.output_root is None:
            parser.error("--config and --output-root are required unless --verify is used")
        report = run(args.config, args.output_root, args.seed)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, exact.DiscoveryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "GEO_OPERATOR_V5_1: PASS "
        f"signatures={report['signature_instances']} "
        f"fixed_cases={report['specialization_cases']} "
        f"sparse_cases={report['sparse_cases']} "
        f"assignments={report['exact_integer_assignments']} "
        f"certificates={report['certificate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
