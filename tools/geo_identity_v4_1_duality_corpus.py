#!/usr/bin/env python3
"""Build the exact V4.1 duality identity/control corpus.

The corpus uses the fixed pseudoscalar ``I=e1234`` in ``Cl(2,2)`` with
``I^2=1`` and the right-dual convention ``dual(A)=A*I``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import geo_identity_v4_1_exact as exact


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(value: object, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def fixed(blade: int, coefficient: int = 1) -> dict[str, Any]:
    return {"fixed_blade": {"blade": blade, "coefficient": coefficient}}


def gp(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "gp", "args": [copy.deepcopy(left), copy.deepcopy(right)]}


def neg(arg: dict[str, Any]) -> dict[str, Any]:
    return {"op": "neg", "arg": copy.deepcopy(arg)}


def grade(value: int, arg: dict[str, Any]) -> dict[str, Any]:
    return {"op": "grade", "grade": value, "arg": copy.deepcopy(arg)}


def dual(arg: dict[str, Any], pseudoscalar: dict[str, Any]) -> dict[str, Any]:
    return gp(arg, pseudoscalar)


def relation_definitions() -> list[dict[str, Any]]:
    I = fixed(15)
    definitions: list[dict[str, Any]] = [
        {
            "key": "pseudoscalar_square",
            "family": "pseudoscalar_inverse_contract",
            "lhs": gp(I, I),
            "rhs": {"scalar": 1},
        }
    ]
    grade_map = {"v": 3, "B": 2, "T": 1}
    for variable, target_grade in grade_map.items():
        seed = {"var": variable}
        first_dual = dual(seed, I)
        definitions.extend(
            [
                {
                    "key": f"dual_round_trip_{variable}",
                    "family": "right_dual_round_trip",
                    "lhs": dual(first_dual, I),
                    "rhs": seed,
                },
                {
                    "key": f"dual_grade_{variable}",
                    "family": "right_dual_grade_complement",
                    "lhs": first_dual,
                    "rhs": grade(target_grade, first_dual),
                },
            ]
        )
    definitions.extend(
        [
            {
                "key": "reverse_dual_v",
                "family": "dual_reversion_grade_sign",
                "lhs": {"op": "reverse", "arg": dual({"var": "v"}, I)},
                "rhs": neg(dual({"var": "v"}, I)),
            },
            {
                "key": "reverse_dual_B",
                "family": "dual_reversion_grade_sign",
                "lhs": {"op": "reverse", "arg": dual({"var": "B"}, I)},
                "rhs": neg(dual({"var": "B"}, I)),
            },
            {
                "key": "reverse_dual_T",
                "family": "dual_reversion_grade_sign",
                "lhs": {"op": "reverse", "arg": dual({"var": "T"}, I)},
                "rhs": dual({"var": "T"}, I),
            },
        ]
    )
    return definitions


def control_definitions() -> list[dict[str, Any]]:
    I = fixed(15)
    return [
        {
            "key": "pseudoscalar_square_sign_control",
            "family": "pseudoscalar_inverse_contract",
            "lhs": gp(I, I),
            "rhs": {"scalar": -1},
            "mutation": "negate_scalar_rhs",
        },
        {
            "key": "dual_round_trip_v_sign_control",
            "family": "right_dual_round_trip",
            "lhs": dual(dual({"var": "v"}, I), I),
            "rhs": neg({"var": "v"}),
            "mutation": "negate_rhs",
        },
        {
            "key": "dual_round_trip_B_sign_control",
            "family": "right_dual_round_trip",
            "lhs": dual(dual({"var": "B"}, I), I),
            "rhs": neg({"var": "B"}),
            "mutation": "negate_rhs",
        },
        {
            "key": "dual_v_wrong_grade_control",
            "family": "right_dual_grade_complement",
            "lhs": dual({"var": "v"}, I),
            "rhs": grade(1, dual({"var": "v"}, I)),
            "mutation": "wrong_grade_projection",
        },
    ]


def build_spec(
    grammar: dict[str, Any],
    definition: dict[str, Any],
    *,
    prime: int,
    expected: str,
    name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "description": (
            f"V4.1 exact duality corpus: {definition['key']} "
            f"under right dual A*I in Cl(2,2)"
        ),
        "expected": expected,
        "dimension": int(grammar["dimension"]),
        "signature": [int(value) for value in grammar["signature"]],
        "prime": prime,
        "coefficient_bound": int(grammar.get("coefficient_bound", 3)),
        "seed": int(grammar.get("seed", 3141592653)),
        "variables": copy.deepcopy(grammar["variables"]),
        "lhs": copy.deepcopy(definition["lhs"]),
        "rhs": copy.deepcopy(definition["rhs"]),
    }


def validate_definition(
    specification: dict[str, Any], expected: str, prechecks: int
) -> dict[str, Any]:
    exact.validate_spec(specification)
    polynomial = exact.extract_polynomial(specification)
    if expected == "identity" and not polynomial["zero"]:
        raise ValueError(f"identity is not exact: {specification['name']}")
    if expected == "counterexample" and polynomial["zero"]:
        raise ValueError(f"control is unexpectedly exact: {specification['name']}")
    witness_assignment = None
    witness = None
    if expected == "counterexample" and prechecks > 0:
        witness_assignment, witness = exact.precheck(specification, prechecks)
        if witness_assignment is None or witness is None:
            raise ValueError(
                f"control produced no witness in {prechecks} checks: {specification['name']}"
            )
    return {
        "zero": bool(polynomial["zero"]),
        "canonical_hash": polynomial["canonical_hash"],
        "total_terms": int(polynomial["total_terms"]),
        "maximum_degree": int(polynomial["maximum_degree"]),
        "witness_assignment": witness_assignment,
        "witness": None
        if witness is None
        else {"blade": witness.blade, "lhs": witness.lhs, "rhs": witness.rhs},
    }


def build_corpus(
    grammar_path: Path,
    output_root: Path,
    primes: list[int],
    prechecks: int,
) -> dict[str, Any]:
    grammar = json.loads(grammar_path.read_text(encoding="utf-8"))
    if int(grammar.get("dimension", 0)) != 4:
        raise ValueError("V4.1 duality corpus currently requires dimension 4")
    if list(grammar.get("signature", [])) != [1, 1, -1, -1]:
        raise ValueError("V4.1 duality corpus currently requires signature [1,1,-1,-1]")
    variable_names = {entry.get("name") for entry in grammar.get("variables", [])}
    if variable_names != {"v", "B", "T"}:
        raise ValueError("grammar variables must be exactly v, B, and T")
    if prechecks < 0:
        raise ValueError("prechecks must be non-negative")
    if not primes or any(
        prime < 3 or prime % 2 == 0 or not exact.is_prime(prime) for prime in primes
    ):
        raise ValueError("all primes must be valid odd primes")

    corpus_directory = output_root / "corpus"
    corpus_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    definitions = [
        (definition, "identity", None) for definition in relation_definitions()
    ] + [
        (definition, "counterexample", definition["mutation"])
        for definition in control_definitions()
    ]

    for definition_index, (definition, expected, mutation) in enumerate(definitions, 1):
        relation_id = f"d{definition_index:03d}_{stable_id(definition)}"
        variant = "identity" if expected == "identity" else "control"
        for prime in primes:
            name = f"v4_1_duality_{relation_id}_{variant}_p{prime}"
            specification = build_spec(
                grammar,
                definition,
                prime=prime,
                expected=expected,
                name=name,
            )
            evidence = validate_definition(specification, expected, prechecks)
            filename = f"{name}.json"
            write_json(corpus_directory / filename, specification)
            rows.append(
                {
                    "name": name,
                    "path": str(Path("corpus") / filename).replace("\\", "/"),
                    "relation_id": relation_id,
                    "definition_key": definition["key"],
                    "family": definition["family"],
                    "variant": variant,
                    "expected": expected,
                    "mutation": mutation,
                    "prime": prime,
                    "polynomial": evidence,
                }
            )

    identity_count = sum(row["expected"] == "identity" for row in rows)
    control_count = len(rows) - identity_count
    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_duality_corpus",
        "grammar": grammar["name"],
        "dimension": 4,
        "signature": [1, 1, -1, -1],
        "duality": {
            "side": "right",
            "pseudoscalar_blade": 15,
            "pseudoscalar_coefficient": 1,
            "pseudoscalar_square": 1,
        },
        "primes": primes,
        "identity_definition_count": len(relation_definitions()),
        "control_definition_count": len(control_definitions()),
        "identity_row_count": identity_count,
        "control_row_count": control_count,
        "statement_count": len(rows),
        "statements": rows,
    }
    write_json(output_root / "corpus-manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primes", type=int, nargs="+", default=[65521, 65519])
    parser.add_argument("--prechecks", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_corpus(
            args.grammar,
            args.output_root,
            list(args.primes),
            args.prechecks,
        )
    except (OSError, ValueError, json.JSONDecodeError, exact.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_DUALITY_CORPUS: PASS "
        f"identities={manifest['identity_row_count']} "
        f"controls={manifest['control_row_count']} "
        f"statements={manifest['statement_count']}"
    )
    print(f"manifest: {args.output_root / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
