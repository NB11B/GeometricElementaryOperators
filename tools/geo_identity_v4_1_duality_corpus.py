#!/usr/bin/env python3
"""Generate the first end-to-end V4.1 fixed-blade duality corpus."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import geo_identity_v4_1_ir as ir


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_id(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def gp(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "gp", "args": [copy.deepcopy(left), copy.deepcopy(right)]}


def reverse(value: dict[str, Any]) -> dict[str, Any]:
    return {"op": "reverse", "arg": copy.deepcopy(value)}


def build(output_root: Path, primes: list[int]) -> dict[str, Any]:
    dimension = 4
    signature = [1, 1, -1, -1]
    variables = [
        {"name": "v", "grades": [1]},
        {"name": "B", "grades": [2]},
        {"name": "T", "grades": [3]},
    ]
    I = {"fixed_blade": {"blade": 15, "coefficient": 1}}
    seeds = {name: {"var": name} for name in ("v", "B", "T")}
    relations: list[dict[str, Any]] = [
        {"family": "pseudoscalar_square", "lhs": gp(I, I), "rhs": {"scalar": 1}},
    ]
    for name, value in seeds.items():
        dual = gp(value, I)
        relations.append(
            {
                "family": f"dual_round_trip_{name}",
                "lhs": gp(dual, I),
                "rhs": value,
            }
        )
    dual_T = gp(seeds["T"], I)
    relations.append(
        {
            "family": "trivector_dual_reversion_invariance",
            "lhs": reverse(dual_T),
            "rhs": dual_T,
        }
    )

    corpus = output_root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    controls = 0
    for index, relation in enumerate(relations, 1):
        relation_id = f"d{index:03d}_{stable_id(relation)}"
        variants = [
            ("identity", "identity", relation["rhs"], None),
        ]
        if controls < 3:
            controls += 1
            variants.append(
                (
                    f"control_{controls:02d}",
                    "counterexample",
                    {"op": "neg", "arg": copy.deepcopy(relation["rhs"])},
                    "negate_rhs",
                )
            )
        for variant, expected, rhs, mutation in variants:
            for prime in primes:
                name = f"v4_1_duality_{relation_id}_{variant}_p{prime}"
                spec = {
                    "schema_version": 1,
                    "name": name,
                    "description": f"V4.1 fixed-blade duality family {relation['family']}",
                    "expected": expected,
                    "dimension": dimension,
                    "signature": signature,
                    "prime": prime,
                    "coefficient_bound": 3,
                    "seed": 2718281828,
                    "variables": variables,
                    "lhs": copy.deepcopy(relation["lhs"]),
                    "rhs": copy.deepcopy(rhs),
                }
                ir.validate_spec(spec)
                polynomial = ir.extract_polynomial(spec)
                if expected == "identity" and not polynomial["zero"]:
                    raise ValueError(f"identity has nonzero polynomial: {name}")
                if expected == "counterexample" and polynomial["zero"]:
                    raise ValueError(f"control has zero polynomial: {name}")
                filename = f"{name}.json"
                write_json(corpus / filename, spec)
                rows.append(
                    {
                        "name": name,
                        "path": str(Path("corpus") / filename).replace("\\", "/"),
                        "relation_id": relation_id,
                        "family": relation["family"],
                        "variant": variant,
                        "expected": expected,
                        "prime": prime,
                        "mutation": mutation,
                    }
                )
    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_duality_corpus",
        "dimension": dimension,
        "signature": signature,
        "primes": primes,
        "relation_count": len(relations),
        "control_count": controls,
        "statement_count": len(rows),
        "statements": rows,
    }
    write_json(output_root / "corpus-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primes", type=int, nargs="+", default=[65521, 65519])
    args = parser.parse_args()
    try:
        manifest = build(args.output_root, list(args.primes))
    except (OSError, ValueError, ir.IdentityError, ir.DiscoveryError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_DUALITY_CORPUS: PASS "
        f"relations={manifest['relation_count']} controls={manifest['control_count']} "
        f"statements={manifest['statement_count']}"
    )
    print(f"manifest: {args.output_root / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
