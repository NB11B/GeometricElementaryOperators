#!/usr/bin/env python3
"""Generate the V4.2 signature-general duality and contraction corpus."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import geo_identity_v4_2_engine as engine
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_engine as engine

Expression = dict[str, Any]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_id(value: object) -> str:
    return hashlib.sha256(engine.canonical_json(value).encode()).hexdigest()[:12]


def support_projection(argument: Expression, grades: list[int]) -> Expression:
    return engine.add_many([engine.grade(grade, argument) for grade in grades])


def relation_variables(relation: dict[str, Any], variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = engine.expression_variables(relation["lhs"]) | engine.expression_variables(relation["rhs"])
    selected = [{"name": item["name"], "grades": list(item["grades"])} for item in variables if item["name"] in names]
    if not selected:
        first = variables[0]
        selected = [{"name": first["name"], "grades": list(first["grades"])}]
    return selected


def base_spec(config: dict[str, Any], signature: list[int], relation: dict[str, Any], prime: int, name: str, expected: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "description": f"V4.2 {relation['family']} over signature {signature}",
        "expected": expected,
        "dimension": int(config["dimension"]),
        "signature": list(signature),
        "prime": int(prime),
        "coefficient_bound": int(config.get("coefficient_bound", 3)),
        "seed": int(config.get("seed", 1414213562)),
        "variables": relation_variables(relation, config["variables"]),
        "lhs": copy.deepcopy(relation["lhs"]),
        "rhs": copy.deepcopy(relation["rhs"]),
    }


def exact_relation(config: dict[str, Any], signature: list[int], relation: dict[str, Any]) -> bool:
    spec = base_spec(config, signature, relation, int(config["primes"][0]), "preflight", "identity")
    return bool(engine.extract_polynomial(spec, int(config["corpus"].get("term_limit", 500000)))["zero"])


def add_required(relations: list[dict[str, Any]], config: dict[str, Any], signature: list[int], family: str, lhs: Expression, rhs: Expression, source: str) -> None:
    relation = {"family": family, "lhs": lhs, "rhs": rhs, "source": source, "classification": "required"}
    if not exact_relation(config, signature, relation):
        raise ValueError(f"required relation is not exact: {family} signature={signature}")
    relations.append(relation)


def discover_signed(relations: list[dict[str, Any]], config: dict[str, Any], signature: list[int], family: str, lhs: Expression, candidate: Expression, source: str) -> bool:
    hits: list[tuple[int, dict[str, Any]]] = []
    for sign in (1, -1):
        relation = {
            "family": family,
            "lhs": copy.deepcopy(lhs),
            "rhs": engine.scale(sign, candidate),
            "source": source,
            "classification": "discovered-sign",
            "sign": sign,
        }
        if exact_relation(config, signature, relation):
            hits.append((sign, relation))
    if len(hits) > 1:
        raise ValueError(f"ambiguous sign discovery for {family}")
    if not hits:
        return False
    relations.append(hits[0][1])
    return True


def build_relations(config: dict[str, Any], signature: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dimension = int(config["dimension"])
    square = engine.pseudoscalar_square(signature)
    I = engine.pseudoscalar(signature)
    I_inv = engine.pseudoscalar(signature, True)
    variables = {item["name"]: engine.variable(item["name"]) for item in config["variables"]}
    grade_map = {item["name"]: list(item["grades"]) for item in config["variables"]}
    relations: list[dict[str, Any]] = []

    add_required(relations, config, signature, "pseudoscalar_square", engine.gp(I, I), engine.scalar(square), "metric-contract")
    add_required(relations, config, signature, "pseudoscalar_inverse_right", engine.gp(I_inv, I), engine.scalar(1), "metric-contract")
    add_required(relations, config, signature, "pseudoscalar_inverse_left", engine.gp(I, I_inv), engine.scalar(1), "metric-contract")

    for name, value in variables.items():
        add_required(relations, config, signature, f"right_dual_round_trip_{name}", engine.right_undual(engine.right_dual(value, signature), signature), value, "duality-definition")
        add_required(relations, config, signature, f"right_dual_square_{name}", engine.right_dual(engine.right_dual(value, signature), signature), engine.scale(square, value), "pseudoscalar-square")

    homogeneous = {"v": 1, "B": 2, "T": 3}
    for name, source_grade in homogeneous.items():
        value = variables[name]
        right = engine.right_dual(value, signature)
        target_grade = dimension - source_grade
        add_required(relations, config, signature, f"right_dual_grade_{name}_{target_grade}", right, engine.grade(target_grade, right), "homogeneous-grade-complement")
        commute_sign = -1 if ((source_grade * (dimension - source_grade)) & 1) else 1
        add_required(relations, config, signature, f"left_right_dual_sign_{name}", engine.left_dual(value, signature), engine.scale(commute_sign, right), "pseudoscalar-grade-commutation")

    for name in ("E", "O"):
        value = variables[name]
        dual = engine.right_dual(value, signature)
        add_required(relations, config, signature, f"mixed_dual_support_{name}", dual, support_projection(dual, grade_map[name]), "mixed-grade-complement-support")

    pairs = [("v", "B"), ("v", "T"), ("B", "T")]
    for left_name, right_name in pairs:
        left = variables[left_name]
        right = variables[right_name]
        contraction = engine.left_contraction(left, right, dimension)
        add_required(
            relations,
            config,
            signature,
            f"contraction_dual_round_trip_{left_name}_{right_name}",
            engine.right_undual(engine.right_dual(contraction, signature), signature),
            contraction,
            "contraction-plus-duality-round-trip",
        )

    max_discovered = int(config["corpus"].get("max_exact_contraction_duality_relations_per_signature", 8))
    discovered = 0
    candidate_pairs = [("v", "B"), ("v", "T"), ("B", "T"), ("E", "O"), ("O", "M")]
    for left_name, right_name in candidate_pairs:
        if discovered >= max_discovered:
            break
        left = variables[left_name]
        right = variables[right_name]
        candidates = [
            (
                f"dual_wedge_lcon_{left_name}_{right_name}",
                engine.right_dual(engine.wedge(left, right), signature),
                engine.left_contraction(left, engine.right_dual(right, signature), dimension),
            ),
            (
                f"dual_wedge_rcon_{left_name}_{right_name}",
                engine.right_dual(engine.wedge(left, right), signature),
                engine.right_contraction(engine.right_dual(left, signature), right, dimension),
            ),
            (
                f"dual_lcon_wedge_{left_name}_{right_name}",
                engine.right_dual(engine.left_contraction(left, right, dimension), signature),
                engine.wedge(left, engine.right_dual(right, signature)),
            ),
        ]
        for family, lhs, rhs in candidates:
            if discovered >= max_discovered:
                break
            if discover_signed(relations, config, signature, family, lhs, rhs, "exact-sign-and-order-search"):
                discovered += 1

    controls = [
        {"family": "control_wrong_pseudoscalar_square", "lhs": engine.gp(I, I), "rhs": engine.scalar(-square), "source": "wrong-sign", "classification": "control"},
        {"family": "control_wrong_vector_dual_square", "lhs": engine.right_dual(engine.right_dual(variables["v"], signature), signature), "rhs": engine.scale(-square, variables["v"]), "source": "wrong-sign", "classification": "control"},
        {"family": "control_wrong_vector_dual_grade", "lhs": engine.right_dual(variables["v"], signature), "rhs": engine.grade(2, engine.right_dual(variables["v"], signature)), "source": "wrong-grade", "classification": "control"},
        {"family": "control_zero_vector_bivector_contraction", "lhs": engine.left_contraction(variables["v"], variables["B"], dimension), "rhs": engine.scalar(0), "source": "erased-contraction", "classification": "control"},
    ]
    requested_controls = int(config["corpus"].get("controls_per_signature", 4))
    controls = controls[:requested_controls]
    for control in controls:
        if exact_relation(config, signature, control):
            raise ValueError(f"control unexpectedly classifies as an identity: {control['family']}")
    return relations, controls


def build(config_path: Path, output_root: Path, max_relations_per_signature: int = 0) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus = output_root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []

    for signature_row in config["signatures"]:
        signature_name = signature_row["name"]
        signature = [int(value) for value in signature_row["metric"]]
        relations, controls = build_relations(config, signature)
        if max_relations_per_signature > 0:
            relations = relations[:max_relations_per_signature]
        for expected, collection in (("identity", relations), ("counterexample", controls)):
            for relation_index, relation in enumerate(collection, 1):
                definition = {
                    "signature_name": signature_name,
                    "signature": signature,
                    "pseudoscalar_square": engine.pseudoscalar_square(signature),
                    "expected": expected,
                    **relation,
                }
                definition_id = f"{signature_name}_{'i' if expected == 'identity' else 'c'}{relation_index:03d}_{stable_id(definition)}"
                definition["definition_id"] = definition_id
                definitions.append(definition)
                for prime in config["primes"]:
                    name = f"v4_2_{definition_id}_p{prime}"
                    spec = base_spec(config, signature, relation, int(prime), name, expected)
                    engine.validate_spec(spec)
                    polynomial = engine.extract_polynomial(spec, int(config["corpus"].get("term_limit", 500000)))
                    if expected == "identity" and not polynomial["zero"]:
                        raise ValueError(f"identity has nonzero polynomial: {name}")
                    if expected == "counterexample" and polynomial["zero"]:
                        raise ValueError(f"control has zero polynomial: {name}")
                    filename = f"{name}.json"
                    write_json(corpus / filename, spec)
                    rows.append({
                        "name": name,
                        "path": str(Path("corpus") / filename).replace("\\", "/"),
                        "definition_id": definition_id,
                        "signature_name": signature_name,
                        "signature": signature,
                        "pseudoscalar_square": definition["pseudoscalar_square"],
                        "family": relation["family"],
                        "classification": relation.get("classification"),
                        "source": relation.get("source"),
                        "sign": relation.get("sign"),
                        "expected": expected,
                        "prime": int(prime),
                        "polynomial_hash": polynomial["hash"],
                    })

    manifest = {
        "schema_version": 1,
        "engine": "geometric_identity_v4_2_native_signature_matrix",
        "config": config_path.as_posix(),
        "dimension": config["dimension"],
        "primes": config["primes"],
        "signature_count": len(config["signatures"]),
        "definition_count": len(definitions),
        "identity_definition_count": sum(item["expected"] == "identity" for item in definitions),
        "control_definition_count": sum(item["expected"] == "counterexample" for item in definitions),
        "statement_count": len(rows),
        "definitions": definitions,
        "statements": rows,
    }
    write_json(output_root / "corpus-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-relations-per-signature", type=int, default=0)
    args = parser.parse_args()
    try:
        report = build(args.config, args.output_root, args.max_relations_per_signature)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, engine.IdentityError, engine.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_2_CORPUS: PASS "
        f"signatures={report['signature_count']} "
        f"identity_definitions={report['identity_definition_count']} "
        f"controls={report['control_definition_count']} "
        f"statements={report['statement_count']}"
    )
    print(f"manifest: {args.output_root / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
