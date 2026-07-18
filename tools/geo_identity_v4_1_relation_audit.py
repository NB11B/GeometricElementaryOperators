#!/usr/bin/env python3
"""Classify V4.1 contraction preflight relations before corpus generation.

This audit recognizes grade-decomposition controls, declared-grade support,
and the fixed left/right contraction definitions used by the V4.1 preflight.
It does not treat exact equivalence as novelty evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def clean(label: str) -> str:
    return re.sub(r"\s+", "", label)


def pair(lhs: str, rhs: str) -> tuple[str, str]:
    return tuple(sorted((clean(lhs), clean(rhs))))


def classify(lhs: str, rhs: str) -> tuple[str, str, list[str]]:
    values = set(pair(lhs, rhs))
    joined = "=".join(sorted(values))

    if values == {"(B^v)", "(v^B)"}:
        return "graded_wedge_commutativity_2_1", "known-direct", [
            "graded_commutativity"
        ]

    if "lcon(" in joined or "rcon(" in joined:
        if values == {"(v*v)", "lcon(v,v)"}:
            return "left_contraction_vector_square", "definition-derived", [
                "left_contraction_definition",
                "declared_grade_1",
                "vector_square_grade_0",
            ]
        if values == {"(v*v)", "rcon(v,v)"}:
            return "right_contraction_vector_square", "definition-derived", [
                "right_contraction_definition",
                "declared_grade_1",
                "vector_square_grade_0",
            ]
        if values == {"grade1(v*B)", "lcon(v,B)"}:
            return "left_contraction_vector_bivector", "known-direct", [
                "left_contraction_definition_r1_s2"
            ]
        if values == {"grade1(B*v)", "rcon(B,v)"}:
            return "right_contraction_bivector_vector", "known-direct", [
                "right_contraction_definition_r2_s1"
            ]
        if values == {"grade2(T*v)", "lcon(v,T)"}:
            return "left_contraction_vector_trivector", "known-direct", [
                "left_contraction_definition_r1_s3",
                "grade2_product_symmetry",
            ]
        if values == {"grade2(T*v)", "rcon(T,v)"}:
            return "right_contraction_trivector_vector", "known-direct", [
                "right_contraction_definition_r3_s1"
            ]
        if values == {"lcon(B,B)", "rcon(B,B)"}:
            return "equal_grade_bivector_contraction_symmetry", "known-derived-1", [
                "left_right_contraction_definitions",
                "equal_grade_reversion_sign_cancellation",
            ]
        if values == {"lcon(B,T)", "rcon(T,B)"}:
            return "complementary_bivector_trivector_contraction", "known-derived-1", [
                "left_right_contraction_definitions",
                "reversion_anti_automorphism",
            ]
        if values == {"lcon(T,T)", "rcon(T,T)"}:
            return "equal_grade_trivector_contraction_symmetry", "known-derived-1", [
                "left_right_contraction_definitions",
                "equal_grade_reversion_sign_cancellation",
            ]
        return "contraction_definition_consequence", "definition-derived", [
            "declared_contraction_contract"
        ]

    if "grade" in joined and "^" in joined:
        return "wedge_highest_grade_projection", "known-direct", [
            "wedge_as_highest_grade_product_projection"
        ]

    if values == {"(T*T)", "grade0(T*T)"}:
        return "trivector_square_scalar_support", "support-derived", [
            "homogeneous_grade_3_square_support"
        ]
    if values == {"(v*v)", "grade0(v*v)"}:
        return "vector_square_scalar_support", "support-derived", [
            "homogeneous_grade_1_square_support"
        ]

    if values == {"grade1(B*T)", "grade1(T*B)"}:
        return "bivector_trivector_grade1_symmetry", "known-derived-1", [
            "homogeneous_product_grade_sign",
            "reversion_grade_sign",
        ]
    if values == {"grade2(T*v)", "grade2(v*T)"}:
        return "trivector_vector_grade2_symmetry", "known-derived-1", [
            "homogeneous_product_grade_sign",
            "reversion_grade_sign",
        ]

    return "unclassified_exact_relation", "unclassified", []


def family_key(lhs: str, rhs: str, family: str) -> str:
    payload = {"family": family, "pair": pair(lhs, rhs)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit(source: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, relation in enumerate(source.get("relations", []), 1):
        family, classification, derivation = classify(relation["lhs"], relation["rhs"])
        key = family_key(relation["lhs"], relation["rhs"], family)
        record = {
            "relation_id": f"v41r{index:03d}",
            "lhs": relation["lhs"],
            "rhs": relation["rhs"],
            "lhs_source": relation.get("lhs_source"),
            "rhs_source": relation.get("rhs_source"),
            "combined_cost": relation.get("combined_cost"),
            "family_key": key,
            "known_family": family,
            "classification": classification,
            "derivation": derivation,
        }
        records.append(record)
        groups[key].append(record)

    families: list[dict[str, Any]] = []
    for key, members in groups.items():
        members.sort(key=lambda item: (item["combined_cost"], item["lhs"], item["rhs"]))
        representative = members[0]
        families.append(
            {
                "family_key": key,
                "known_family": representative["known_family"],
                "classification": representative["classification"],
                "derivation": representative["derivation"],
                "representative": {
                    "lhs": representative["lhs"],
                    "rhs": representative["rhs"],
                },
                "member_count": len(members),
                "members": [member["relation_id"] for member in members],
                "minimum_cost": representative["combined_cost"],
            }
        )
    families.sort(
        key=lambda item: (
            item["classification"] == "unclassified",
            item["minimum_cost"],
            item["known_family"],
        )
    )
    counts = Counter(record["classification"] for record in records)
    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_1_relation_audit",
        "grammar": source.get("grammar"),
        "source_relation_count": len(records),
        "normalized_family_count": len(families),
        "classification_counts": dict(sorted(counts.items())),
        "unclassified_relation_count": counts.get("unclassified", 0),
        "unclassified_family_count": sum(
            1 for family in families if family["classification"] == "unclassified"
        ),
        "duality_generation_enabled": bool(
            source.get("operator_contract", {}).get("dual_relation_generation", False)
        ),
        "families": families,
        "relations": records,
    }


def markdown(report: dict[str, Any]) -> str:
    counts = report["classification_counts"]
    lines = [
        "# V4.1 contraction relation audit",
        "",
        f"- source relations: {report['source_relation_count']}",
        f"- normalized families: {report['normalized_family_count']}",
        f"- known-direct: {counts.get('known-direct', 0)}",
        f"- known-derived-1: {counts.get('known-derived-1', 0)}",
        f"- definition-derived: {counts.get('definition-derived', 0)}",
        f"- support-derived: {counts.get('support-derived', 0)}",
        f"- unclassified relations: {report['unclassified_relation_count']}",
        f"- unclassified families: {report['unclassified_family_count']}",
        f"- duality generation enabled: `{report['duality_generation_enabled']}`",
        "",
        "| relation | class | family | derivation |",
        "|---|---|---|---|",
    ]
    for family in report["families"]:
        relation = (
            f"`{family['representative']['lhs']} = "
            f"{family['representative']['rhs']}`"
        )
        derivation = "; ".join(family["derivation"]) or "none"
        lines.append(
            f"| {relation} | {family['classification']} | "
            f"`{family['known_family']}` | {derivation} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The contraction rows are exact consequences of the explicitly declared",
            "left/right contraction convention, grade support, and standard product",
            "decomposition rules. Duality remains outside the ranked relation corpus",
            "until fixed basis-blade constants are supported end to end.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = json.loads(args.preflight_json.read_text(encoding="utf-8"))
        report = audit(source)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "V4_1_RELATION_AUDIT: PASS "
        f"relations={report['source_relation_count']} "
        f"families={report['normalized_family_count']} "
        f"unclassified={report['unclassified_family_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
