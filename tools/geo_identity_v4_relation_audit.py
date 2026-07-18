#!/usr/bin/env python3
"""Normalize and classify V4 mixed-involution preflight relations.

The tool consumes the host-preflight JSON report, quotients side order and selected
operand-order presentations, assigns deterministic checked families, and emits a
family-level audit before near-miss controls or CUDA corpus generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def clean_label(label: str) -> str:
    return re.sub(r"\s+", "", label)


def canonical_pair(lhs: str, rhs: str) -> tuple[str, str]:
    left = clean_label(lhs)
    right = clean_label(rhs)
    return tuple(sorted((left, right)))


def classify(lhs: str, rhs: str) -> tuple[str, str, list[str]]:
    pair = set(canonical_pair(lhs, rhs))
    joined = "=".join(sorted(pair))

    if pair == {"v", "reverse(v)"}:
        return "vector_reversion_invariance", "support-derived", ["declared_grade_1"]
    if pair == {"B", "hat(B)"}:
        return "bivector_grade_involution", "support-derived", ["declared_grade_2"]
    if pair == {"reverse(B)", "bar(B)"}:
        return "bivector_conjugation_reversion", "known-derived-1", [
            "clifford_conjugation_definition",
            "bivector_grade_involution",
        ]
    if "(B^v)" in pair and "(v^B)" in pair:
        return "graded_wedge_commutativity_2_1", "known-direct", ["graded_commutativity"]
    if "hat(" in joined and "*" in joined and "reverse(" in joined:
        return "involution_product_interaction", "known-derived-1", [
            "grade_involution_automorphism",
            "reversion_anti_automorphism",
        ]
    if "bar(" in joined and "*" in joined:
        return "clifford_conjugation_anti_automorphism", "known-direct", [
            "clifford_conjugation_anti_automorphism"
        ]
    if "hat(" in joined and "[" in joined:
        return "grade_involution_commutator_interaction", "known-derived-1", [
            "grade_involution_automorphism",
            "commutator_definition",
        ]
    if "bar(" in joined and "[" in joined:
        return "clifford_conjugation_commutator_interaction", "known-derived-1", [
            "clifford_conjugation_anti_automorphism",
            "commutator_definition",
        ]
    if "reverse(" in joined and "^" in joined:
        return "reversion_wedge_interaction", "known-derived-1", [
            "reversion_grade_sign",
            "graded_commutativity",
        ]
    if "reverse(" in joined:
        return "reversion_support_or_product_symmetry", "known-derived-1", [
            "reversion_anti_automorphism",
            "declared_grade_support",
        ]
    if "hat(" in joined:
        return "grade_involution_support", "support-derived", ["declared_parity_support"]
    if "bar(" in joined:
        return "clifford_conjugation_support", "known-derived-1", [
            "clifford_conjugation_definition",
            "declared_grade_support",
        ]
    return "unclassified_exact_relation", "unclassified", []


def family_key(lhs: str, rhs: str, family: str) -> str:
    payload = {"family": family, "pair": canonical_pair(lhs, rhs)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit(report: dict[str, Any]) -> dict[str, Any]:
    records = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, relation in enumerate(report.get("relations", []), 1):
        family, classification, derivation = classify(relation["lhs"], relation["rhs"])
        key = family_key(relation["lhs"], relation["rhs"], family)
        record = {
            "relation_id": f"v4r{index:03d}",
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

    families = []
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
        "engine": "geometric_identity_v4_relation_audit",
        "grammar": report.get("grammar"),
        "source_relation_count": len(records),
        "normalized_family_count": len(families),
        "classification_counts": dict(sorted(counts.items())),
        "unclassified_relation_count": counts.get("unclassified", 0),
        "unclassified_family_count": sum(
            1 for family in families if family["classification"] == "unclassified"
        ),
        "families": families,
        "relations": records,
    }


def markdown(report: dict[str, Any]) -> str:
    counts = report["classification_counts"]
    lines = [
        "# V4 mixed-involution relation audit",
        "",
        f"- source relations: {report['source_relation_count']}",
        f"- normalized families: {report['normalized_family_count']}",
        f"- known-direct: {counts.get('known-direct', 0)}",
        f"- known-derived-1: {counts.get('known-derived-1', 0)}",
        f"- support-derived: {counts.get('support-derived', 0)}",
        f"- unclassified relations: {report['unclassified_relation_count']}",
        f"- unclassified families: {report['unclassified_family_count']}",
        "",
        "| relation | class | family | members | derivation |",
        "|---|---|---|---:|---|",
    ]
    for family in report["families"]:
        relation = (
            f"`{family['representative']['lhs']} = {family['representative']['rhs']}`"
        )
        derivation = "; ".join(family["derivation"]) or "none"
        lines.append(
            f"| {relation} | {family['classification']} | `{family['known_family']}` | "
            f"{family['member_count']} | {derivation} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This audit classifies exact host-preflight equivalences. It does not independently",
            "prove them and does not authorize a novelty claim. Unclassified families are only",
            "candidates for deeper symbolic, literature, and CUDA review.",
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
        "V4_RELATION_AUDIT: PASS "
        f"relations={report['source_relation_count']} "
        f"families={report['normalized_family_count']} "
        f"unclassified={report['unclassified_family_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
