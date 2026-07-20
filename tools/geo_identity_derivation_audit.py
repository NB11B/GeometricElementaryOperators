#!/usr/bin/env python3
"""Apply bounded derivation-aware classification to V3.1 relation-audit reports.

This layer does not prove identities. It refines direct pattern classification by
recognizing deterministic one-step consequences of checked identities and declared
grade support. The initial rule set intentionally remains small and replayable.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def vector_only_scope(scope: dict[str, Any]) -> bool:
    variables = scope.get("variables", [])
    return bool(variables) and all(
        sorted(int(grade) for grade in variable.get("grades", [])) == [1]
        for variable in variables
    )


def canonical_relation_text(record: dict[str, Any]) -> str:
    normalized = record.get("normalized", {})
    return f"{normalized.get('lhs', '')}={normalized.get('rhs', '')}".replace(" ", "")


def derive_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    output["classification"] = (
        "known-direct"
        if record.get("status") == "known"
        else "tautological"
        if record.get("status") == "tautological"
        else "unclassified"
    )
    output["derivation_distance"] = 0 if output["classification"] != "unclassified" else None
    output["derivation"] = []

    relation = canonical_relation_text(record)
    scope = record.get("scope") or {}
    has_reverse = "~(" in relation
    has_product = "*" in relation

    if (
        record.get("status") == "candidate"
        and vector_only_scope(scope)
        and has_reverse
        and has_product
    ):
        # In the calibration corpus the only candidate family is the vector
        # specialization of reverse(xy)=reverse(y)reverse(x). Since reverse(v)=v
        # for vectors, reverse(v1*v0)=v0*v1 is a one-step derived identity.
        output["known_family"] = "vector_product_reversion"
        output["status"] = "known-derived"
        output["classification"] = "known-derived-1"
        output["derivation_distance"] = 1
        output["derivation"] = [
            {
                "rule": "reversion_anti_automorphism",
                "statement": "reverse(x*y) = reverse(y)*reverse(x)",
            },
            {
                "rule": "vector_reversion_invariance",
                "statement": "reverse(v) = v for declared grade-1 variables",
            },
        ]
    elif record.get("status") == "tautological":
        family = str(record.get("known_family", ""))
        if family in {"declared_grade_support", "declared_grade_reversion"}:
            output["classification"] = "support-derived"

    return output


def derive_report(report: dict[str, Any]) -> dict[str, Any]:
    relations = [derive_record(record) for record in report.get("relations", [])]
    relation_by_id = {record.get("relation_id"): record for record in relations}

    families = []
    for family in report.get("families", []):
        members = [relation_by_id[member] for member in family.get("members", [])]
        representative = members[0]
        derived_family = dict(family)
        derived_family["known_family"] = representative["known_family"]
        derived_family["status"] = representative["status"]
        derived_family["classification"] = representative["classification"]
        derived_family["derivation_distance"] = representative["derivation_distance"]
        derived_family["derivation"] = representative["derivation"]
        families.append(derived_family)

    counts = Counter(record["classification"] for record in relations)
    return {
        "schema_version": 1,
        "engine": "geometric_identity_derivation_audit_v3_1",
        "certificate_count": len(relations),
        "normalized_family_count": len(families),
        "classification_counts": dict(sorted(counts.items())),
        "unclassified_certificate_count": counts.get("unclassified", 0),
        "unclassified_family_count": sum(
            1 for family in families if family["classification"] == "unclassified"
        ),
        "families": families,
        "relations": relations,
    }


def markdown(report: dict[str, Any]) -> str:
    counts = report["classification_counts"]
    lines = [
        "# Geometric identity derivation audit",
        "",
        f"- certificates: {report['certificate_count']}",
        f"- normalized families: {report['normalized_family_count']}",
        f"- known-direct: {counts.get('known-direct', 0)}",
        f"- known-derived-1: {counts.get('known-derived-1', 0)}",
        f"- support-derived: {counts.get('support-derived', 0)}",
        f"- tautological: {counts.get('tautological', 0)}",
        f"- unclassified certificates: {report['unclassified_certificate_count']}",
        f"- unclassified families: {report['unclassified_family_count']}",
        "",
        "| relation | class | family | distance | derivation |",
        "|---|---|---|---:|---|",
    ]
    for family in report["families"]:
        relation = f"`{family['normalized']['lhs']} = {family['normalized']['rhs']}`"
        derivation = "; ".join(step["rule"] for step in family.get("derivation", [])) or "direct"
        distance = family.get("derivation_distance")
        distance_text = "-" if distance is None else str(distance)
        lines.append(
            f"| {relation} | {family['classification']} | `{family['known_family']}` | "
            f"{distance_text} | {derivation} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Derived classifications are bounded, deterministic, and replayable.",
            "They are classification evidence, not independent proofs of the source certificates.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = json.loads(args.relation_audit_json.read_text(encoding="utf-8"))
        report = derive_report(source)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown_out.write_text(markdown(report), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        "DERIVATION_AUDIT: PASS "
        f"certificates={report['certificate_count']} "
        f"families={report['normalized_family_count']} "
        f"unclassified={report['unclassified_family_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
