#!/usr/bin/env python3
"""Quotient V4.2 relation presentations into structural families."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import geo_identity_v4_2_engine as engine
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_engine as engine

Expression = dict[str, Any]


def role_map(config: dict[str, Any]) -> dict[str, str]:
    return {str(item["name"]): str(item.get("role", item["name"])) for item in config["variables"]}


def normalize_expression(expression: Expression, roles: dict[str, str], dimension: int) -> Expression:
    if "var" in expression:
        return {"var_role": roles.get(str(expression["var"]), str(expression["var"]))}
    if "scalar" in expression:
        return {"scalar": int(expression["scalar"])}
    if "fixed_blade" in expression:
        payload = expression["fixed_blade"]
        blade = int(payload["blade"])
        coefficient = int(payload["coefficient"])
        if blade == (1 << dimension) - 1:
            return {"pseudoscalar": coefficient}
        return {"fixed_blade": {"blade": blade, "coefficient": coefficient}}
    op = str(expression["op"])
    if "arg" in expression:
        output: Expression = {"op": op, "arg": normalize_expression(expression["arg"], roles, dimension)}
        if op == "scale":
            output["value"] = int(expression["value"])
        if op == "grade":
            output["grade"] = int(expression["grade"])
        return output
    arguments = [normalize_expression(argument, roles, dimension) for argument in expression["args"]]
    if op == "add":
        arguments.sort(key=engine.canonical_json)
    return {"op": op, "args": arguments}


def negate(expression: Expression) -> Expression:
    return {"op": "neg", "arg": copy.deepcopy(expression)}


def canonical_relation(lhs: Expression, rhs: Expression, roles: dict[str, str], dimension: int) -> dict[str, Any]:
    left = normalize_expression(lhs, roles, dimension)
    right = normalize_expression(rhs, roles, dimension)
    presentations = [
        {"lhs": left, "rhs": right},
        {"lhs": right, "rhs": left},
        {"lhs": negate(left), "rhs": negate(right)},
        {"lhs": negate(right), "rhs": negate(left)},
    ]
    presentations.sort(key=engine.canonical_json)
    return presentations[0]


def normalized_family_name(name: str, roles: dict[str, str]) -> str:
    output = str(name)
    for variable, role in sorted(roles.items(), key=lambda item: -len(item[0])):
        output = re.sub(rf"(?<![A-Za-z0-9]){re.escape(variable)}(?![A-Za-z0-9])", role, output)
        output = output.replace(f"_{variable}_", f"_{role}_")
        if output.endswith(f"_{variable}"):
            output = output[: -len(variable)] + role
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        roles = role_map(config)
        dimension = int(config["dimension"])
        definitions = manifest.get("definitions")
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("manifest definitions must be non-empty")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        records: list[dict[str, Any]] = []
        for definition in definitions:
            presentation = canonical_relation(definition["lhs"], definition["rhs"], roles, dimension)
            square_class = int(definition["pseudoscalar_square"])
            family = normalized_family_name(str(definition["family"]), roles)
            payload = {
                "expected": definition["expected"],
                "square_class": square_class,
                "family": family,
                "presentation": presentation,
            }
            key = hashlib.sha256(engine.canonical_json(payload).encode()).hexdigest()
            record = {
                "definition_id": definition["definition_id"],
                "signature_name": definition["signature_name"],
                "pseudoscalar_square": square_class,
                "expected": definition["expected"],
                "source_family": definition["family"],
                "normalized_family": family,
                "classification": definition.get("classification"),
                "source": definition.get("source"),
                "sign": definition.get("sign"),
                "family_key": key,
                "canonical_presentation": presentation,
            }
            records.append(record)
            groups[key].append(record)
        families: list[dict[str, Any]] = []
        for key, members in groups.items():
            members.sort(key=lambda item: (item["signature_name"], item["definition_id"]))
            representative = members[0]
            families.append({
                "family_key": key,
                "expected": representative["expected"],
                "pseudoscalar_square": representative["pseudoscalar_square"],
                "normalized_family": representative["normalized_family"],
                "canonical_presentation": representative["canonical_presentation"],
                "member_count": len(members),
                "signature_names": sorted({member["signature_name"] for member in members}),
                "members": [member["definition_id"] for member in members],
            })
        families.sort(key=lambda item: (item["expected"], item["normalized_family"], item["pseudoscalar_square"], item["family_key"]))
        counts = Counter(record["expected"] for record in records)
        report = {
            "schema_version": 1,
            "engine": "geometric_identity_v4_2_relation_audit",
            "source_definition_count": len(records),
            "normalized_family_count": len(families),
            "identity_definition_count": counts["identity"],
            "control_definition_count": counts["counterexample"],
            "quotient_reduction": len(records) - len(families),
            "families": families,
            "relations": records,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# V4.2 structural relation quotient",
            "",
            f"- source definitions: {report['source_definition_count']}",
            f"- normalized families: {report['normalized_family_count']}",
            f"- quotient reduction: {report['quotient_reduction']}",
            f"- identity definitions: {report['identity_definition_count']}",
            f"- control definitions: {report['control_definition_count']}",
            "",
            "| family | expected | I^2 | signatures | members |",
            "|---|---|---:|---|---:|",
        ]
        for family in families:
            signatures = ", ".join(family["signature_names"])
            lines.append(f"| `{family['normalized_family']}` | {family['expected']} | {family['pseudoscalar_square']} | {signatures} | {family['member_count']} |")
        lines.extend([
            "",
            "## Quotient contract",
            "",
            "Relations are normalized under side exchange, simultaneous sign reversal, variable-to-role renaming, additive operand ordering, and pseudoscalar-square class. Left/right dual presentations remain distinct unless their normalized ASTs coincide.",
            "",
        ])
        args.markdown_out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"V4_2_RELATION_AUDIT: PASS definitions={report['source_definition_count']} families={report['normalized_family_count']} reduction={report['quotient_reduction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
