#!/usr/bin/env python3
"""Normalize, quotient, classify, and rank exact geometric identity certificates.

The tool consumes V3 certificate JSON files. It does not re-prove relations; each
certificate is assumed to have already passed the exact zero-polynomial gate.
It removes low-value syntactic variation, canonicalizes variable names, groups
relations into mathematical families, compares them with a checked identity
library, and emits a novelty-oriented report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

Expr = dict[str, Any]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def expression_cost(expr: Expr) -> int:
    if "var" in expr or "scalar" in expr:
        return 1
    if "arg" in expr:
        return 1 + expression_cost(expr["arg"])
    return 1 + sum(expression_cost(arg) for arg in expr.get("args", []))


def expression_variables(expr: Expr) -> set[str]:
    if "var" in expr:
        return {str(expr["var"])}
    output: set[str] = set()
    if "arg" in expr:
        output.update(expression_variables(expr["arg"]))
    for argument in expr.get("args", []):
        output.update(expression_variables(argument))
    return output


def expression_ops(expr: Expr) -> Counter[str]:
    output: Counter[str] = Counter()
    if "op" in expr:
        output[str(expr["op"])] += 1
    if "arg" in expr:
        output.update(expression_ops(expr["arg"]))
    for argument in expr.get("args", []):
        output.update(expression_ops(argument))
    return output


def scalar(value: int) -> Expr:
    return {"scalar": int(value)}


def is_zero(expr: Expr) -> bool:
    return expr == {"scalar": 0}


def reverse_sign_for_grade(grade: int) -> int:
    return -1 if ((grade * (grade - 1) // 2) & 1) else 1


def rename_variables(
    expr: Expr,
    mapping: dict[str, str] | None = None,
) -> tuple[Expr, dict[str, str]]:
    if mapping is None:
        mapping = {}
    if "var" in expr:
        name = str(expr["var"])
        if name not in mapping:
            mapping[name] = f"v{len(mapping)}"
        return {"var": mapping[name]}, mapping
    if "scalar" in expr:
        return scalar(int(expr["scalar"])), mapping
    output: Expr = {"op": expr["op"]}
    if "value" in expr:
        output["value"] = int(expr["value"])
    if "grade" in expr:
        output["grade"] = int(expr["grade"])
    if "arg" in expr:
        output["arg"], mapping = rename_variables(expr["arg"], mapping)
    if "args" in expr:
        arguments = []
        for argument in expr["args"]:
            renamed, mapping = rename_variables(argument, mapping)
            arguments.append(renamed)
        output["args"] = arguments
    return output, mapping


def simplify(expr: Expr, grade_support: dict[str, tuple[int, ...]]) -> Expr:
    if "var" in expr:
        return {"var": str(expr["var"])}
    if "scalar" in expr:
        return scalar(int(expr["scalar"]))

    op = str(expr["op"])
    if op in {"neg", "reverse", "grade", "scale"}:
        argument = simplify(expr["arg"], grade_support)
        if op == "neg":
            if "scalar" in argument:
                return scalar(-int(argument["scalar"]))
            if argument.get("op") == "neg":
                return simplify(argument["arg"], grade_support)
            if argument.get("op") == "scale":
                return simplify(
                    {
                        "op": "scale",
                        "value": -int(argument["value"]),
                        "arg": argument["arg"],
                    },
                    grade_support,
                )
            return {"op": "neg", "arg": argument}
        if op == "scale":
            value = int(expr["value"])
            if value == 0 or is_zero(argument):
                return scalar(0)
            if value == 1:
                return argument
            if value == -1:
                return simplify({"op": "neg", "arg": argument}, grade_support)
            if "scalar" in argument:
                return scalar(value * int(argument["scalar"]))
            if argument.get("op") == "scale":
                return simplify(
                    {
                        "op": "scale",
                        "value": value * int(argument["value"]),
                        "arg": argument["arg"],
                    },
                    grade_support,
                )
            return {"op": "scale", "value": value, "arg": argument}
        if op == "reverse":
            if argument.get("op") == "reverse":
                return simplify(argument["arg"], grade_support)
            if "var" in argument:
                grades = grade_support.get(str(argument["var"]), ())
                if len(grades) == 1:
                    sign = reverse_sign_for_grade(grades[0])
                    return (
                        argument
                        if sign == 1
                        else simplify({"op": "neg", "arg": argument}, grade_support)
                    )
            if "scalar" in argument:
                return argument
            return {"op": "reverse", "arg": argument}
        if op == "grade":
            grade = int(expr["grade"])
            if argument.get("op") == "grade":
                inner_grade = int(argument["grade"])
                return argument if grade == inner_grade else scalar(0)
            if "var" in argument:
                grades = grade_support.get(str(argument["var"]), ())
                if len(grades) == 1:
                    return argument if grades[0] == grade else scalar(0)
            if "scalar" in argument:
                return argument if grade == 0 else scalar(0)
            return {"op": "grade", "grade": grade, "arg": argument}

    arguments = [simplify(argument, grade_support) for argument in expr.get("args", [])]
    if op == "add":
        flattened: list[Expr] = []
        for argument in arguments:
            if is_zero(argument):
                continue
            if argument.get("op") == "add":
                flattened.extend(argument["args"])
            else:
                flattened.append(argument)
        if not flattened:
            return scalar(0)
        flattened.sort(key=canonical_json)
        if len(flattened) == 1:
            return flattened[0]
        return {"op": "add", "args": flattened}
    if op == "sub":
        left, right = arguments
        if is_zero(right):
            return left
        if is_zero(left):
            return simplify({"op": "neg", "arg": right}, grade_support)
        if canonical_json(left) == canonical_json(right):
            return scalar(0)
        return {"op": "sub", "args": [left, right]}
    if op in {"gp", "wedge", "commutator"}:
        left, right = arguments
        if is_zero(left) or is_zero(right):
            return scalar(0)
        return {"op": op, "args": [left, right]}
    raise ValueError(f"unsupported operation {op!r}")


def apply_scale(
    expr: Expr,
    factor: int,
    support: dict[str, tuple[int, ...]],
) -> Expr:
    return simplify({"op": "scale", "value": factor, "arg": expr}, support)


def normalize_relation(certificate: dict[str, Any]) -> dict[str, Any]:
    scope = certificate.get("scope", {})
    original_support = {
        str(variable["name"]): tuple(
            sorted(int(grade) for grade in variable.get("grades", []))
        )
        for variable in scope.get("variables", [])
    }
    lhs = copy.deepcopy(certificate["lhs"]["expression"])
    rhs = (
        scalar(0)
        if certificate.get("rhs") is None
        else copy.deepcopy(certificate["rhs"]["expression"])
    )
    lhs, mapping = rename_variables(lhs)
    rhs, mapping = rename_variables(rhs, mapping)
    support = {
        mapping.get(name, name): grades for name, grades in original_support.items()
    }
    lhs = apply_scale(lhs, int(certificate.get("lhs_scale", 1)), support)
    rhs = apply_scale(rhs, int(certificate.get("rhs_scale", 1)), support)

    candidates: list[tuple[str, Expr, Expr]] = []
    for left, right in ((lhs, rhs), (rhs, lhs)):
        candidates.append((canonical_json([left, right]), left, right))
        negative_left = simplify({"op": "neg", "arg": left}, support)
        negative_right = simplify({"op": "neg", "arg": right}, support)
        candidates.append(
            (
                canonical_json([negative_left, negative_right]),
                negative_left,
                negative_right,
            )
        )
    _, normalized_lhs, normalized_rhs = min(candidates, key=lambda item: item[0])
    payload = {
        "dimension": int(scope.get("dimension", 0)),
        "signature": list(scope.get("signature", [])),
        "variable_grades": [list(support[name]) for name in sorted(support)],
        "lhs": normalized_lhs,
        "rhs": normalized_rhs,
    }
    return {
        "lhs": normalized_lhs,
        "rhs": normalized_rhs,
        "support": support,
        "variable_mapping": mapping,
        "family_key": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "payload": payload,
    }


def expression_label(expr: Expr) -> str:
    if "var" in expr:
        return str(expr["var"])
    if "scalar" in expr:
        return str(expr["scalar"])
    op = expr["op"]
    if op == "neg":
        return f"-({expression_label(expr['arg'])})"
    if op == "reverse":
        return f"~({expression_label(expr['arg'])})"
    if op == "scale":
        return f"{expr['value']}*({expression_label(expr['arg'])})"
    if op == "grade":
        return f"<{expression_label(expr['arg'])}>_{expr['grade']}"
    if op == "add":
        return "(" + " + ".join(expression_label(arg) for arg in expr["args"]) + ")"
    left, right = expr["args"]
    if op == "commutator":
        return f"[{expression_label(left)},{expression_label(right)}]"
    symbol = {"sub": "-", "gp": "*", "wedge": "^"}[op]
    return f"({expression_label(left)} {symbol} {expression_label(right)})"


def contains_operation(expr: Expr, operation: str) -> bool:
    if expr.get("op") == operation:
        return True
    if "arg" in expr and contains_operation(expr["arg"], operation):
        return True
    return any(
        contains_operation(argument, operation) for argument in expr.get("args", [])
    )


def relation_tags(
    certificate: dict[str, Any],
    normalized: dict[str, Any],
) -> list[str]:
    original_lhs = certificate["lhs"]["expression"]
    original_rhs = (
        scalar(0)
        if certificate.get("rhs") is None
        else certificate["rhs"]["expression"]
    )
    lhs, rhs = normalized["lhs"], normalized["rhs"]
    tags: set[str] = set()
    if sum(expression_ops(value)["commutator"] for value in (lhs, rhs)) >= 3:
        if is_zero(lhs) or is_zero(rhs):
            tags.add("jacobi_or_nested_commutator")
    if contains_operation(original_lhs, "reverse") or contains_operation(
        original_rhs, "reverse"
    ):
        tags.add("reversion")
    if sum(
        expression_ops(value)["reverse"] for value in (original_lhs, original_rhs)
    ) >= 2:
        tags.add("reversion_involution_or_anti_automorphism")
    if contains_operation(lhs, "commutator") or contains_operation(rhs, "commutator"):
        tags.add("commutator")
    if contains_operation(lhs, "wedge") or contains_operation(rhs, "wedge"):
        tags.add("wedge")
    if contains_operation(original_lhs, "grade") or contains_operation(
        original_rhs, "grade"
    ):
        tags.add("grade_projection")
    if normalized["lhs"] == normalized["rhs"]:
        tags.add("normalization_tautology")
    if expression_cost(normalized["lhs"]) + expression_cost(normalized["rhs"]) <= 4:
        tags.add("low_complexity")
    return sorted(tags)


def identify_known_family(
    certificate: dict[str, Any],
    normalized: dict[str, Any],
) -> tuple[str, str]:
    lhs = normalized["lhs"]
    rhs = normalized["rhs"]
    operations = expression_ops(lhs) + expression_ops(rhs)
    original = certificate["lhs"].get("label", "") + "=" + (
        "0" if certificate.get("rhs") is None else certificate["rhs"].get("label", "")
    )
    normalized_json = canonical_json(lhs) + canonical_json(rhs)

    if operations["commutator"] >= 3 and (is_zero(lhs) or is_zero(rhs)):
        return "jacobi_identity", "known"
    if "commutator" in normalized_json:
        if "wedge" in normalized_json:
            return "vector_commutator_wedge", "known"
        if "[a,b]" in original and "[b,a]" in original:
            return "commutator_antisymmetry", "known"
    if "~((" in original and "~(b) * ~(a)" in original:
        return "reversion_anti_automorphism", "known"
    if "~(~(" in original:
        return "reversion_involution", "known"
    if "<_0" in original or ">_0" in original:
        if "<<(" in original:
            return "projection_idempotence", "tautological"
        if "~(" in original:
            return "scalar_part_reversion_invariance", "known"
        return "scalar_part_product_symmetry", "known"
    if ">_1" in original and " * " in original:
        return "declared_grade_support", "tautological"
    if "~(a)" in original or "~(b)" in original:
        return "declared_grade_reversion", "tautological"
    if "0 -" in original or "-1*(" in original:
        return "additive_or_scale_normalization", "tautological"
    return "unclassified_exact_relation", "candidate"


def novelty_score(
    normalized: dict[str, Any],
    status: str,
    tags: Iterable[str],
) -> float:
    lhs, rhs = normalized["lhs"], normalized["rhs"]
    complexity = expression_cost(lhs) + expression_cost(rhs)
    variables = len(expression_variables(lhs) | expression_variables(rhs))
    operation_count = sum(expression_ops(lhs).values()) + sum(
        expression_ops(rhs).values()
    )
    score = complexity * 3.0 + variables * 5.0 + operation_count * 2.0
    if status == "known":
        score -= 25.0
    elif status == "tautological":
        score -= 60.0
    tag_set = set(tags)
    if "normalization_tautology" in tag_set:
        score -= 50.0
    if "low_complexity" in tag_set:
        score -= 8.0
    if "jacobi_or_nested_commutator" in tag_set:
        score += 12.0
    return round(score, 3)


def load_certificates(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"no certificate JSON files found in {path}")
    records = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "lhs" not in data or "scope" not in data:
            raise ValueError(f"invalid certificate: {file}")
        data["_file"] = file.name
        records.append(data)
    return records


def audit(certificates: list[dict[str, Any]]) -> dict[str, Any]:
    audited = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for certificate in certificates:
        normalized = normalize_relation(certificate)
        tags = relation_tags(certificate, normalized)
        family, status = identify_known_family(certificate, normalized)
        record = {
            "relation_id": certificate.get("relation_id"),
            "certificate_file": certificate.get("_file"),
            "grammar": certificate.get("grammar"),
            "scope": certificate.get("scope"),
            "original": {
                "lhs_scale": certificate.get("lhs_scale", 1),
                "lhs": certificate["lhs"].get("label"),
                "rhs_scale": certificate.get("rhs_scale", 1),
                "rhs": (
                    "0"
                    if certificate.get("rhs") is None
                    else certificate["rhs"].get("label")
                ),
            },
            "normalized": {
                "lhs": expression_label(normalized["lhs"]),
                "rhs": expression_label(normalized["rhs"]),
            },
            "family_key": normalized["family_key"],
            "known_family": family,
            "status": status,
            "tags": tags,
            "novelty_score": novelty_score(normalized, status, tags),
            "generator_score": certificate.get("score"),
        }
        audited.append(record)
        groups[normalized["family_key"]].append(record)

    families = []
    for key, members in groups.items():
        members = sorted(
            members,
            key=lambda record: (
                -record["novelty_score"],
                str(record["relation_id"]),
            ),
        )
        representative = members[0]
        families.append(
            {
                "family_key": key,
                "representative_relation_id": representative["relation_id"],
                "known_family": representative["known_family"],
                "status": representative["status"],
                "normalized": representative["normalized"],
                "member_count": len(members),
                "members": [member["relation_id"] for member in members],
                "novelty_score": representative["novelty_score"],
            }
        )
    families.sort(
        key=lambda family: (
            -family["novelty_score"],
            family["known_family"],
            family["family_key"],
        )
    )
    audited.sort(
        key=lambda record: (-record["novelty_score"], str(record["relation_id"]))
    )
    counts = Counter(record["status"] for record in audited)
    return {
        "schema_version": 1,
        "engine": "geometric_identity_relation_audit_v3_1",
        "certificate_count": len(audited),
        "normalized_family_count": len(families),
        "status_counts": dict(sorted(counts.items())),
        "families": families,
        "relations": audited,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Geometric identity relation audit",
        "",
        f"- certificates: {report['certificate_count']}",
        f"- normalized families: {report['normalized_family_count']}",
        f"- known: {report['status_counts'].get('known', 0)}",
        f"- tautological/support-derived: {report['status_counts'].get('tautological', 0)}",
        f"- unclassified candidates: {report['status_counts'].get('candidate', 0)}",
        "",
        "| rank | normalized relation | class | status | members | novelty |",
        "|---:|---|---|---|---:|---:|",
    ]
    for rank, family in enumerate(report["families"], 1):
        relation = (
            f"`{family['normalized']['lhs']} = {family['normalized']['rhs']}`"
        )
        lines.append(
            f"| {rank} | {relation} | `{family['known_family']}` | "
            f"{family['status']} | {family['member_count']} | "
            f"{family['novelty_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit quotients syntactic presentation, first-occurrence variable naming,",
            "simultaneous sign, side order, double negation/reversion, redundant projections,",
            "and declared single-grade support. It is a ranking and deduplication layer, not",
            "an independent proof of the source certificates.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(load_certificates(args.certificates))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown_out.write_text(
            markdown(report),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"AUDIT: PASS certificates={report['certificate_count']} "
        f"families={report['normalized_family_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
