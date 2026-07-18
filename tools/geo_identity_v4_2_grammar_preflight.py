#!/usr/bin/env python3
"""Native V4.2 grammar preflight with fixed blades and mixed supports."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import geo_identity_v4_2_engine as engine
except ModuleNotFoundError:
    from tools import geo_identity_v4_2_engine as engine


def expression_spec(config: dict[str, Any], signature: list[int], expression: dict[str, Any]) -> dict[str, Any]:
    names = engine.expression_variables(expression)
    variables = [{"name": item["name"], "grades": item["grades"]} for item in config["variables"] if item["name"] in names]
    if not variables:
        first = config["variables"][0]
        variables = [{"name": first["name"], "grades": first["grades"]}]
    return {
        "schema_version": 1,
        "name": "v4_2_grammar_expression",
        "description": "V4.2 grammar expression classifier",
        "expected": "counterexample",
        "dimension": config["dimension"],
        "signature": signature,
        "prime": config["primes"][0],
        "coefficient_bound": config["coefficient_bound"],
        "seed": config["seed"],
        "variables": variables,
        "lhs": expression,
        "rhs": {"scalar": 0},
    }


def seeds(config: dict[str, Any], signature: list[int]) -> list[tuple[str, dict[str, Any], str]]:
    dimension = int(config["dimension"])
    variables = {item["name"]: engine.variable(item["name"]) for item in config["variables"]}
    I = engine.pseudoscalar(signature)
    I_inv = engine.pseudoscalar(signature, True)
    output: list[tuple[str, dict[str, Any], str]] = [
        ("I", I, "fixed_blade"),
        ("Iinv", I_inv, "fixed_blade"),
        ("I*I", engine.gp(I, I), "pseudoscalar_contract"),
        ("Iinv*I", engine.gp(I_inv, I), "pseudoscalar_contract"),
    ]
    for name, value in variables.items():
        output.extend([
            (name, value, "seed"),
            (f"dual_r({name})", engine.right_dual(value, signature), "right_dual"),
            (f"dual_l({name})", engine.left_dual(value, signature), "left_dual"),
            (f"dual_r2({name})", engine.right_dual(engine.right_dual(value, signature), signature), "dual_square"),
        ])
    for left_name, right_name in (("v", "B"), ("v", "T"), ("B", "T"), ("E", "O")):
        left = variables[left_name]
        right = variables[right_name]
        output.extend([
            (f"lcon({left_name},{right_name})", engine.left_contraction(left, right, dimension), "left_contraction"),
            (f"rcon({right_name},{left_name})", engine.right_contraction(right, left, dimension), "right_contraction"),
            (f"dual_r(lcon({left_name},{right_name}))", engine.right_dual(engine.left_contraction(left, right, dimension), signature), "contraction_duality"),
            (f"{left_name}^dual_r({right_name})", engine.wedge(left, engine.right_dual(right, signature)), "contraction_duality_candidate"),
        ])
    return output


def build(config: dict[str, Any]) -> dict[str, Any]:
    signature_reports: list[dict[str, Any]] = []
    for signature_row in config["signatures"]:
        signature_name = signature_row["name"]
        signature = [int(value) for value in signature_row["metric"]]
        records: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[str] = set()
        for label, expression, source in seeds(config, signature):
            canonical = engine.canonicalize(expression)
            syntax_key = engine.canonical_json(canonical)
            if syntax_key in seen:
                continue
            seen.add(syntax_key)
            polynomial = engine.extract_polynomial(expression_spec(config, signature, canonical), int(config["corpus"].get("term_limit", 500000)))
            record = {
                "label": label,
                "source": source,
                "expression": canonical,
                "cost": engine.expression_cost(canonical),
                "variables": sorted(engine.expression_variables(canonical)),
                "polynomial_hash": polynomial["hash"],
                "zero": polynomial["zero"],
                "total_terms": polynomial["total_terms"],
                "maximum_degree": polynomial["maximum_degree"],
            }
            records.append(record)
            groups[polynomial["hash"]].append(record)
        relations: list[dict[str, Any]] = []
        zero_expressions = 0
        for polynomial_hash, members in groups.items():
            if all(member["zero"] for member in members):
                zero_expressions += len(members)
                continue
            members.sort(key=lambda item: (item["cost"], item["label"]))
            if len(members) < 2:
                continue
            representative = members[0]
            for other in members[1:]:
                relations.append({
                    "polynomial_hash": polynomial_hash,
                    "lhs": representative["label"],
                    "rhs": other["label"],
                    "lhs_source": representative["source"],
                    "rhs_source": other["source"],
                    "combined_cost": representative["cost"] + other["cost"],
                })
        relations.sort(key=lambda item: (item["combined_cost"], item["lhs"], item["rhs"]))
        signature_reports.append({
            "signature_name": signature_name,
            "signature": signature,
            "pseudoscalar_square": engine.pseudoscalar_square(signature),
            "expression_count": len(records),
            "equivalence_class_count": len(groups),
            "zero_expression_count": zero_expressions,
            "relation_count": len(relations),
            "records": records,
            "relations": relations,
        })
    return {
        "schema_version": 1,
        "engine": "geometric_identity_v4_2_native_grammar_preflight",
        "signature_count": len(signature_reports),
        "signatures": signature_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = build(config)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# V4.2 native grammar preflight",
            "",
            "| signature | I^2 | expressions | classes | zero expressions | exact relations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in report["signatures"]:
            lines.append(f"| `{row['signature_name']}` | {row['pseudoscalar_square']} | {row['expression_count']} | {row['equivalence_class_count']} | {row['zero_expression_count']} | {row['relation_count']} |")
        lines.extend([
            "",
            "The grammar includes native fixed pseudoscalars, right and left duals, dual squares, homogeneous and mixed-grade variables, contractions, and contraction-duality candidates. Universal zero expressions are excluded from relation ranking.",
            "",
        ])
        args.markdown_out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, engine.DiscoveryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    total = sum(row["relation_count"] for row in report["signatures"])
    print(f"V4_2_GRAMMAR_PREFLIGHT: PASS signatures={report['signature_count']} relations={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
