#!/usr/bin/env python3
"""Small independent verifier for V4.8 geometric operator certificates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def gp_sign(left: int, right: int, signature: Sequence[int]) -> int:
    sign = 1
    for index, metric in enumerate(signature):
        bit = 1 << index
        if left & bit:
            if (right & (bit - 1)).bit_count() & 1:
                sign = -sign
            if right & bit:
                sign *= int(metric)
    return sign


def expected_rows(certificate: dict[str, Any]) -> list[list[dict[str, int]]]:
    dimension = int(certificate["dimension"])
    signature = [int(value) for value in certificate["signature"]]
    side = str(certificate["side"])
    if len(signature) != dimension or any(value not in (-1, 1) for value in signature):
        raise ValueError("invalid signature")
    if side not in {"left", "right"}:
        raise ValueError("invalid side")
    blade_count = 1 << dimension
    terms = certificate["terms"]
    if not isinstance(terms, list) or not terms:
        raise ValueError("certificate has no terms")
    rows: list[list[dict[str, int]]] = [[] for _ in range(blade_count)]
    seen: set[int] = set()
    for term in terms:
        blade = int(term["blade"])
        coefficient = int(term["coefficient"])
        if not 0 <= blade < blade_count or coefficient == 0 or blade in seen:
            raise ValueError("invalid fixed term")
        seen.add(blade)
        for source in range(blade_count):
            factor = coefficient * (
                gp_sign(source, blade, signature)
                if side == "right"
                else gp_sign(blade, source, signature)
            )
            rows[source ^ blade].append({"source": source, "factor": factor})
    for row in rows:
        row.sort(key=lambda item: (item["source"], item["factor"]))
    return rows


def dense_matrix(rows: list[list[dict[str, int]]]) -> list[list[int]]:
    size = len(rows)
    matrix = [[0] * size for _ in range(size)]
    for target, row in enumerate(rows):
        for item in row:
            matrix[target][int(item["source"])] += int(item["factor"])
    return matrix


def verify(certificate: dict[str, Any]) -> None:
    if certificate.get("schema_version") != 1:
        raise ValueError("unsupported certificate schema")
    if certificate.get("certificate_type") != "geometric_fixed_operator_equivalence":
        raise ValueError("unsupported certificate type")
    unsigned = dict(certificate)
    supplied_hash = unsigned.pop("certificate_hash", None)
    if supplied_hash != digest(unsigned):
        raise ValueError("certificate hash mismatch")
    rows = expected_rows(certificate)
    if rows != certificate.get("rows"):
        raise ValueError("row plan mismatch")
    if digest(dense_matrix(rows)) != certificate.get("matrix_hash"):
        raise ValueError("matrix hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    checked = 0
    try:
        for candidate in args.paths:
            paths = sorted(candidate.glob("*.json")) if candidate.is_dir() else [candidate]
            for path in paths:
                verify(json.loads(path.read_text(encoding="utf-8")))
                checked += 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if checked == 0:
        print("ERROR: no certificates were checked", file=sys.stderr)
        return 2
    print(f"V4_8_INDEPENDENT_VERIFY: PASS certificates={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
