#!/usr/bin/env python3
"""Compile a V4.1 fixed-blade manifest into the shared host/CUDA header."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geo_identity_v4_1_ir as ir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-checks", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def manifest_paths(path: Path) -> tuple[list[str], list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("statements")
    if not isinstance(rows, list) or not rows:
        raise ir.IdentityError("manifest contains no statements")
    relative: list[str] = []
    absolute: list[Path] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ir.IdentityError(f"manifest statement {index} has no path")
        rel = row["path"].replace("\\", "/")
        target = path.parent / rel
        if not target.is_file():
            raise ir.IdentityError(f"identity file does not exist: {target}")
        relative.append(rel)
        absolute.append(target)
    return relative, absolute


def main() -> int:
    args = parse_args()
    try:
        if args.python_checks < 0:
            raise ir.IdentityError("--python-checks must be non-negative")
        relative, paths = manifest_paths(args.manifest)
        identities = ir.load_corpus(paths)
        generated = ir.emit_header(identities, relative)
        for identity in identities:
            found = False
            for assignment in range(args.python_checks):
                equal, _, _, _ = ir.evaluate_identity(identity, assignment)
                if not equal:
                    found = True
                    if identity.expected == "identity":
                        raise ir.IdentityError(
                            f"{identity.name} failed exact Python check at assignment {assignment}"
                        )
                    break
            if args.python_checks and identity.expected == "counterexample" and not found:
                raise ir.IdentityError(
                    f"{identity.name} produced no counterexample in {args.python_checks} checks"
                )
    except (OSError, json.JSONDecodeError, ir.IdentityError, ir.DiscoveryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: unable to read {args.output}: {exc}", file=sys.stderr)
            return 2
        if existing != generated:
            print(f"ERROR: {args.output} is stale", file=sys.stderr)
            return 1
        print(f"PASS: {args.output} matches {len(identities)} manifest identities")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"WROTE: {args.output}")
    print(f"IDENTITIES: {len(identities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
