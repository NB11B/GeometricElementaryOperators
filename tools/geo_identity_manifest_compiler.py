#!/usr/bin/env python3
"""Compile a generated geometric identity corpus without long command lines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geo_identity_compiler as compiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile all identity files listed by a discovery manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-checks", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def manifest_entries(manifest: object) -> list[dict[str, object]]:
    if not isinstance(manifest, dict):
        raise compiler.IdentityError("manifest must be a JSON object")

    entries = manifest.get("entries")
    if isinstance(entries, list) and entries:
        return entries

    statements = manifest.get("statements")
    if isinstance(statements, list) and statements:
        return statements

    raise compiler.IdentityError("manifest contains no identity entries")


def entry_file(entry: object, index: int) -> str:
    if not isinstance(entry, dict):
        raise compiler.IdentityError(f"manifest entry {index} must be an object")

    value = entry.get("file")
    if not isinstance(value, str):
        value = entry.get("path")
    if not isinstance(value, str) or not value:
        raise compiler.IdentityError(
            f"manifest entry {index} does not contain a file/path string"
        )
    return value


def main() -> int:
    args = parse_args()
    try:
        if args.python_checks < 0:
            raise compiler.IdentityError("--python-checks must be non-negative")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = manifest_entries(manifest)
        base_directory = args.manifest.parent
        relative_paths: list[str] = []
        identity_paths: list[Path] = []
        for index, entry in enumerate(entries):
            relative = entry_file(entry, index)
            path = base_directory / relative
            if not path.is_file():
                raise compiler.IdentityError(f"identity file does not exist: {path}")
            relative_paths.append(relative.replace("\\", "/"))
            identity_paths.append(path)
        identities = compiler.load_corpus(identity_paths)
        generated = compiler.emit_header(identities, relative_paths)
        for identity in identities:
            found_counterexample = False
            for assignment in range(args.python_checks):
                equal, _, _, _ = compiler.evaluate_identity(identity, assignment)
                if not equal:
                    found_counterexample = True
                    if identity.expected == "identity":
                        raise compiler.IdentityError(
                            f"{identity.name} failed exact Python check at "
                            f"assignment {assignment}"
                        )
                    break
            if (
                args.python_checks > 0
                and identity.expected == "counterexample"
                and not found_counterexample
            ):
                raise compiler.IdentityError(
                    f"{identity.name} produced no counterexample in "
                    f"{args.python_checks} Python checks"
                )
    except (OSError, json.JSONDecodeError, compiler.IdentityError) as exc:
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
        print(
            f"PASS: {args.output} matches {len(identities)} manifest identities"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"WROTE: {args.output}")
    print(f"IDENTITIES: {len(identities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
