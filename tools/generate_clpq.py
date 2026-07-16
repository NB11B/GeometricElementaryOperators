#!/usr/bin/env python3
"""Generate portable C geometric-algebra kernels for Cl(p,q).

The generated implementation uses bit-mask blades and a frozen multiplication
lookup table, making it suitable for reference validation and subsequent
specialization into unrolled kernels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def product_sign(left: int, right: int, p: int, q: int) -> tuple[int, int]:
    n = p + q
    sign = 1
    for i in range(n):
        if left & (1 << i):
            lower = right & ((1 << i) - 1)
            if lower.bit_count() & 1:
                sign = -sign
    common = left & right
    for i in range(p, n):
        if common & (1 << i):
            sign = -sign
    return sign, left ^ right


def grade(mask: int) -> int:
    return mask.bit_count()


def blade_name(mask: int, n: int) -> str:
    if mask == 0:
        return "scalar"
    return "e" + "".join(str(i + 1) for i in range(n) if mask & (1 << i))


def emit_header(p: int, q: int, prefix: str) -> str:
    n = p + q
    blades = 1 << n
    guard = f"{prefix.upper()}_H"
    return f'''#ifndef {guard}
#define {guard}

#include <stdbool.h>
#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define {prefix.upper()}_P {p}
#define {prefix.upper()}_Q {q}
#define {prefix.upper()}_DIMENSION {n}
#define {prefix.upper()}_BLADE_COUNT {blades}

typedef struct {{
    geo_real_t c[{blades}];
}} {prefix}_t;

{prefix}_t {prefix}_zero(void);
{prefix}_t {prefix}_basis(unsigned mask);
{prefix}_t {prefix}_add({prefix}_t a, {prefix}_t b);
{prefix}_t {prefix}_mul({prefix}_t a, {prefix}_t b);
{prefix}_t {prefix}_reverse({prefix}_t value);
bool {prefix}_near({prefix}_t a, {prefix}_t b, geo_real_t tolerance);

#ifdef __cplusplus
}}
#endif

#endif
'''


def emit_source(p: int, q: int, prefix: str, header_name: str) -> str:
    n = p + q
    blades = 1 << n
    table_rows = []
    for left in range(blades):
        row = []
        for right in range(blades):
            sign, out = product_sign(left, right, p, q)
            encoded = out + (blades if sign < 0 else 0)
            row.append(str(encoded))
        table_rows.append("    {" + ", ".join(row) + "}")
    table_text = ",\n".join(table_rows)
    reverse_signs = ["-1" if (grade(i) * (grade(i) - 1) // 2) & 1 else "1" for i in range(blades)]
    reverse_text = ", ".join(reverse_signs)
    return f'''#include "{header_name}"

#include <math.h>
#include <stddef.h>

static const unsigned short GEO_PRODUCT_TABLE[{blades}][{blades}] = {{
{table_text}
}};

static const signed char GEO_REVERSE_SIGNS[{blades}] = {{{reverse_text}}};

{prefix}_t {prefix}_zero(void) {{
    {prefix}_t result = {{{{0}}}};
    return result;
}}

{prefix}_t {prefix}_basis(unsigned mask) {{
    {prefix}_t result = {prefix}_zero();
    if (mask < {blades}u) result.c[mask] = (geo_real_t)1;
    return result;
}}

{prefix}_t {prefix}_add({prefix}_t a, {prefix}_t b) {{
    {prefix}_t result;
    size_t i;
    for (i = 0u; i < {blades}u; ++i) result.c[i] = a.c[i] + b.c[i];
    return result;
}}

{prefix}_t {prefix}_mul({prefix}_t a, {prefix}_t b) {{
    {prefix}_t result = {prefix}_zero();
    size_t i;
    size_t j;
    for (i = 0u; i < {blades}u; ++i) {{
        for (j = 0u; j < {blades}u; ++j) {{
            const unsigned short encoded = GEO_PRODUCT_TABLE[i][j];
            const unsigned out = encoded % {blades}u;
            const geo_real_t sign = encoded >= {blades}u ? (geo_real_t)-1 : (geo_real_t)1;
            result.c[out] += sign * a.c[i] * b.c[j];
        }}
    }}
    return result;
}}

{prefix}_t {prefix}_reverse({prefix}_t value) {{
    size_t i;
    for (i = 0u; i < {blades}u; ++i) value.c[i] *= (geo_real_t)GEO_REVERSE_SIGNS[i];
    return value;
}}

bool {prefix}_near({prefix}_t a, {prefix}_t b, geo_real_t tolerance) {{
    size_t i;
    for (i = 0u; i < {blades}u; ++i) {{
        if (fabs((double)(a.c[i] - b.c[i])) > (double)tolerance) return false;
    }}
    return true;
}}
'''


def manifest(p: int, q: int, prefix: str) -> dict:
    n = p + q
    blades = 1 << n
    return {
        "signature": {"p": p, "q": q},
        "prefix": prefix,
        "dimension": n,
        "blade_count": blades,
        "blades": [{"mask": i, "name": blade_name(i, n), "grade": grade(i)} for i in range(blades)],
        "multiplication": [
            [product_sign(i, j, p, q) for j in range(blades)] for i in range(blades)
        ],
    }


def self_test() -> None:
    for p, q in [(2, 0), (3, 0), (1, 3), (0, 2)]:
        n = p + q
        for i in range(n):
            sign, out = product_sign(1 << i, 1 << i, p, q)
            assert out == 0
            assert sign == (1 if i < p else -1)
        blades = 1 << n
        for a in range(blades):
            for b in range(blades):
                sab, ab = product_sign(a, b, p, q)
                rb = -1 if (grade(b) * (grade(b) - 1) // 2) & 1 else 1
                ra = -1 if (grade(a) * (grade(a) - 1) // 2) & 1 else 1
                rab = -1 if (grade(ab) * (grade(ab) - 1) // 2) & 1 else 1
                sba, ba = product_sign(b, a, p, q)
                assert ab == ba
                assert sab * rab == rb * ra * sba


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--prefix", default="geo_clpq")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.p < 0 or args.q < 0 or args.p + args.q > 8:
        parser.error("require p,q >= 0 and p+q <= 8")
    if args.self_test:
        self_test()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    header = args.out_dir / f"{args.prefix}.h"
    source = args.out_dir / f"{args.prefix}.c"
    metadata = args.out_dir / f"{args.prefix}.json"
    header.write_text(emit_header(args.p, args.q, args.prefix), encoding="utf-8")
    source.write_text(emit_source(args.p, args.q, args.prefix, header.name), encoding="utf-8")
    metadata.write_text(json.dumps(manifest(args.p, args.q, args.prefix), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
