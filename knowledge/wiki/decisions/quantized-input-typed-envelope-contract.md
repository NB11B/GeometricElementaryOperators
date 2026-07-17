# Quantized-input typed numerical-envelope contract

Status: accepted and verified for v0.19 Stage 3.

## Decision

Fixed GEB-36 validation is a strict typed floating-reference envelope over
decoded quantized inputs:

- capability is the exact frozen set of 36 target names and result kinds;
- fixtures are deterministic raw Q-format values and the floating reference
  receives their decoded values;
- every requested sample must complete with the expected kind and zero failure
  counts;
- error budgets are explicit per operation in fixed-point LSBs;
- projective normalization is performed in double precision;
- the executable reports compiled precision and Q format, and the reporter
  rejects disagreement with the request; and
- the Q30 envelope uses a double reference because float cannot reliably
  resolve one Q30 LSB.

The requested seed is also part of the contract. A deterministic 64-fixture
preflight rejects a corpus without both signs, distinct operands, mixed-sign
general values, and nonparallel vectors.

## Rationale

A comparison against unquantized floating inputs confounds input quantization
with fixed execution error. A permissive report can hide a missing target,
wrong result variant, overflow, or incomplete run. Trusting a command-line
precision label can also report a false precision when the executable was
compiled differently. The strict manifest, decoded-input reference, complete
accounting, compiled-configuration handshake, and diversity preflight make
those failures observable.

## Consequences

- Changes to target names, typed results, or per-operation budgets require an
  intentional manifest and contract-test update.
- Bounded fixtures are chosen to complete; overflow is a report failure rather
  than an excluded sample.
- Passing this envelope does not establish an independent oracle because the
  floating GEB reference shares implementation primitives.
- Float and double host builds are both tested at Q16, while Q30 LSB claims are
  made only against the explicitly compiled double reference.

## Provenance

- `benchmarks/bench_numerical.c`
- `tools/numerical_report.py`
- `tests/test_numerical_report.py`
- `.github/workflows/ci.yml`
- `docs/v0.19-validation.md`
- `knowledge/wiki/components/fixed-geb36-numerical-envelope.md`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
