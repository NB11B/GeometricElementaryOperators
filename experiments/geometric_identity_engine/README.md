# Geometric Identity and Counterexample Engine

Status: v1 implementation scaffold with an exact Python backend, checked expression DAG, generated host/CUDA evaluator, and deterministic GPU search harness.

## Purpose

This experiment treats structured mathematics as compilable input.

A checked Clifford-algebra identity is described once in JSON. The compiler:

1. validates the coefficient domain, dimension, signature, variables, and operations;
2. constructs a canonical expression DAG;
3. performs exact common-subexpression elimination;
4. propagates exact possible basis-blade support through the DAG;
5. emits one evaluator usable by host C++ and CUDA;
6. evaluates assignments exactly modulo an odd prime;
7. searches millions of independent assignments on the GPU;
8. returns a reproducible minimal assignment index and mismatching blade for false statements.

The GPU does not prove an identity over the integers or reals. It provides an exact finite-field stress test and counterexample search. Surviving statements remain conjectures requiring symbolic or formal proof.

## V1 mathematical domain

- Clifford algebras of dimension 1 through 6;
- non-degenerate diagonal signatures with entries `+1` or `-1`;
- exact arithmetic modulo an odd prime;
- variables constrained to one or more grades;
- deterministic bounded coefficients;
- scalar constants;
- addition, subtraction, negation, and integer scaling;
- geometric product;
- exterior/wedge product;
- reversion;
- grade projection;
- commutator as a derived operation.

Dimensions are capped at six in v1 because possible basis support is represented by one 64-bit mask. A later typed-DAG revision can replace this with arbitrary-width support sets.

## Identity IR

Each identity includes:

```json
{
  "schema_version": 1,
  "name": "vector_square_is_scalar",
  "expected": "identity",
  "dimension": 4,
  "signature": [1, 1, 1, 1],
  "prime": 65521,
  "coefficient_bound": 3,
  "seed": 610839776,
  "variables": [
    {"name": "a", "grades": [1]}
  ],
  "lhs": {
    "op": "gp",
    "args": [{"var": "a"}, {"var": "a"}]
  },
  "rhs": {
    "op": "grade",
    "grade": 0,
    "arg": {
      "op": "gp",
      "args": [{"var": "a"}, {"var": "a"}]
    }
  }
}
```

The repeated geometric product is represented once in the canonical DAG.

The schema is recorded at:

```text
experiments/geometric_identity_engine/geo_identity_v1.schema.json
```

## Initial corpus

The checked corpus contains three true identities and two deliberately mutated false statements:

| Name | Expected |
|---|---|
| `vector_square_is_scalar` | identity |
| `reverse_reverses_product_order` | identity |
| `vector_wedge_is_antisymmetric` | identity |
| `vector_product_is_not_commutative` | counterexample |
| `reverse_preserves_product_order_false` | counterexample |

The false statements are part of the acceptance test. The engine must locate an exact witness rather than merely reporting a floating-point discrepancy.

## Exact semantics

Every coefficient is normalized modulo the identity's prime.

For basis blades \(e_A\) and \(e_B\), the generated geometric product computes:

\[
e_A e_B =
(-1)^{N(A,B)}
\left(\prod_{i \in A \cap B} g_{ii}\right)
e_{A \triangle B},
\]

where \(N(A,B)\) is the number of basis swaps required to restore canonical blade order.

The wedge product is zero when the blade supports overlap and otherwise retains the canonical permutation sign.

Reversion applies:

\[
\widetilde{e_A}
=
(-1)^{r(r-1)/2} e_A,
\]

where \(r\) is the blade grade.

## Deterministic assignments

One GPU thread evaluates one complete assignment.

Assignment coefficients are generated from:

- identity seed;
- assignment index;
- variable index;
- blade index.

The counter-based `splitmix64` mapping means assignments require no mutable random state and are reproduced identically on CPU and GPU.

A variable that would otherwise be entirely zero is deterministically given a coefficient of one on its first allowed blade.

## Generated evaluator

The compiler generates during the build/evidence run:

```text
benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh
```

For every identity, the header contains:

- dimension and signature;
- coefficient domain;
- grade-constrained assignment generator;
- exact modular arithmetic;
- geometric and wedge product;
- reversion and grade projection;
- canonical DAG evaluation;
- exact witness reporting.

Node support masks are compile-time template parameters. This permits the CUDA compiler to remove blade loops that cannot contribute to a node.

## GPU search contract

The CUDA benchmark:

- runs every identity in the checked corpus;
- performs an exact CPU prefix check;
- searches a configurable assignment range on the GPU;
- uses `atomicMin` to preserve the smallest failing assignment index observed;
- re-evaluates every GPU witness on the host;
- requires true identities to have no witness;
- requires mutated false identities to produce a witness;
- reports exact blade and coefficient disagreement;
- records assignments per second and kernel time.

Default search:

```text
1,048,576 assignments per identity
4,096 exact CPU prefix checks
256 CUDA threads per block
```

## Build and run

```powershell
cd C:\Users\nateb\Documents\GeometricElementaryOperators

python .\tools\geo_identity_compiler.py `
    --identity .\experiments\geometric_identity_engine\corpus\01_vector_square_scalar.json `
    --identity .\experiments\geometric_identity_engine\corpus\02_reverse_product_order.json `
    --identity .\experiments\geometric_identity_engine\corpus\03_vector_wedge_antisymmetry.json `
    --identity .\experiments\geometric_identity_engine\corpus\04_vector_product_commutativity_false.json `
    --identity .\experiments\geometric_identity_engine\corpus\05_mutated_reverse_order_false.json `
    --output .\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh `
    --python-checks 256

python -m unittest tests.test_geo_identity_compiler
```

Physical CUDA run:

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_search.ps1 `
    -Assignments 1048576 `
    -CpuChecks 4096 `
    -Archive
```

## Acceptance

The v1 milestone passes when:

- repeated generation is byte-for-byte deterministic;
- all host generator tests pass;
- exact Python checks retain every known identity;
- exact Python checks reject every mutation;
- the host C++ smoke executable passes;
- the CUDA executable builds;
- true identities survive the configured modular assignment set;
- every false identity returns a host-reproducible witness;
- all CSV rows report `pass`;
- the evidence summary reports `VALIDATION: PASS`;
- the evidence package and archive receive SHA-256 manifests.

## Interpretation boundary

A finite-field search can disprove a purported universal polynomial identity when it finds a valid witness. Failure to find a witness is not a proof over characteristic zero.

The initial corpus is intentionally elementary. Its purpose is to validate the compiler, exact semantics, CPU/GPU agreement, deterministic witness mechanism, and evidence architecture before attempting unknown identities or larger classification searches.

## Next mathematical stages

1. Multiple independent primes for stronger polynomial identity testing.
2. Integer and rational exact backends.
3. Interval and floating backends for analytic expressions.
4. Arbitrary-width blade support and dimensions beyond six.
5. Typed intermediate values carrying grade, signature, symmetry, and coefficient domain.
6. Canonical symbolic polynomial extraction.
7. Automated mutation and candidate-identity generation.
8. Minimal witness reduction by coefficient magnitude and blade support.
9. Export of normalized identities and witnesses to Lean, Coq, or Isabelle.
10. Operator-family and representation-classification searches.
