# Geometric Identity Discovery Engine v2

The v2 stage extends the physically validated exact finite-field engine into a reproducible mathematical discovery pipeline.

## Pipeline

```text
checked v1 identities
    -> deterministic one-edit mutations
    -> canonical integer polynomial extraction by basis blade
    -> prime-matrix expansion
    -> exact deterministic CPU prechecks
    -> generated CUDA evaluators
    -> multi-prime counterexample search
    -> host witness reproduction
    -> greedy witness minimization
    -> cross-prime discovery report
```

## Exact polynomial layer

Every permitted v1 expression is polynomial in the independent scalar coefficients of the declared multivector variables. The v2 extractor expands each side exactly over the integers, preserving:

- basis blade;
- dimension;
- metric signature;
- variable grade support;
- scalar coefficient monomials;
- integer coefficients.

When every blade polynomial in `lhs - rhs` is identically zero, this is an exact symbolic proof for the declared dimension, signature, and variable grade supports. It does not automatically prove the same statement in other dimensions, signatures, or variable domains.

A nonzero primitive polynomial receives a stable canonical hash. Scalar multiples share the same primitive form, allowing mutations to be clustered into algebraic equivalence classes.

## Automatic mutations

The initial mutation generator performs deterministic one-edit changes including:

- whole-side sign inversion;
- operand exchange;
- geometric product to wedge or commutator;
- wedge to geometric product;
- commutator to product or wedge;
- subtraction to addition;
- reverse removal;
- grade-projection removal or adjacent-grade substitution;
- scale-sign inversion.

Every mutation records its source statement, side, expression path, edit type, and stable identifier.

## Multi-prime search

The default exact fields are:

- 65,521;
- 65,519;
- 65,497;
- 32,749.

The same structural variant is emitted once for every configured prime. Canonical polynomial hashes remain prime-independent, while CUDA results and witnesses are recorded separately by field.

## Witness reduction

A GPU witness is first reproduced by the exact host evaluator. The reducer then greedily:

1. removes coefficients while preserving failure;
2. reduces surviving coefficient magnitudes;
3. reports the resulting active basis blades and coefficient values.

The minimized witness remains an exact counterexample in the specified finite field.

## Initial physical gate

The first v2 physical run should use:

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_discovery.ps1 `
    -Assignments 262144 `
    -CpuChecks 1024 `
    -Primes @(65521,65519,65497,32749) `
    -MaxMutations 8 `
    -Archive
```

This produces 116 finite-field statements from the five v1 controls:

- 20 cross-prime originals;
- 96 one-edit mutations of the three known identities.

The exact count changes when the mutation limit or prime matrix changes.

## Interpretation boundary

A nonzero polynomial proves that the two formal expressions are not universally identical for the declared structural setting. The CUDA search is then used to locate a concrete finite-field witness and validate the generated backend. A zero polynomial is stronger than sampled survival: it is exact symbolic cancellation for that declared setting.

Cross-prime sampling is still not a proof across unspecified dimensions or signatures, and a finite-field witness does not by itself identify the most natural characteristic-zero counterexample. The witness reducer is intended to bridge that gap by producing small human-readable assignments.
