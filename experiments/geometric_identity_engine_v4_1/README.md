# Geometric Identity Engine V4.1

V4.1 extends the accepted V4 mixed-grade pipeline with an explicit contraction contract and a fixed basis-blade constant needed for exact duality.

## Contracts

The algebra is `Cl(2,2)` with signature `[+1,+1,-1,-1]`.

- Left contraction: `A_r ⌟ B_s = <reverse(A_r) B_s>_{s-r}` for `r <= s`, otherwise zero, extended bilinearly by grade decomposition.
- Right contraction: `A_r ⌞ B_s = <A_r reverse(B_s)>_{r-s}` for `r >= s`, otherwise zero, extended bilinearly by grade decomposition.
- Pseudoscalar: `I=e1234`, represented by `{"fixed_blade":{"blade":15,"coefficient":1}}`.
- In this signature, `I^2=1`, so `I^-1=I`.
- Right dual: `dual(A)=A I`.

## Safety architecture

The accepted V1-V4 compiler and exact backend remain unchanged. V4.1 is implemented as a feature-gated extension layer:

- `geo_identity_v4_1_exact.py` installs strict fixed-blade validation and exact symbolic/finite-field semantics.
- `geo_identity_compiler_v4_1.py` extends the established DAG, Python evaluator, and generated host/CUDA emitter.
- malformed fixed-blade nodes, zero coefficients, out-of-range blades, booleans, extra fields, and coefficients that vanish modulo the statement prime are rejected.
- existing regression suites run before any V4.1 corpus is compiled.

## Host acceptance gate

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_1_host_gate.ps1
```

The gate runs the existing identity-engine tests, the V4.1 fixed-blade tests, exact corpus generation, exact corpus validation, deterministic control prechecks, and generated-header compilation with Python evaluation.

Expected markers:

```text
V4_1_DUALITY_CORPUS: PASS identities=20 controls=8 statements=28
V4_1_DUALITY_CORPUS_VALIDATION: PASS statements=28 identities=20 controls=8
V4_1_HOST_GATE: PASS
```

Counts are rows across the default two-prime matrix: ten identity definitions and four control definitions.

## Physical CUDA gate

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_1_duality_clean.ps1 `
    -Device 0 `
    -Assignments 131072 `
    -CpuChecks 512 `
    -Archive
```

The clean wrapper removes stale Visual Studio environment variables, uses the long host-compiler path correction, validates dynamic corpus names, packages the exact corpus with the evidence, regenerates the SHA-256 manifest after packaging, and restores the generated header.

Expected final markers:

```text
VALIDATION: PASS
GEO_IDENTITY_V4_1_DUALITY,status=complete
```

## Interpretation boundary

A zero characteristic-zero difference polynomial is exact for the declared dimension, signature, grade supports, fixed-blade constants, and expression semantics. A finite-field witness exactly rejects the corresponding modular control. Neither result alone establishes novelty or generalization outside the declared contract.
