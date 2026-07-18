# Geometric Identity Discovery Engine V4

V4 begins with a host-only preflight for operators that can be expanded exactly into the validated V3 AST vocabulary.

The first macro operators are:

- grade involution: `hat(x) = sum_k (-1)^k <x>_k`
- Clifford conjugation: `bar(x) = reverse(hat(x))`

The preflight combines those macros with seed variables, geometric products, wedges, commutators, and reversion. Every macro is expanded before exact blade-wise integer polynomial classification.

## Initial grammar

`grammars/01_mixed_vector_bivector_involutions.json`

Scope:

- dimension 4
- signature `(2,2)`
- one vector variable
- one bivector variable

## Run

```powershell
New-Item -ItemType Directory -Force -Path .\local-evidence\v4 | Out-Null

python .\tools\geo_identity_v4_host_preflight.py `
    --grammar .\experiments\geometric_identity_engine_v4\grammars\01_mixed_vector_bivector_involutions.json `
    --output-json .\local-evidence\v4\mixed-involution-preflight.json `
    --markdown-out .\local-evidence\v4\mixed-involution-preflight.md
```

Expected marker:

```text
V4_PREFLIGHT: PASS expressions=<count> classes=<count> relations=<count>
```

## Boundary

This stage produces exact host-side polynomial equivalence classes only. It does not yet generate controls, finite-field prime matrices, or CUDA evidence. Contraction and duality require dedicated semantic definitions and are intentionally deferred until this involution preflight passes.
