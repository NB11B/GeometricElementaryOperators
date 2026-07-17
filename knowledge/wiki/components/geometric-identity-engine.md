---
id: component:geometric-identity-engine
title: "Geometric Identity and Counterexample Engine"
type: component
status: experimental
created: 2026-07-17
updated: 2026-07-17
confidence: implemented-unvalidated-on-gpu
---

# Geometric Identity and Counterexample Engine

The engine compiles structured Clifford-algebra statements into exact host and CUDA evaluators.

## Pipeline

```text
checked identity JSON
→ validated typed expression
→ canonical DAG
→ common-subexpression elimination
→ blade-support propagation
→ exact modular host/CUDA evaluator
→ CPU prefix verification
→ GPU assignment search
→ exact witness or surviving conjecture
```

## Distinguishing properties

- mathematical grade and signature remain explicit through compilation;
- coefficients are exact modulo a declared prime;
- generated CPU and GPU evaluators share one semantic source;
- false statements return a reproducible assignment, blade, and coefficient mismatch;
- support masks permit backend compilers to remove impossible blade operations.

## Current limits

- dimensions at most six;
- non-degenerate diagonal signatures;
- one odd-prime field per identity;
- bounded deterministic assignments;
- no proof generation;
- no symbolic polynomial certificate yet.

## Primary files

- `tools/geo_identity_compiler.py`
- `experiments/geometric_identity_engine/`
- `benchmarks/geo_identity_search/`
- `tests/test_geo_identity_compiler.py`
