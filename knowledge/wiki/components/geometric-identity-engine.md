---
id: component:geometric-identity-engine
title: "Geometric Identity and Counterexample Engine"
type: component
status: experimental
created: 2026-07-17
updated: 2026-07-17
confidence: verified-v1-v2-implemented
---

# Geometric Identity and Counterexample Engine

The engine compiles structured Clifford-algebra statements into exact host and CUDA evaluators. V1 physically validated exact finite-field counterexample search. V2 adds symbolic polynomial extraction, deterministic candidate mutation, multi-prime expansion, and exact witness reduction.

## Pipeline

```text
checked identity JSON
→ validated typed expression
→ canonical DAG
→ common-subexpression elimination
→ blade-support propagation
→ canonical integer polynomial by basis blade
→ deterministic mutations and prime-matrix expansion
→ exact modular host/CUDA evaluator
→ CPU prefix verification
→ GPU assignment search
→ host witness reproduction
→ reduced witness or exact zero-polynomial certificate
```

## Distinguishing properties

- mathematical grade, dimension, and signature remain explicit through compilation;
- coefficients are exact modulo a declared prime during search;
- generated CPU and GPU evaluators share one semantic source;
- false statements return a reproducible assignment, blade, and coefficient mismatch;
- support masks permit backend compilers to remove impossible blade operations;
- every v1 expression can be expanded into an exact integer polynomial in the declared scalar coefficients;
- a zero blade-wise difference polynomial proves the identity for the declared dimension, signature, and grade supports;
- primitive polynomial hashes cluster scalar-equivalent statement differences;
- automatic mutations retain source, edit, expression path, and stable identifier provenance;
- modular witnesses are greedily reduced to smaller human-readable coefficient assignments.

## Physically validated v1 boundary

On the RTX 5070 Laptop GPU, five statements were evaluated over 5,242,880 exact assignments. Three known identities survived the configured sample set, two false controls yielded exact assignment-zero witnesses, host reproduction succeeded, and the evidence validator passed.

## V2 implementation boundary

The multi-prime discovery layer is implemented with:

- four default prime fields;
- deterministic one-edit mutation generation;
- exact polynomial certificates;
- multi-prime corpus manifests;
- CPU prechecks;
- manifest-driven generated evaluator compilation;
- CUDA discovery orchestration;
- exact witness reduction;
- cross-prime aggregation and evidence packaging.

The complete default v2 physical matrix remains to be executed.

## Current limits

- dimensions at most six;
- non-degenerate diagonal signatures;
- scalar coefficient variables are commutative;
- mutation generation is one edit from a checked source statement;
- polynomial expansion is protected by a configurable term limit;
- witness reduction is greedy rather than globally minimal;
- no automatic dimension or signature quantification;
- no proof-assistant export yet.

## Primary files

- `tools/geo_identity_compiler.py`
- `tools/geo_identity_discovery.py`
- `tools/geo_identity_manifest_compiler.py`
- `experiments/geometric_identity_engine/`
- `experiments/geometric_identity_engine_v2/`
- `benchmarks/geo_identity_search/`
- `tests/test_geo_identity_compiler.py`
- `tests/test_geo_identity_discovery.py`
