---
id: component:geometric-identity-engine
title: "Geometric Identity and Counterexample Engine"
type: component
status: experimental-validated-v2-v3-implemented
created: 2026-07-17
updated: 2026-07-18
confidence: verified-v2-v3-unvalidated-on-gpu
---

# Geometric Identity and Counterexample Engine

The engine compiles structured Clifford-algebra statements into exact host and CUDA evaluators. V1 physically validated exact finite-field counterexample search. V2 physically validated symbolic polynomial classification, deterministic candidate mutation, four-prime corpus expansion, and exact witness reduction. V3 adds a bounded grammar front end that generates candidate expressions and exact equivalence classes without requiring every equation to be supplied manually.

## Pipeline

```text
checked identity or grammar
→ validated typed expression domain
→ deterministic canonical expression enumeration
→ common-subexpression and syntax deduplication
→ blade-support propagation
→ exact integer polynomial by basis blade
→ exact and primitive polynomial equivalence classes
→ relation ranking and scoped certificates
→ near-miss controls and prime-matrix expansion
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
- every supported expression can be expanded into an exact integer polynomial in the declared scalar coefficients;
- a zero blade-wise difference polynomial establishes the identity for the declared dimension, signature, and grade supports;
- exact polynomial hashes identify identical expression semantics;
- primitive polynomial hashes identify integer scalar multiples modulo content and sign;
- automatic mutations retain source, edit, expression path, and stable identifier provenance;
- modular witnesses are greedily reduced to smaller human-readable coefficient assignments;
- bounded grammars can generate, classify, and rank expression relations independently of a supplied equation.

## Physically validated v1 boundary

On the RTX 5070 Laptop GPU, five statements were evaluated over 5,242,880 exact assignments. Three known identities survived the configured sample set, two false controls yielded exact assignment-zero witnesses, host reproduction succeeded, and the evidence validator passed.

## Physically validated v2 boundary

The full v2 matrix executed on the same GPU:

- four exact prime fields;
- 27 unique structural variants;
- 108 generated finite-field statements;
- 28,311,552 total CUDA assignments;
- 0.045645 seconds summed kernel time;
- 620,253,207.501 aggregate assignments per second;
- base and discovery validators both passed;
- exact GPU witnesses reproduced on the host;
- reduced witnesses remained valid;
- evidence finalization completed.

Evidence: `knowledge/wiki/experiments/geometric-identity-discovery-20260717.md`.

## V3 implementation boundary

The grammar front end currently includes:

- cost-bounded expression enumeration;
- canonical syntax and commutative-addition normalization;
- exact and primitive semantic classes;
- deterministic relation ranking;
- recovery of integer-multiple identities;
- exact relation certificates;
- near-miss control generation;
- v1-compatible finite-field corpus emission;
- vector-product, general-reversion, and Jacobi grammars;
- host tests and a Windows CUDA evidence runner.

Representative baseline classes include vector commutator/wedge decomposition, reversion of products, commutator antisymmetry, and the cyclic Jacobi zero relation. These validate the discovery mechanism and are not presented as new mathematical results.

## Current limits

- dimensions at most six;
- non-degenerate diagonal signatures;
- scalar coefficient variables are commutative;
- grammar enumeration is bounded rather than exhaustive without limits;
- only the current v1 expression operations are available;
- relation ranking is heuristic;
- no quotienting by variable renaming or general symmetry groups yet;
- polynomial expansion is protected by a configurable term limit;
- witness reduction is greedy rather than globally minimal;
- no automatic dimension or signature quantification;
- no proof-assistant export yet.

## Primary files

- `tools/geo_identity_compiler.py`
- `tools/geo_identity_discovery.py`
- `tools/geo_identity_manifest_compiler.py`
- `tools/geo_identity_grammar_discovery.py`
- `experiments/geometric_identity_engine/`
- `experiments/geometric_identity_engine_v2/`
- `experiments/geometric_identity_engine_v3/`
- `benchmarks/geo_identity_search/`
- `tests/test_geo_identity_compiler.py`
- `tests/test_geo_identity_discovery.py`
- `tests/test_geo_identity_grammar_discovery.py`
