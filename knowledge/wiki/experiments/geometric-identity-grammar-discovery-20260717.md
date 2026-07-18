---
id: experiment:geometric-identity-grammar-discovery-20260717
title: "Geometric identity grammar discovery CUDA validation — 2026-07-17"
status: completed_with_limitations
created: 2026-07-17
updated: 2026-07-18
confidence: verified
related_entities:
  - component:geometric-identity-engine
  - task:task-20260718-009
related_decisions:
  - decision:exact-modular-search-before-symbolic-proof
  - decision:canonical-polynomial-before-sampling
---

# Geometric identity grammar discovery CUDA validation — 2026-07-17

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- compute capability: 12.0
- CUDA compiler/runtime generation: 13.1
- exact prime fields: 65,521; 65,519; 65,497; 32,749
- CUDA assignments per generated statement: 262,144
- exact CPU checks per statement: 1,024
- grammar-control prechecks: 2,048

## Generated corpus

The V3 grammar engine produced:

- 23 selected exact relations;
- 9 near-miss controls;
- 32 structural variants;
- 128 finite-field statements.

The corpus was generated from three bounded grammars covering vector products, general multivector reversion/product relations, and nested commutator/Jacobi relations.

## Physical results

- statements evaluated: 128;
- total exact modular assignments: 33,554,432;
- summed CUDA kernel time: 0.047260 seconds;
- aggregate assignment rate: 709,992,598.077 assignments per second;
- base validator: `VALIDATION: PASS`;
- grammar-discovery validator: `VALIDATION: PASS`;
- finalizer: `GEO_IDENTITY_GRAMMAR_DISCOVERY,status=complete`.

All generated statements satisfied their declared behavior:

- zero-polynomial exact relation rows produced no witness;
- nonzero near-miss controls produced exact witnesses in every configured field;
- GPU witnesses reproduced on the host evaluator;
- reduced witnesses remained valid;
- evidence manifests and archive hashes were created.

## Validated pipeline

```text
checked grammar
→ deterministic bounded expression enumeration
→ exact blade-wise integer polynomial classification
→ exact and primitive equivalence classes
→ deterministic relation selection
→ exact certificates and near-miss controls
→ four-prime host/CUDA corpus
→ exact GPU witness search
→ host reproduction and witness reduction
→ finalized evidence package
```

## Interpretation

The experiment validates the grammar-bounded discovery machinery rather than mathematical novelty. A zero polynomial establishes an exact identity only for the declared dimension, diagonal signature, variable grade supports, and commutative scalar-coefficient model. A finite-field witness is exact in its stated field.

## Evidence

Local evidence directory:

`benchmarks/geo_identity_search/evidence/identity-20260717-231854`

The directory contains the generated grammar corpus, certificates, CUDA output, summaries, reduced-witness report, SHA-256 manifest, and archive metadata.
