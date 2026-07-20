---
id: experiment:geometric-identity-discovery-20260717
title: "Geometric identity discovery CUDA validation — 2026-07-17"
status: completed_with_limitations
created: 2026-07-17
updated: 2026-07-17
confidence: verified
related_entities:
  - component:geometric-identity-engine
  - task:task-20260717-008
related_decisions:
  - decision:exact-modular-search-before-symbolic-proof
  - decision:canonical-polynomial-before-sampling
---

# Geometric identity discovery CUDA validation — 2026-07-17

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- compute capability: 12.0
- CUDA compiler/runtime generation: 13.1
- CUDA assignments per generated statement: 262,144
- exact CPU prechecks per statement: 1,024
- mutation-classification checks before compilation: 4,096
- CUDA block size: 256
- exact prime fields: 65,521; 65,519; 65,497; 32,749

## Corpus

The full v2 matrix contained:

- five checked source controls;
- three known identities eligible for deterministic one-edit mutation;
- two known false controls retained unchanged;
- up to eight mutations per known identity with deterministic duplicate removal;
- 27 unique structural variants;
- 108 generated finite-field statements.

The exact polynomial layer classified:

- 16 zero-polynomial field rows representing four structural zero-polynomial variants;
- 92 nonzero-polynomial field rows.

One zero-polynomial mutation is a syntactic tautology created by removing the grade projection from the vector-square control. It is retained as a classifier control and is not presented as a new theorem.

## Physical results

- statements evaluated: 108;
- total exact modular assignments: 28,311,552;
- summed CUDA kernel time: 0.045645 seconds;
- aggregate assignment rate: 620,253,207.501 assignments per second;
- base CUDA result validator: `VALIDATION: PASS`;
- discovery manifest/result validator: `VALIDATION: PASS`;
- discovery finalizer: `GEO_IDENTITY_DISCOVERY_FINALIZE,status=complete`.

All generated rows satisfied their declared behavior:

- zero-polynomial identity rows produced no finite-field witness in the configured searches;
- nonzero-polynomial and known-false rows produced exact witnesses;
- GPU witnesses reproduced on the exact host evaluator;
- reduced witnesses remained valid;
- prime-matrix and polynomial-manifest agreement passed.

## Interpretation

This experiment physically validates the complete v2 discovery path:

```text
checked geometric statements
→ deterministic mutations
→ exact blade-wise integer polynomial classification
→ four-prime corpus expansion
→ generated host/CUDA evaluators
→ exact GPU witness search
→ host reproduction and witness reduction
→ cross-prime discovery report
```

A zero blade-wise difference polynomial is an exact symbolic identity for the declared dimension, signature, and variable grade supports. It does not automatically quantify over other dimensions, signatures, or variable domains.

A nonzero polynomial proves that the formal expressions are not universally equal in the declared structural setting. A returned finite-field witness is an exact counterexample in that field. The current corpus validates the discovery machinery; it does not claim a previously unknown theorem.

## Evidence

Local evidence directory:

`benchmarks/geo_identity_search/evidence/identity-20260717-204401`

The directory contains the CUDA output, summaries, discovery manifest, reduced-witness report, SHA-256 manifest, and finalized evidence package. Local evidence remains intentionally excluded from Git.
