---
id: experiment:geometric-identity-engine-20260717
title: "Geometric identity engine CUDA validation — 2026-07-17"
status: completed_with_limitations
created: 2026-07-17
updated: 2026-07-17
confidence: verified
related_entities:
  - component:geometric-identity-engine
  - task:task-20260717-007
---

# Geometric identity engine CUDA validation — 2026-07-17

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- compute capability: 12.0
- CUDA runtime: 13.1
- CUDA driver API version reported by the executable: 13.1
- exact coefficient domain: arithmetic modulo 65,521
- assignments per statement: 1,048,576
- CPU prechecks per statement: 4,096
- CUDA block size: 256

## Corpus

Known identities:

- `vector_square_is_scalar`
- `reverse_reverses_product_order`
- `vector_wedge_is_antisymmetric`

Deliberately false mutations:

- `vector_product_is_not_commutative`, which encodes the false claim that vector geometric products commute
- `reverse_preserves_product_order_false`

## Results

| Statement | Expected | Kernel microseconds | Assignments per second | Exact witness | Result |
|---|---|---:|---:|---|:---:|
| vector_square_is_scalar | identity | 1,204.352051 | 870,655,718.417 | none | pass |
| reverse_reverses_product_order | identity | 4,440.959961 | 236,114,716.013 | none | pass |
| vector_wedge_is_antisymmetric | identity | 279.359985 | 3,753,493,896.703 | none | pass |
| vector_product_is_not_commutative | counterexample | 431.488007 | 2,430,139,387.378 | assignment 0, blade 3, -6 != 6 | pass |
| reverse_preserves_product_order_false | counterexample | 1,262.495972 | 830,557,897.626 | assignment 0, blade 2, -9 != 9 | pass |

Aggregate:

- statements evaluated: 5
- total exact modular assignments: 5,242,880
- summed CUDA kernel time: 0.007619 seconds
- aggregate assignment rate: 688,163,373.765 assignments per second
- host smoke test: pass
- CUDA search completion marker: pass
- result validator: `VALIDATION: PASS`
- evidence manifest: created
- evidence ZIP and ZIP hash: created

## Interpretation

The experiment physically validates the complete v1 path from checked Clifford identity IR through generated exact host/CUDA evaluators to parallel finite-field counterexample search and CPU-reproduced witnesses.

The two false mutations produced exact modular witnesses, which is sufficient to reject those finite-field statements. The three known identities survived the configured assignment set, but that survival is not a proof over characteristic zero or over all assignments in the finite field.

The result validates the compiler and search machinery, not new mathematical theorems. Its significance is that structured geometric statements can now be compiled into exact GPU-scale experiments with reproducible witness extraction.

## Evidence

Local evidence directory:

`benchmarks/geo_identity_search/evidence/identity-20260717-194050`

The local evidence directory is intentionally ignored by Git. The manifest, manifest hash, archive, and archive hash are preserved in the local evidence package.
