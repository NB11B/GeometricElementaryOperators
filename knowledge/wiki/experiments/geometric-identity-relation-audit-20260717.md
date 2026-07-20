---
id: experiment:geometric-identity-relation-audit-20260717
title: "V3.1 exact relation normalization and novelty audit"
date: 2026-07-17
status: verified
related_entities:
  - component:geometric-identity-engine
  - task:TASK-20260718-010
source_evidence:
  - benchmarks/geo_identity_search/evidence/identity-20260717-231854
archive_sha256: 7FFC0A5FC20E4378FA88625519481ADE2CB69EAA0B752C91B6C82AA6028BB95F
---

# V3.1 exact relation normalization and novelty audit

## Input

The audit consumed the 23 exact relation certificates produced by the physically validated V3 grammar-discovery run in:

`benchmarks/geo_identity_search/evidence/identity-20260717-231854`

The archived evidence package has SHA-256:

`7FFC0A5FC20E4378FA88625519481ADE2CB69EAA0B752C91B6C82AA6028BB95F`

## Local validation

The combined V1 through V3.1 unit-test matrix completed successfully:

- tests: 34
- result: PASS

The relation audit completed with:

- certificates: 23
- normalized families: 11
- directly recognized known certificates: 13
- tautological or support-derived certificates: 8
- initially unclassified certificates: 2

The two initially unclassified certificates are members of one normalized family representing vector-product reversion:

`reverse(v1 * v0) = v0 * v1`

Under declared vector-grade support this is a one-step consequence of:

1. reversion anti-automorphism, `reverse(xy) = reverse(y) reverse(x)`;
2. vector reversion invariance, `reverse(v) = v`.

Therefore the family is classified as `known-derived-1`, not as a novelty candidate.

## Final interpretation

After derived-identity correction:

- known or known-derived certificates: 15
- tautological or support-derived certificates: 8
- genuinely unclassified certificates: 0
- genuinely unclassified normalized families: 0

The initial V3 corpus is a successful calibration corpus. It demonstrates exact generation, equivalence recovery, finite-field validation, witness discrimination, normalization, and family-level classification, but it does not presently contain a defensible claim of a previously unknown identity.

## Next research gate

The next corpus should add rewrite-distance classification and search operators less dominated by immediate reversion, grade-support, and additive-normalization consequences. Priority operators are contraction, duality, grade involution, and Clifford conjugation, combined with mixed-grade variables and dimension/signature sweeps.
