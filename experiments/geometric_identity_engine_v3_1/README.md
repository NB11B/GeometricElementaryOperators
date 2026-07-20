# Geometric Identity Discovery Engine v3.1

V3.1 converts the raw exact relations selected by the grammar engine into a smaller, more useful mathematical catalogue.

```text
exact V3 certificates
→ expression simplification
→ first-occurrence variable renaming
→ side/sign quotienting
→ grade-support and involution reduction
→ normalized mathematical families
→ checked-family classification
→ novelty-oriented ranking
```

## Motivation

The physically validated V3 matrix selected 23 exact relations. Several were different presentations of the same mathematics, including sign-scaled equations, additive zero forms, variable-order variants, double reversion, redundant grade projection, and identities implied directly by declared single-grade support.

V3.1 does not alter the exact certificate. It adds a post-certificate audit layer that distinguishes:

- checked known identities;
- declared-support or normalization tautologies;
- unclassified exact candidates that merit further analysis.

## Normalization

The audit tool currently applies:

- deterministic variable renaming by first occurrence;
- equality-side exchange;
- simultaneous sign normalization;
- double negation removal;
- nested integer-scale collapse;
- subtraction-by-zero normalization;
- double reversion removal;
- redundant equal-grade projection removal;
- grade projection and reversion reduction for declared pure-grade variables;
- canonical ordering of additive arguments.

The family key retains dimension, metric signature, and variable-grade support. Relations from different structural settings are therefore not merged merely because their printed forms coincide.

## Checked relation classes

The initial classifier recognizes:

- Jacobi identity;
- commutator antisymmetry;
- vector commutator/wedge equivalence;
- reversion involution;
- reversion anti-automorphism;
- scalar-part product symmetry;
- scalar-part reversion invariance;
- projection idempotence;
- declared-grade support and reversion consequences;
- additive and scale normalization artifacts.

Unmatched relations are preserved as candidates rather than declared novel.

## Usage

```powershell
python .\tools\geo_identity_relation_audit.py `
    --certificates .\benchmarks\geo_identity_search\evidence\identity-20260717-231854\grammar-discovery\certificates `
    --output-json .\local-evidence\v3-1\relation-audit.json `
    --markdown-out .\local-evidence\v3-1\relation-audit.md
```

Expected terminal marker:

```text
AUDIT: PASS certificates=23 families=<normalized count>
```

## Interpretation boundary

The audit is a quotienting and ranking layer. It relies on the exact V3 zero-polynomial certificates and does not independently establish them. A relation marked `candidate` is unclassified by the current checked-family library; it is not automatically new mathematics.
