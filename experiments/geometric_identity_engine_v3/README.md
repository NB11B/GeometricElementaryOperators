# Geometric Identity Discovery Engine v3

V3 changes the input to the exact identity engine from a human-supplied equation to a bounded mathematical grammar.

```text
checked grammar
→ deterministic expression enumeration
→ exact blade-wise integer polynomial for every expression
→ exact and scalar-multiple equivalence classes
→ ranked candidate relations
→ scoped symbolic certificates
→ automatic near-miss controls
→ exact prime-field corpus
→ generated host/CUDA evaluators
→ exact witness search and reduction
```

## What is generated

A grammar declares:

- Clifford dimension;
- diagonal metric signature;
- variables and permitted grades;
- unary and binary operators;
- integer scale factors and grade projections;
- maximum expression cost and corpus size;
- semantic-representative limits;
- relation-ranking limits;
- optional family seeds for important but higher-cost expression shapes.

The enumerator constructs canonical expression trees up to the configured cost. Syntactic duplicates are removed before exact classification. Each retained expression is expanded over the integers as

\[
E=\sum_B P_B(x_1,\ldots,x_n)e_B,
\]

where `B` is a basis blade and each `P_B` is a polynomial in the independent scalar coefficients of the declared multivector variables.

## Equivalence classes

Two hashes are recorded:

- **exact polynomial hash:** expressions have identical blade-wise integer polynomials;
- **primitive polynomial hash:** the coefficient content and overall sign are removed, so integer scalar multiples share a class.

The second form allows the engine to discover relations such as

\[
[a,b]=2(a\wedge b)
\]

for vector variables without having to generate both sides with matching literal scale factors first.

Selected relations are recompiled as ordinary v1 identity specifications. The compiler independently extracts the polynomial of `lhs-rhs` and requires it to be zero before the relation is admitted to the CUDA corpus.

## Initial grammars

### Vector-product relations

Two vector variables in `Cl(4,0)` with geometric product, wedge, commutator, reversion, grade projection, addition, subtraction, and integer scaling.

Baseline exact classes include:

- commutator and twice the wedge product;
- product-order and reversion relations specific to vectors;
- commutator antisymmetry;
- scalar and bivector grade decompositions of the vector product.

### General reversion relations

Two unrestricted multivectors in signature `(2,2)`. The grammar exercises:

- reversion as an involution;
- reversal of geometric-product order;
- commutator antisymmetry;
- additive and product interactions.

### Commutator/Jacobi relations

Three unrestricted multivectors in `Cl(3,0)`. A bounded nested-commutator family includes the cyclic Jacobi expression

\[
[a,[b,c]]+[b,[c,a]]+[c,[a,b]],
\]

which the exact polynomial layer classifies as zero in the declared setting.

These baseline relations validate the discovery machinery. They are not presented as previously unknown theorems.

## Near-miss controls

Every selected relation is eligible for one deterministic one-edit mutation using the v2 mutation engine. A control is admitted only when:

1. its integer difference polynomial is nonzero;
2. a deterministic exact finite-field precheck finds a witness;
3. the generated v1 specification passes validation.

The controls ensure the generated CUDA evaluator can distinguish a true zero-polynomial relation from a nearby false statement.

## Determinism and boundedness

Enumeration is deterministic because:

- expression JSON is canonicalized;
- candidate keys are sorted before admission;
- commutative addition operands are ordered canonically;
- exact and primitive polynomial classes use SHA-256 over canonical payloads;
- relation identifiers are derived from grammar name, polynomial class, scales, and expression keys.

The search is deliberately bounded by expression cost, candidate count, representatives per semantic class, polynomial term count, selected relations, and control count.

## Host gate

```powershell
python -m unittest `
    tests.test_geo_identity_compiler `
    tests.test_geo_identity_discovery `
    tests.test_identity_result_summarizer `
    tests.test_geo_identity_grammar_discovery
```

A standalone corpus can be generated with:

```powershell
python .\tools\geo_identity_grammar_discovery.py build-corpus `
    --grammar .\experiments\geometric_identity_engine_v3\grammars\01_vector_product_relations.json `
    --grammar .\experiments\geometric_identity_engine_v3\grammars\02_general_reversion_relations.json `
    --grammar .\experiments\geometric_identity_engine_v3\grammars\03_commutator_jacobi.json `
    --output-dir .\local-evidence\grammar-discovery-host `
    --prime 65521 `
    --prime 65519 `
    --precheck-assignments 1024 `
    --max-relations 8 `
    --max-controls 2 `
    --clean
```

## Physical CUDA gates

Smoke:

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_grammar_discovery.ps1 `
    -Assignments 131072 `
    -CpuChecks 512 `
    -Primes @(65521,65519) `
    -PrecheckAssignments 1024 `
    -MaxRelations 8 `
    -MaxControls 2 `
    -Archive
```

Full gate:

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_grammar_discovery.ps1 `
    -Assignments 262144 `
    -CpuChecks 1024 `
    -Primes @(65521,65519,65497,32749) `
    -PrecheckAssignments 2048 `
    -MaxRelations 12 `
    -MaxControls 4 `
    -Archive
```

The runner generates the grammar corpus, runs all host tests, compiles it through the existing manifest compiler, executes the exact CUDA search, reproduces and reduces witnesses on the host, and seals the evidence package.

## Interpretation boundary

A zero blade-wise integer difference polynomial is an exact identity for the declared dimension, signature, variable grades, and scalar-coefficient model. It does not automatically quantify over other dimensions, signatures, variable domains, or noncommuting coefficient rings.

A nonzero polynomial proves that two formal expressions are not universally equal in that declared structural setting. A finite-field witness is exact in its stated field. The grammar rank is a discovery heuristic, not a mathematical measure of importance.
