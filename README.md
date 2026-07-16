# Geometric Elementary Operators

Portable C11 reference and embedded kernel for the unified geometric elementary operator architecture:

\[
\Omega((s,X),(t,Y))=(\exp(s)-\log(t),XY).
\]

The implementation is designed for microcontrollers and eventual hardware realization:

- fixed-size C11 data structures;
- no heap allocation in the kernel;
- no recursion in the execution path;
- fully unrolled `Cl(2,0)` geometric product;
- opposite-lane propagation for reversion;
- unipotent addition encoding;
- shared ordered products and Hadamard dot/wedge mixing;
- algebraic `M2(R)` routing control;
- projective scale metadata and deferred normalization;
- scalar EML lane;
- flat instruction programs;
- physically separate scalar, geometric, and unified register banks;
- compile-time constant folding, lane pruning, scale propagation, and routing elision.

## Current status

The repository now contains a complete reference reconstruction of the frozen GEB-36 basis:

- 6 targets execute as direct compiled Omega trees;
- 30 targets execute through compiler-visible enlarged-representation programs;
- all 36 are compared against the direct GEB-36 C reference API;
- the original classification remains 29 exact, 5 projective/scaled, and 2 exact with a supplied transformation.

The preserved Milestone W package did not contain the original complete per-target witness corpus. The repository therefore distinguishes imported evidence from reconstructed executable witnesses. The reconstruction catalog is stored in `artifacts/geb_witness_catalog.json`.

## Implemented layers

1. Exact `Cl(2,0)` microkernel.
2. Opposite-lane state and reversion propagation.
3. Unified scalar/geometric Omega state.
4. Flat nonrecursive Omega interpreter.
5. Unipotent addition representation.
6. Shared `ab`/`ba` products and exact/projective Hadamard mixing.
7. Deferred projective normalization and vector metric helpers.
8. Generative control algebra `Gc(X,Y)=XY-X` over `M2(R)`.
9. Complete GEB-36 reference API and manifest.
10. Witness-tree validation and lowering.
11. Reachability pruning and lane-liveness propagation.
12. Duplicate-subtree elimination and register compaction.
13. Constant folding.
14. Typed and physically banked register allocation.
15. Rational projective-scale propagation.
16. Routing-state elision.
17. Complete 36-target reconstructed execution coverage.

## Execution forms

### Omega trees

Examples include:

- `Omega(e1,e2)` for the pseudoscalar;
- `Omega(A,B)` for the geometric product;
- the opposite output of `Omega(A,B)` for the reverse product;
- `Omega(Omega(R,x),reverse(R))` for rotor action;
- `Omega(R1,R2)` for rotor composition;
- `Omega(Omega(T,x),reverse(T))` for supplied sandwich transforms.

### Enlarged-representation programs

The structured bytecode exposes the proof representation explicitly:

- terminals for constants and basis values;
- unary involutions and grade projections;
- unipotent composition for addition and subtraction;
- ordered-product and Hadamard paths for dot, wedge, commutator, and anticommutator;
- scalar extraction and projective numerators;
- deferred normalization;
- dual, rotor norm, and translation-unipotent operations.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Single-precision host build:

```sh
cmake -S . -B build-float -DGEO_USE_DOUBLE=OFF
cmake --build build-float
ctest --test-dir build-float --output-on-failure
```

## Next stage

The project now moves from closure reconstruction to embedded measurement:

1. Confirm all GCC and Clang CI builds.
2. Add deterministic benchmark programs for every operation family.
3. Measure instruction count, runtime, flash, stack, and static RAM.
4. Add ESP32-S3 and ARM Cortex-M build targets.
5. Compare direct GEB functions, unified-state execution, and banked execution.
6. Add fixed-point and RTL-oriented backends.

## Design constraints

The kernel intentionally avoids exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved where needed and lowered to sparse, typed, fixed-routing microkernels for embedded targets.
