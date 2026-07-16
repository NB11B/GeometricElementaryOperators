# Geometric Elementary Operators

Portable C11 reference and embedded kernel for the unified geometric elementary operator architecture.

The project implements the computational architecture developed in *Elementary Geometry from a Unified Product-Space Operator*:

\[
\Omega((s,X),(t,Y)) = (\exp(s)-\log(t), XY).
\]

The implementation is designed for microcontrollers and eventual hardware realization:

- portable C11 core;
- fixed-size data structures;
- no dynamic allocation;
- no recursion in the execution path;
- fully unrolled `Cl(2,0)` geometric product;
- explicit opposite-lane propagation for reversion;
- unipotent addition encoding;
- shared ordered products and Hadamard dot/wedge mixing;
- algebraic `M2(R)` routing control;
- projective scale metadata and deferred normalization;
- optional scalar EML lane;
- flat instruction programs for deterministic execution;
- explicit GEB-36 reference API and closure manifest;
- fixed-size witness-tree validation and compilation;
- backward lane-liveness and duplicate-subtree elimination;
- desktop verification and embedded backends.

## Implemented kernel layers

1. Exact `Cl(2,0)` microkernel.
2. Opposite-lane state and reversion propagation.
3. Unified scalar/geometric Omega state.
4. Flat, nonrecursive Omega-program interpreter.
5. Unipotent addition representation.
6. Shared `ab`/`ba` products and exact/projective Hadamard mixing.
7. Deferred projective normalization and vector metric helpers.
8. Generative control algebra `Gc(X,Y)=XY-X` over `M2(R)`.
9. Complete 36-target GEB reference API.
10. Machine-readable GEB-36 closure manifest.
11. Topologically ordered witness-tree representation.
12. Iterative witness validation, register allocation, and lowering to Omega bytecode.
13. Reachability pruning, backward lane-liveness, duplicate-node merging, and compact register allocation.

## Witness compiler and optimizer

A witness tree is stored as a fixed array of topologically ordered nodes. Terminal nodes map to caller-preloaded registers. Each Omega node references only earlier nodes and is lowered to one flat `GEO_OPCODE_OMEGA` instruction.

The baseline compiler preserves every Omega node. The optimized compiler additionally:

- removes nodes unreachable from the selected root;
- propagates scalar/geometric lane requirements backward;
- merges equivalent Omega nodes with identical operands and live lanes;
- compacts the emitted instruction and register sequence;
- reports original and optimized instruction counts.

Both paths require caller-owned buffers and perform no heap allocation or recursive traversal. The JSON interchange definition is available at `artifacts/witness_tree_schema.json`.

## Remaining milestones

1. Import the verified witness-tree artifacts produced during operator discovery.
2. Attach terminal type, routing, and expected-scale metadata to each imported tree.
3. Add constant folding, scale propagation, routing elision, and typed register allocation.
4. Reproduce each GEB-36 target through compiled Omega programs and compare against the direct reference API.
5. Benchmark direct GEB operations against compiled Omega programs.
6. Add ESP32-S3 and ARM Cortex-M targets.
7. Add fixed-point and RTL-oriented backends.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Use single precision for the microcontroller-oriented host build:

```sh
cmake -S . -B build-float -DGEO_USE_DOUBLE=OFF
cmake --build build-float
ctest --test-dir build-float --output-on-failure
```

## GEB-36 classification

The reference manifest preserves the paper's frozen classification:

- 29 exact targets;
- 5 projective/scaled targets;
- 2 exact targets requiring a supplied transformation element.

The C manifest is available through `geo_geb36_manifest()`. The corresponding machine-readable artifact is `artifacts/geb36_manifest.json`.

## Design constraints

The kernel intentionally avoids heap allocation, exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved in the reference implementation and will be lowered to sparse, fixed-routing microkernels for embedded targets.
