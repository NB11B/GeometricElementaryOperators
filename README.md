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
- compile-time Omega evaluation for constant subgraphs;
- physically separate scalar, geometric, and unified register banks;
- compile-time rational scale propagation;
- fixed control-matrix lowering to data-movement routes;
- compiled GEB witness validation against the direct reference API.

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
14. Compile-time constant folding for Omega subgraphs.
15. Typed scalar/geometric/unified register planning and memory accounting.
16. Physically banked runtime storage and typed operand references.
17. Rational projective-scale propagation across geometric Omega products.
18. Routing elision for zero, identity, sign, matrix-unit, and exchange control states.
19. Imported Milestone W validation report.
20. First executable compiled witness catalog for GEB targets 6, 16, 17, 30, 31, and 33.

## Witness compiler and optimizer

A witness tree is stored as a fixed array of topologically ordered nodes. Terminal nodes map to caller-preloaded registers. Each Omega node references only earlier nodes and is lowered to one flat `GEO_OPCODE_OMEGA` instruction.

The baseline compiler preserves every Omega node. The optimized compiler additionally:

- removes nodes unreachable from the selected root;
- propagates scalar/geometric lane requirements backward;
- merges equivalent Omega nodes with identical operands and live lanes;
- compacts the emitted instruction and register sequence;
- reports original and optimized instruction counts.

A second pass accepts terminal values and compile-time-constant flags. It evaluates any Omega instruction whose two inputs are constant, removes that instruction from the runtime program, and writes the result into the caller-owned initial register image.

## Physically banked runtime

The banked planner converts logical registers into compact typed references:

- scalar registers store only `geo_real_t`;
- geometric registers store an opposite-lane value and projective scale;
- unified registers retain the full `geo_state_t` only when both lanes are live.

The banked executor reads and writes these physical arrays directly. It does not require an all-registers-as-`geo_state_t` shadow bank. The caller supplies every bank and all planning buffers, preserving deterministic memory use.

The planner reports the exact byte requirement:

\[
B = N_s\,\mathrm{sizeof}(\texttt{geo\_real\_t})
  + N_g\,\mathrm{sizeof}(\texttt{geo\_geometric\_register\_t})
  + N_u\,\mathrm{sizeof}(\texttt{geo\_state\_t}).
\]

## Lowering passes

The scale pass propagates exact rational projective factors through geometric products. For operands with scales

\[
\lambda_A = \frac{p_A}{q_A},\qquad
\lambda_B = \frac{p_B}{q_B},
\]

the product receives

\[
\lambda_{AB}=\lambda_A\lambda_B.
\]

The routing pass recognizes the fixed control matrices `0`, `I`, `-I`, `E11`, `E12`, `E21`, `E22`, and the exchange matrix. These lower to zeroing, copying, sign inversion, projection, transfer, or exchange operations without runtime matrix multiplication.

## Compiled GEB witnesses

The repository now executes genuine Omega trees through the full pipeline and compares them against the direct GEB reference functions. The first catalog covers:

- pseudoscalar: `Omega(e1,e2)`;
- geometric product: `Omega(A,B)`;
- reverse product through the opposite output lane;
- rotor action: `Omega(Omega(R,x),reverse(R))`;
- rotor composition: `Omega(R1,R2)`;
- dilation/sandwich action: `Omega(Omega(T,x),reverse(T))`.

The imported Milestone W archive contains the unified numerical validation report but not the full per-target witness-tree corpus. That limitation is recorded explicitly in `artifacts/milestone_w_unified_operator.json`. Reconstructed executable witnesses are recorded in `artifacts/geb_witness_catalog.json`.

All compiler and runtime paths remain heap-free and nonrecursive. The JSON witness interchange definition is available at `artifacts/witness_tree_schema.json`.

## Remaining milestones

1. Reconstruct constants, involutions, and projection witnesses through typed routing terminals.
2. Reconstruct addition and subtraction through the unipotent representation.
3. Reconstruct dot, wedge, commutator, and anticommutator through ordered lanes and Hadamard mixing.
4. Reconstruct metric and projective targets through central scalar injection and normalization.
5. Complete compiled coverage for all 36 GEB targets.
6. Benchmark direct GEB operations against compiled Omega programs.
7. Add ESP32-S3 and ARM Cortex-M targets.
8. Add fixed-point and RTL-oriented backends.

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

The kernel intentionally avoids heap allocation, exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved in the reference implementation and lowered to sparse, typed, fixed-routing microkernels for embedded targets.
