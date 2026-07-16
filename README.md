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
- projective scale metadata and deferred normalization;
- optional scalar EML lane;
- flat instruction programs for deterministic execution;
- desktop verification and embedded backends.

## Initial milestones

1. Exact `Cl(2,0)` microkernel.
2. Opposite-lane state and reversion propagation.
3. Projective scale tracking.
4. GEB-36 reference operations.
5. Flat Omega-program interpreter.
6. ESP32-S3 and ARM Cortex-M benchmarks.
7. Fixed-point and RTL-oriented backends.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Design constraints

The kernel intentionally avoids heap allocation, exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation will be preserved in the reference implementation and lowered to sparse, fixed-routing microkernels for embedded targets.

## Repository status

The repository is in the initial kernel-construction phase. The first implementation target is a verified, fully unrolled `Cl(2,0)` product with involutions and grade projection.
