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

The repository contains a complete reference reconstruction of the frozen GEB-36 basis:

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
18. Portable cycle/timer abstraction and benchmark API.
19. Static type-size and bank-memory reporting.
20. ESP-IDF component and ESP32-S3 timing adapter.
21. Configurable Q-format fixed-point arithmetic and fixed `Cl(2,0)` product.

## Build and test

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

## Host benchmarks

```sh
cmake -S . -B build-bench \
  -DGEO_BUILD_BENCHMARKS=ON \
  -DGEO_USE_DOUBLE=OFF
cmake --build build-bench --config Release
./build-bench/bench_host
```

The host benchmark currently reports:

- `geo_real_t`, `geo_cl20_t`, opposite-state, unified-state, and banked-register sizes;
- flat and banked instruction sizes;
- structured-register size;
- `Cl(2,0)` product time;
- direct rotor-action time.

The benchmark API also accepts banked and structured programs, so direct-versus-compiled comparisons can use the same timing source.

## Flash, map, and stack reporting

Generate GCC/Clang stack-usage files and a benchmark linker map:

```sh
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_USE_DOUBLE=OFF \
  -DGEO_ENABLE_STACK_USAGE=ON \
  -DGEO_ENABLE_LINK_MAP=ON
cmake --build build-report
```

This emits:

- compiler `.su` files for per-function static stack estimates;
- `build-report/bench_host.map` for section and symbol accounting;
- the normal benchmark memory report for runtime data structures.

Standard platform tools such as `size`, `nm`, and `objdump` can then be applied to `bench_host` and `libgeo_kernel.a`.

## ESP32-S3 / ESP-IDF

The `ports/esp32` directory is an ESP-IDF component. Add it to an ESP-IDF project's component search path, include `geo_esp32.h`, and call:

```c
geo_esp32_print_memory_report();
geo_esp32_run_smoke_benchmarks(100000);
```

The ESP32 adapter uses `esp_timer_get_time()` as a 1 MHz monotonic timing source and reports microseconds per operation through `ESP_LOGI`.

## Fixed-point backend

`include/geo/fixed.h` provides a configurable signed 32-bit Q-format backend. The default is Q16.16:

```c
#define GEO_FIXED_FRACTION_BITS 16
```

The current fixed-point layer includes:

- conversion to and from `double`;
- checked multiply and divide;
- overflow and divide-by-zero status;
- a complete fixed-point `Cl(2,0)` geometric product;
- basis-product tests for `e1*e2=e12` and `e2*e1=-e12`.

## Performance phase

The next measurement expansion is workload-level rather than architectural:

1. Add benchmark fixtures for all major GEB operation families.
2. Compare direct functions, unified Omega bytecode, structured bytecode, and banked execution on identical inputs.
3. Record flash, static RAM, stack, cycles, and normalization counts on ESP32-S3.
4. Add ARM Cortex-M DWT cycle-counter support.
5. Quantify float versus Q-format error and throughput.
6. Generate RTL-oriented operation schedules from the optimized instruction stream.

## Design constraints

The kernel intentionally avoids exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved where needed and lowered to sparse, typed, fixed-routing microkernels for embedded targets.
