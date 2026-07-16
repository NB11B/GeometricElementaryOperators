# Geometric Elementary Operators

Portable C11 reference, embedded, CUDA, and generated RTL kernels for the unified geometric elementary operator architecture:

\[
\Omega((s,X),(t,Y))=(\exp(s)-\log(t),XY).
\]

The portable kernel is designed for microcontrollers and hardware realization:

- fixed-size C11 data structures;
- no heap allocation in the kernel;
- no recursion in execution paths;
- fully unrolled `Cl(2,0)` geometric products;
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
21. Configurable signed 32-bit Q-format arithmetic.
22. Complete flat fixed-point GEB-36 program executor with checked projective normalization.
23. Optional CUDA 13.x batched execution backend with a stable C ABI.
24. Shared workload timing/error reports across direct, specialized, structured, fused, fixed, banked, and optional CUDA paths.
25. Generated fixed-point RTL product, controller, and executable schedule datapaths with explicit overflow signaling.
26. Standalone ESP32-S3 correctness, benchmark, and heap-stability soak application.

## Host build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Single-precision host build:

```sh
cmake -S . -B build-float \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_USE_DOUBLE=OFF
cmake --build build-float --parallel
ctest --test-dir build-float --output-on-failure
```

The public executor symbols `geo_struct_program_execute` and `geo_banked_execute` are emitted out of line for ABI compatibility. The `_impl` entry points remain available to existing source consumers.

## Workload benchmarks and error reports

The shared CPU harness uses identical deterministic fixtures for every supported backend and reports correctness together with timing:

```sh
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_BUILD_BENCHMARKS=ON \
  -DGEO_USE_DOUBLE=OFF
cmake --build build-bench --parallel
./build-bench/bench_workloads \
  --iterations 100000 \
  --warmup 1000 \
  --seed 608135816 \
  --csv workload.csv
```

Generate repeated-run CSV, JSON, and Markdown reports with minimum, P50, P95, P99, mean, maximum, standard deviation, absolute error, relative error, and mismatch counts:

```sh
cmake --build build-bench --target workload_report
```

Reports are written to `build-bench/workload-report/`. CUDA rows are added automatically when the build includes `GEO_BUILD_CUDA=ON` and a CUDA device is available.

The older focused host and selected-backend executables remain available:

```sh
./build-bench/bench_host
./build-bench/bench_compare
./build-bench/bench_selected_backends
```

## CUDA 13.x

CUDA is optional and disabled by default, so portable C11 and ESP32 builds do not require the CUDA toolkit. CUDA builds require CMake 3.24 or newer and CUDA Toolkit 13.x.

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_BUILD_CUDA=ON \
  -DGEO_BUILD_TESTS=ON \
  -DGEO_BUILD_BENCHMARKS=ON \
  -DGEO_USE_DOUBLE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

The initial batched CUDA operations are:

- `Cl(2,0)` addition;
- geometric product;
- reverse;
- vector dot;
- vector wedge;
- rotor action.

Run the CUDA event harness:

```sh
./build-cuda/bench_cuda \
  --device 0 \
  --batch 262144 \
  --iterations 100 \
  --warmup 10 \
  --operation all \
  --csv cuda-benchmark.csv
```

The harness reports device properties, precision, batch size, upload time, kernel-only time, download time, transfer-inclusive time, throughput, and CPU/GPU error statistics.

Validate memory access and races on a CUDA-capable host:

```sh
compute-sanitizer --tool memcheck ./build-cuda/test_cuda
compute-sanitizer --tool racecheck ./build-cuda/test_cuda
```

Profile with current NVIDIA tools:

```sh
nsys profile --stats=true --output=geo_cuda ./build-cuda/bench_cuda
ncu --set full --target-processes all ./build-cuda/bench_cuda
```

## Fixed-point program backend

`include/geo/fixed.h` provides configurable signed 32-bit Q-format arithmetic. The default is Q16.16:

```c
#define GEO_FIXED_FRACTION_BITS 16
```

Supported fractional-bit counts are 1 through 30. The fixed layer includes checked conversion, multiplication, division, involutions, vector products, rotor action, all frozen GEB-36 targets, and a flat program executor in `include/geo/fixed_program.h`.

The fixed program executor:

- performs no allocation or recursion;
- leaves a destination register unchanged when an instruction fails;
- propagates overflow and divide-by-zero status;
- rejects invalid targets, register references, types, and zero projective scales;
- normalizes projective values only when a following operation consumes them;
- promotes scalar results to scalar multivectors when required by a following GEB operation.

## ESP32-S3 / ESP-IDF

The `ports/esp32` directory is an ESP-IDF component. Add `ports` to an ESP-IDF component search path, include `geo_esp32.h`, and call:

```c
geo_esp32_print_memory_report();
geo_esp32_run_smoke_benchmarks(100000);
```

The adapter uses `esp_timer_get_time()` as a 1 MHz monotonic source and reports microseconds per operation through `ESP_LOGI`.

A complete physical-device validation application is in `examples/esp32_s3`:

```sh
cd examples/esp32_s3
idf.py set-target esp32s3
idf.py build
idf.py -p PORT flash monitor
```

It runs deterministic float and fixed checks, reports memory and timing records, and continuously verifies that allocation-free execution does not change free heap or the largest free block.

## Generated RTL and equivalence

Generate the fixed product cell and controller:

```sh
python3 tools/generate_rtl.py \
  --out-dir build-rtl \
  --schedule-json rtl/examples/rotor_action_schedule.json \
  --self-test
```

Generate executable schedule datapaths plus matching checked-C harnesses:

```sh
python3 tools/generate_rtl_schedule.py \
  --out-dir build-rtl \
  --self-test \
  --schedule-json rtl/examples/addition_schedule.json \
  --schedule-json rtl/examples/geometric_product_schedule.json \
  --schedule-json rtl/examples/vector_dot_schedule.json \
  --schedule-json rtl/examples/vector_wedge_schedule.json \
  --schedule-json rtl/examples/rotor_action_schedule.json
```

CI compiles and runs the generated C harnesses, simulates each SystemVerilog datapath with the same nominal and overflow vectors, and synthesizes every generated top. Fixed C and RTL share the same round-half-away-from-zero, overflow, and instruction-order contract.

## Flash, map, and stack reporting

Generate GCC/Clang stack-usage files and benchmark linker maps:

```sh
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_USE_DOUBLE=OFF \
  -DGEO_ENABLE_STACK_USAGE=ON \
  -DGEO_ENABLE_LINK_MAP=ON
cmake --build build-report --parallel
```

This emits compiler `.su` files for per-function static stack estimates and linker maps for the benchmark executables. Standard tools such as `size`, `nm`, and `objdump` can then be applied to the executables and `libgeo_kernel.a`.

## Design constraints

The portable and embedded kernel intentionally avoids exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved where needed and lowered to sparse, typed, fixed-routing microkernels for embedded, GPU, and RTL targets.
