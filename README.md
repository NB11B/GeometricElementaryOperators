# Geometric Elementary Operators

Portable C11 reference, embedded, CUDA, and generated RTL kernels for the unified geometric elementary operator architecture:

\[
\Omega((s,X),(t,Y))=(\exp(s)-\log(t),XY).
\]

The kernel is designed for microcontrollers, GPUs, and hardware realization:

- fixed-size C11 data structures;
- no heap allocation in the portable execution kernel;
- no recursion in execution paths;
- fully unrolled `Cl(2,0)` products;
- opposite-lane propagation for reversion;
- unipotent addition and shared ordered-product representations;
- algebraic `M2(R)` routing control;
- projective scale metadata with deferred normalization;
- flat instruction programs and physically separate register banks;
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
10. Witness-tree validation, optimization, folding, and lowering.
11. Typed and physically banked register allocation.
12. Rational projective-scale propagation and routing-state elision.
13. Portable cycle/timer abstraction and benchmark API.
14. ESP-IDF component and ESP32-S3 timing adapter.
15. Configurable signed 32-bit Q-format arithmetic.
16. Fixed scalar/geometric Omega, opposite-lane, banked, program, and `M2(R)` control execution.
17. Complete flat fixed-point GEB-36 execution with checked projective normalization.
18. Optional CUDA 13.x batched execution backend with a stable C ABI.
19. Generated CUDA schedule kernels compiled from checked schedule JSON.
20. Shared workload timing and error reports across direct, specialized, structured, fused, fixed, banked, and optional CUDA paths.
21. Deterministic fixed-versus-floating numerical envelopes with overflow and projective-scale reporting.
22. Generated fixed-point RTL product, controller, and executable schedule datapaths with explicit overflow signaling.
23. Standalone ESP32-S3 correctness, benchmark, and heap-stability soak application.
24. Vendor-neutral ARM Cortex-M DWT cycle source for the portable benchmark API.

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

The public executor symbols `geo_struct_program_execute` and `geo_banked_execute` are emitted out of line for ABI compatibility. The `_impl` entry points remain available to existing source consumers. The caller-owned `geo_optimized_witness_t` layout retains the established ABI size and field offsets.

## Workload benchmarks and numerical reports

The shared CPU harness uses deterministic fixtures for every supported backend and reports correctness together with timing:

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

Generate repeated-run workload reports with minimum, P50, P95, P99, mean, maximum, standard deviation, absolute error, relative error, and mismatch counts:

```sh
cmake --build build-bench --target workload_report
```

Generate the fixed-versus-floating numerical envelope:

```sh
./build-bench/bench_numerical \
  --samples 10000 \
  --seed 1779033703 \
  --csv numerical.csv
cmake --build build-bench --target numerical_report
```

The numerical report freezes all 36 target names and their expected typed result
kinds. Every bounded deterministic sample must complete with the expected kind;
missing, extra, duplicate, overflowing, or mismatching rows are rejected. The
floating reference receives the values decoded from the quantized fixtures, and
the report records componentwise, angular, and projective-scale error. This is a
typed floating-reference envelope over quantized fixtures, not an independently
implemented oracle. Reports are written under `build-bench/workload-report/`
and `build-bench/numerical-report/`.

Focused benchmark executables remain available under their established names:

```sh
./build-bench/bench_host
./build-bench/bench_compare
./build-bench/bench_complete
```

`bench_complete` describes itself as the selected-path comparison benchmark and prints the exact operation/path matrix it measured. Its JSON and Markdown report carry the same matrix.

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

The handwritten batched CUDA backend supports:

- `Cl(2,0)` addition;
- geometric product;
- reverse;
- vector dot;
- vector wedge;
- rotor action.

The build also generates and compiles CUDA schedule kernels for addition, geometric product, dot, wedge, and rotor action:

```sh
cmake --build build-cuda --target generate_cuda_schedules
```

`cuda_equivalence` tests the public batched C ABI. `cuda_schedule_equivalence` executes generated launchers with the same operation semantics. Both return CTest skip code 77 when no compatible CUDA device or driver is available, while the CUDA 13 compile job still builds every source.

Run the CUDA selected-path harness:

```sh
./build-cuda/bench_cuda \
  --device 0 \
  --batch 262144 \
  --iterations 100 \
  --warmup 10 \
  --seed 608135816 \
  --operation all \
  --csv cuda-benchmark.csv
```

The harness validates batch byte counts and grid dimensions before allocation and uses one deterministic seed for every compared path. It reports six public batched C APIs with `backend=cuda_public_api` and `timing_scope=host_end_to_end`; those host wall-clock measurements include the allocation, copies, launch, synchronization, and release performed by the API. It separately reports the five available generated launchers (addition, geometric product, dot, wedge, and rotor action) with `backend=cuda_generated_schedule` and `timing_scope=device_kernel`, measured by CUDA events on device-resident buffers. There is intentionally no generated reverse schedule.

The harness header reports GPU model, compute capability, runtime, and driver.
Every CSV row includes:

- operation, backend/path, timing scope, precision, batch, iterations, warmup, and seed;
- per-item timing for that row's declared scope;
- overflow-checked upload, download, and logical-kernel byte counts;
- maximum absolute and relative error;
- mismatch count.

Byte accounting follows each real interface: public dot and wedge download scalar coefficient arrays, while their generated schedules download full `Cl(2,0)` values; generated rotor action uploads `R`, `x`, and `reverse(R)`, while the public API accepts only `R` and `x`. The workload report rejects missing, extra, duplicate, mislabeled, mis-seeded, or incorrectly byte-counted rows against its exact CPU/CUDA capability manifest.

Canonical operation names in CSV output are `addition`, `geometric_product`, `reverse`, `vector_dot`, `vector_wedge`, and `rotor_action`. The shorter CLI aliases `add`, `product`, `dot`, `wedge`, and `rotor` remain accepted.

Validate memory access and races on a CUDA-capable host:

```sh
compute-sanitizer --tool memcheck ./build-cuda/test_cuda
compute-sanitizer --tool racecheck ./build-cuda/test_cuda
compute-sanitizer --tool memcheck ./build-cuda/test_cuda_schedules
compute-sanitizer --tool racecheck ./build-cuda/test_cuda_schedules
```

Profile with current NVIDIA tools:

```sh
nsys profile --stats=true --output=geo_cuda ./build-cuda/bench_cuda
ncu --set full --target-processes all ./build-cuda/bench_cuda
```

A CUDA compile job is not a GPU execution result. Published GPU claims must record the GPU model, compute capability, driver, exact CUDA 13.x update, and Compute Sanitizer result.

## Fixed-point backends

`include/geo/fixed.h` provides configurable signed 32-bit Q-format arithmetic. The default is Q16.16:

```c
#define GEO_FIXED_FRACTION_BITS 16
```

Supported fractional-bit counts are 1 through 30. The fixed layer includes checked conversion, multiplication, division, involutions, vector products, rotor action, all frozen GEB-36 targets, and these execution surfaces:

- `include/geo/fixed_program.h`: typed flat GEB-36 programs;
- `include/geo/fixed_omega.h`: scalar/geometric Omega state and opposite lanes;
- `include/geo/fixed_banked.h`: physically banked fixed execution;
- `include/geo/fixed_control.h`: checked `M2(R)` routing control.

The fixed executors perform no allocation or recursion, propagate overflow and divide-by-zero status, reject invalid register, type, and scale inputs, and leave destination values unchanged on failure.

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

It runs deterministic floating-point and fixed-point checks, reports memory and timing records, and continuously verifies that allocation-free execution does not change free heap or the largest free block.

## ARM Cortex-M DWT timing

`include/geo/cortex_m_dwt.h` adapts the ARM Data Watchpoint and Trace cycle counter to `geo_cycle_source_t` without depending on a vendor SDK. The caller supplies register addresses and the core clock.

```c
geo_cortex_m_dwt_registers_t registers = {
    &CoreDebug->DEMCR,
    &DWT->CTRL,
    &DWT->CYCCNT,
    CoreDebug_DEMCR_TRCENA_Msk,
    DWT_CTRL_CYCCNTENA_Msk
};
geo_cortex_m_dwt_context_t context;
geo_cycle_source_t source;

geo_cortex_m_dwt_status_t status = geo_cortex_m_dwt_start(
    &context,
    &registers,
    SystemCoreClock,
    &source
);
```

The adapter extends the 32-bit hardware counter to 64 bits in software. Call it more frequently than one complete hardware wrap interval; at 480 MHz that interval is approximately 8.95 seconds.

## Generated RTL and equivalence

Generate the fixed product cell and serialized controller:

```sh
cmake --build build --target generate_rtl
```

Generate executable schedule datapaths plus matching checked-C harnesses:

```sh
cmake --build build --target generate_rtl_schedules
```

The equivalent direct generator command is:

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

CI compiles and runs generated checked-C harnesses, simulates each SystemVerilog datapath with the same nominal and overflow vectors, and synthesizes every generated top. Fixed C and RTL share the same round-half-away-from-zero, rounded-product overflow, final-wide-sum overflow, and instruction-order contract. Cancellation cases accepted by fixed C remain valid in RTL.

## Flash, map, and stack reporting

```sh
cmake -S . -B build-report \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEO_USE_DOUBLE=OFF \
  -DGEO_ENABLE_STACK_USAGE=ON \
  -DGEO_ENABLE_LINK_MAP=ON
cmake --build build-report --parallel
```

This emits compiler `.su` files for per-function static stack estimates and linker maps for the benchmark executables. Standard tools such as `size`, `nm`, and `objdump` can then be applied to the executables and `libgeo_kernel.a`.

## Release and compatibility

Release, ABI, fixed/RTL numerical, benchmark-labeling, generated-artifact, and hardware-evidence requirements are defined in `docs/RELEASE_POLICY.md`. Notable changes are recorded in `CHANGELOG.md`.

Full hosted validation is intentionally gated on a pull request being marked ready for review; draft synchronization runs are skipped and superseded runs are cancelled. A release remains untagged until the exact candidate commit has passing required validation. An empty, skipped, or unstarted hosted workflow is not treated as a pass.

## Design constraints

The portable and embedded kernel intentionally avoids exceptions, RTTI, virtual dispatch, recursive execution, and generic dense matrix arithmetic. The proof-level product-space representation is preserved where needed and lowered to sparse, typed, fixed-routing microkernels for embedded, GPU, and RTL targets.
