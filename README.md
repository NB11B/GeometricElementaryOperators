# Geometric Elementary Operators

Geometric Elementary Operators (GEO) is a portable, geometry-native execution system that compiles geometric relationships into explicit routing, sign, and transformation plans.

The project began as an embedded C11 kernel for preserving mathematical meaning under microcontroller constraints. It now includes exact operator specialization, native gradients, a dynamically sized training runtime, CPU and CUDA execution, independent validation, benchmarking, profiling, and reproducible evidence packaging.

## Core idea

For basis blades, the geometric product can be represented as

\[
e_i e_j = s_{ij} e_{i \oplus j},
\]

where `i XOR j` selects the output blade and `s_ij` captures permutation parity and metric signature.

GEO builds this relationship once as an executable plan rather than repeatedly rediscovering it through generic dense tensor machinery. The same mathematical plan can then be lowered to embedded C, optimized CPU code, CUDA kernels, or future hardware schedules.

The original unified operator architecture remains:

\[
\Omega((s,X),(t,Y))=(\exp(s)-\log(t),XY).
\]

## Current status

The accepted development line now includes:

- portable C11 embedded and host kernels;
- exact `Cl(2,0)` and dimensions 1 through 6;
- all canonical non-degenerate diagonal signatures in the supported dimension range;
- fixed-blade and sparse fixed-multivector specialization;
- stable allocation-free operator APIs;
- independently verified operator certificates;
- native JVP and VJP primitives;
- GEO-owned reverse-mode differentiation;
- native SGD and Adam;
- dynamically sized arbitrary DAG execution;
- minibatch accumulation, parameter constraints, recurrence, checkpointing, prediction, and C export;
- CPU and CUDA full-cycle geometric-product learning workloads;
- independent hand-written CUDA control kernels;
- nine-trial PyTorch eager and `torch.compile` comparisons;
- Nsight Systems and Nsight Compute capture automation;
- hashed acceptance archives and manifest verification.

## Accepted CPU benchmark

On the recorded optimized CPU benchmark, geometric-mean throughput ratios were:

| Comparison | Inference | Training |
|---|---:|---:|
| Optimized GEO vs baseline GEO | 8.32x | 9.10x |
| Optimized GEO vs PyTorch eager | 71.77x | 59.55x |

These results are workload- and environment-specific. Geometric means are primary; peak cases are descriptive only.

## Accepted CUDA benchmark

Physical validation was completed on an NVIDIA GeForce RTX 5070 Laptop GPU using float64, dimensions 2 through 6, and batches 1, 16, 64, 256, and 1024.

| Comparator | Mode | Resident | Transfer + compute | End-to-end |
|---|---|---:|---:|---:|
| PyTorch eager CUDA | Inference | 8.63x | 3.96x | 2.52x |
| PyTorch eager CUDA | Training | 1.88x | 1.73x | 1.46x |
| PyTorch compile CUDA | Inference | 9.46x | 4.32x | 2.75x |
| PyTorch compile CUDA | Training | 1.75x | 1.86x | 1.46x |

The independent `hand_cuda_same_plan` control used separate native CUDA kernels for forward, VJP, loss, and SGD while retaining the same mathematical routing/sign plan. It produced only modest inference gains over PyTorch and was slower for training. This control shows that native CUDA alone did not reproduce the optimized GEO result.

The defensible conclusion is limited to the tested hardware, software stack, precision, dimensions, batches, modes, and timing classes.

## CUDA acceptance evidence

The accepted CUDA package includes:

- physical correctness across all declared signatures, both multiplication sides, and batches 1/16/64/256;
- reference, planned GEO, and independent hand-written CUDA paths;
- nine measured trials for GEO, PyTorch eager, and PyTorch compile;
- per-trial CSV, aggregate CSV, and raw JSON;
- stable numerical checksums;
- four Nsight Systems reports;
- four Nsight Compute reports;
- generated Markdown and CSV acceptance reports;
- a verified SHA-256 manifest.

Final archive record:

```text
MANIFEST_VERIFICATION: PASS files=85
GEO_GPU_EVIDENCE_ARCHIVE: PASS
SHA256=79AE425C841846B5BD6E248C43BE502B0B6861E17C983242A182F97CA594B740
Bytes=196671292
```

## Build and test

### Host C11 build

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

### Host benchmarks

```sh
cmake -S . -B build-bench \
  -DGEO_BUILD_BENCHMARKS=ON \
  -DGEO_USE_DOUBLE=OFF
cmake --build build-bench --config Release
./build-bench/bench_host
```

### CUDA V8 build on Windows

From a PowerShell session with `nvcc` available:

```powershell
cmake -S .\benchmarks\geo_v8_cuda `
    -B .\build\geo-v8-cuda `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build .\build\geo-v8-cuda `
    --config Release `
    --parallel

ctest `
    --test-dir .\build\geo-v8-cuda `
    -C Release `
    --output-on-failure
```

Expected CUDA correctness marker:

```text
GEO_V8_CUDA_CORRECTNESS: PASS
```

See `benchmarks/geo_v8_cuda/README.md` for the full benchmark, profiler, report, and packaging workflow.

## Embedded targets

The kernel retains its embedded design constraints:

- fixed-size C11 data structures where required;
- no heap allocation in embedded execution paths;
- no recursion;
- flat instruction programs;
- typed and physically banked registers;
- configurable Q-format fixed-point arithmetic;
- ESP-IDF integration and ESP32 timing support;
- compiler-visible lowering, pruning, scale propagation, and routing elision.

The `ports/esp32` directory is an ESP-IDF component. Typical entry points are:

```c
geo_esp32_print_memory_report();
geo_esp32_run_smoke_benchmarks(100000);
```

## Repository map

- `include/geo/` — public C and CUDA interfaces;
- `src/` — host, gradient, runtime, and CUDA implementations;
- `tests/` — host and physical CUDA correctness suites;
- `benchmarks/` — CPU, CUDA, PyTorch, profiler, reporting, and packaging tools;
- `tools/` — standalone GEO model, training, prediction, and export utilities;
- `ports/` — embedded platform integrations;
- `experiments/` — accepted milestone records and scientific boundaries;
- `artifacts/` — reconstructed witness and supporting catalogs.

## Scientific claim boundary

GEO has demonstrated that geometry-aware execution can provide a measurable and repeatable advantage on the tested CPU and GPU geometric-product workloads. The evidence does not establish universal superiority over tensor methods, all model architectures, all precisions, all devices, or unrelated workloads.

Current follow-on research includes:

- structured replacement and compression of trained neural-network layers;
- mixed-precision and tensor-core implementations;
- NPU, FPGA, ASIC, and analog mappings;
- larger grammar-driven models;
- workload-level embedded validation;
- direct integration into application models.

## Design principle

GEO does not treat geometry as decorative metadata around dense arithmetic. It preserves geometric relationships as executable structure and lowers that structure into sparse, typed, fixed-routing computation suitable for the target hardware.