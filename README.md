# Geometric Elementary Operators

Geometric Elementary Operators (GEO) is a portable, geometry-native execution system that compiles geometric relationships into explicit routing, sign, transformation, and hardware-execution plans.

The project began as an embedded C11 kernel for preserving mathematical meaning under microcontroller constraints. It now includes exact operator specialization, native gradients, dynamically sized training execution, CPU and CUDA backends, geometry-aware backend dispatch, independent validation, benchmarking, profiling, and reproducible evidence packaging.

## Core idea

For basis blades, the geometric product can be represented as

\[
e_i e_j = s_{ij} e_{i \oplus j},
\]

where `i XOR j` selects the output blade and `s_ij` captures permutation parity and metric signature.

GEO builds this relationship once as an executable plan rather than repeatedly rediscovering it through generic dense tensor machinery. The same mathematical plan can then be lowered to embedded C, optimized CPU code, CUDA kernels, vendor numerical libraries, or future hardware schedules.

The original unified operator architecture remains:

\[
\Omega((s,X),(t,Y))=(\exp(s)-\log(t),XY).
\]

## Current status

The accepted development line includes:

- portable C11 embedded and host kernels;
- exact `Cl(2,0)` and dimensions 1 through 6;
- all canonical non-degenerate diagonal signatures in the supported dimension range;
- fixed-blade and sparse fixed-multivector specialization;
- stable allocation-free operator APIs;
- independently verified operator certificates;
- native JVP and VJP primitives;
- GEO-owned reverse-mode differentiation;
- native SGD, Adam, and fused AdamW primitives;
- dynamically sized arbitrary DAG execution;
- minibatch accumulation, parameter constraints, recurrence, checkpointing, prediction, and C export;
- CPU and CUDA full-cycle learning workloads;
- independent hand-written CUDA control kernels;
- geometry-aware dispatch between GEO kernels and explicitly reported vendor backends;
- full-matrix, recompute, and end-to-end tile-streamed causal attention;
- parallel cross entropy and fused global-gradient clipping;
- Nsight Systems and Nsight Compute capture automation;
- hashed acceptance archives and manifest verification.

## Deep learning execution stack

GEO now serves as the numerical substrate for a three-repository training stack:

```text
GeometricElementaryOperators
    mathematical operators, native CPU/CUDA kernels, execution plans, JVP/VJP

Geo-Deep-Learning-Runtim
    tensor bridge, autograd integration, adaptive dispatcher,
    attention, loss, embedding, clipping, optimizer

GEOSDP
    model, tokenizer, corpus, checkpointing, training, evaluation, export
```

The runtime preserves a clear distinction between mathematical ownership and execution backend selection. A wide dense projection may deliberately use cuBLAS; a smaller projection, reduction, geometric operator, optimizer, or streaming-attention tile may use a GEO-native kernel. The selected implementation is exposed through auditable dispatcher telemetry.

This is an execution-planning architecture, not a claim that every operation should be reimplemented independently of mature vendor libraries.

## Adaptive computational dispatch

Current dispatch policies include:

| Operation class | Accepted policy |
|---|---|
| Linear projection | GEO vectorized CUDA for smaller outputs; cuBLAS for wide projections |
| Causal attention | saved-probability, probability-recompute, and tile-streamed modes |
| Cross entropy | serial small-vocabulary and parallel block-reduction paths |
| Optimizer | fused multi-tensor GeoAdamW and fused global-gradient norm |

The runtime records the actual selected backend, selection reason, algorithm identifier, workspace size, full-matrix status, and fallback status.

## End-to-end streaming attention

The accepted branch includes tile-streamed causal attention in both forward and backward execution.

Streaming backward reconstructs score and probability tiles from saved row statistics and accumulates `dQ`, `dK`, and `dV` without materializing a full global `B × H × T × T` score or probability matrix.

For fixed tile sizes, retained attention state scales as:

```text
O(B × H × T × D) + bounded tile workspace
```

Accepted isolated measurements for a fixed batch/head configuration showed:

| Sequence length | Streaming peak allocated memory |
|---:|---:|
| 256 | 9 MB |
| 512 | 18 MB |
| 1,024 | 36 MB |
| 2,048 | 72 MB |

The observed gradient differences versus the accepted full-matrix path remained in the expected FP32 reduction-order range, approximately `1e-6` to `1e-5` for the tested cases.

These results establish correctness and linear sequence-memory scaling for the tested configuration. Streaming latency optimization remains active work.

## Accepted CPU benchmark

On the recorded optimized CPU benchmark, geometric-mean throughput ratios were:

| Comparison | Inference | Training |
|---|---:|---:|
| Optimized GEO vs baseline GEO | 8.32x | 9.10x |
| Optimized GEO vs PyTorch eager | 71.77x | 59.55x |

These results are workload- and environment-specific. Geometric means are primary; peak cases are descriptive only.

## Accepted geometric-product CUDA benchmark

Physical validation was completed on an NVIDIA GeForce RTX 5070 Laptop GPU using float64, dimensions 2 through 6, and batches 1, 16, 64, 256, and 1024.

| Comparator | Mode | Resident | Transfer + compute | End-to-end |
|---|---|---:|---:|---:|
| PyTorch eager CUDA | Inference | 8.63x | 3.96x | 2.52x |
| PyTorch eager CUDA | Training | 1.88x | 1.73x | 1.46x |
| PyTorch compile CUDA | Inference | 9.46x | 4.32x | 2.75x |
| PyTorch compile CUDA | Training | 1.75x | 1.86x | 1.46x |

The independent `hand_cuda_same_plan` control used separate native CUDA kernels for forward, VJP, loss, and SGD while retaining the same mathematical routing/sign plan. It produced only modest inference gains over PyTorch and was slower for training. This control shows that native CUDA alone did not reproduce the optimized GEO result.

The defensible conclusion is limited to the tested hardware, software stack, precision, dimensions, batches, modes, and timing classes.

## Accepted 32K training-system benchmark

For the recorded GEOSDP configuration (`V=32,768`, `B=2`, `T=64`, `D=64`), targeted kernel and dispatch work reduced the measured training step from `12.337 ms` to `4.918 ms`:

| Phase | Initial | Accepted optimized result |
|---|---:|---:|
| Forward | 4.547 ms | 1.642 ms |
| Backward | 3.456 ms | 2.923 ms |
| Optimizer | 4.335 ms | 0.354 ms |
| Total | 12.337 ms | 4.918 ms |

The improvement came from several independent changes rather than one monolithic kernel: fused clipping, fused AdamW, parallel cross entropy, and wide-projection cuBLAS dispatch.

## CUDA acceptance evidence

The geometric-product CUDA package includes:

- physical correctness across declared signatures, multiplication sides, and accepted batches;
- reference, planned GEO, and independent hand-written CUDA paths;
- repeated trials for GEO, PyTorch eager, and PyTorch compile;
- per-trial CSV, aggregate CSV, and raw JSON;
- stable numerical checksums;
- Nsight Systems and Nsight Compute reports;
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

See `benchmarks/geo_v8_cuda/README.md` for the complete geometric-product benchmark, profiler, report, and packaging workflow.

## Current deep learning acceptance gate

The accepted three-repository gate reports:

- **GEO Host C++ Tests:** 36/36 passed
- **Runtime Tests:** 59/59 passed
- **CUDA Parity Tests:** 37/37 passed
- **GEOSDP Subsystem Tests:** 98/98 passed
- **End-to-End Native Training:** 3/3 passed

The two-step training smoke is a finite-execution and parameter-update check, not a convergence benchmark.

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
- `src/` — host, gradient, runtime, optimizer, attention, loss, and CUDA implementations;
- `tests/` — host and physical CUDA correctness suites;
- `benchmarks/` — CPU, CUDA, PyTorch, profiler, reporting, and packaging tools;
- `tools/` — standalone GEO model, training, prediction, and export utilities;
- `ports/` — embedded platform integrations;
- `experiments/` — accepted milestone records and scientific boundaries;
- `artifacts/` — reconstructed witness and supporting catalogs.

## Scientific claim boundary

GEO has demonstrated that geometry-aware execution can provide measurable and repeatable advantages on tested CPU and GPU workloads, and that its operators can support a complete small-model CUDA training stack with auditable backend selection.

The evidence does not establish universal superiority over tensor methods, vendor libraries, all model architectures, all precisions, all devices, or unrelated workloads. Results should be reported with their exact shape, backend, timing boundary, hardware, software stack, and numerical contract.

Current follow-on research includes:

- streaming-attention latency optimization;
- mixed precision and tensor-core execution;
- structured replacement and compression of trained neural-network layers;
- NPU, FPGA, ASIC, and analog mappings;
- larger grammar-driven models;
- workload-level embedded validation;
- direct integration into application models.

## Design principle

GEO does not treat geometry as decorative metadata around dense arithmetic. It preserves geometric relationships as executable structure and lowers that structure into sparse, typed, fixed-routing computation suitable for the target hardware.

Where an operation's geometry is better served by a specialized existing engine, GEO can select that engine explicitly rather than obscure the execution boundary.