# GEO V8 GPU Pull, Run, and Test Plan

## Purpose

This plan extends the CPU benchmark into a physically executed GPU study without weakening the scientific boundary. It defines how to pull the accepted source, validate the machine, build GPU comparators, run matched correctness gates, benchmark resident kernels and end-to-end execution separately, and preserve the full evidence chain.

The GPU study is a separate milestone from the CPU benchmark. CPU ratios must not be described as GPU ratios, and GPU results must not be inferred before physical execution.

## Planned GPU comparators

The staged comparison ladder is:

1. GEO CUDA reference kernel;
2. GEO CUDA planned-batch kernel with precomputed routing and signs;
3. PyTorch eager CUDA;
4. `torch.compile` CUDA with backend and mode recorded;
5. hand-written CUDA using the same routing/sign plan;
6. optional JAX/XLA CUDA after the first five are accepted.

The hand-written CUDA comparator is essential. It separates the value of GEO's architecture and planning from the generic benefit of replacing a framework path with native specialized code.

## Supported first physical target

Initial acceptance should run on the known NVIDIA system when available:

- NVIDIA GeForce RTX 5070 Laptop GPU;
- CUDA compiler and driver versions recorded at runtime;
- compute capability recorded from the device;
- power mode, clocks, and thermal state recorded where permissions permit.

The workflow must remain portable to another NVIDIA GPU, but results are hardware-specific and must carry the exact device identity.

## Pull and source verification

Use a clean directory and pull the exact benchmark head or accepted merge commit:

```bash
git clone https://github.com/NB11B/GeometricElementaryOperators.git
cd GeometricElementaryOperators
git fetch --all --tags --prune
git checkout bench/geo-v8-vs-pytorch-cpu
git reset --hard <RECORDED_HEAD_SHA>
git status --short
git rev-parse HEAD
```

Before execution, record:

```bash
git remote -v
git branch --show-current
git rev-parse HEAD
git submodule status --recursive || true
```

The evidence report must identify the exact source SHA. A dirty working tree invalidates an acceptance run.

## Machine preflight

Run and preserve:

```bash
uname -a
lscpu
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.limit,clocks.sm,clocks.mem,memory.total --format=csv
nvcc --version
cmake --version
ninja --version || true
gcc --version
python --version
```

Python environment:

```bash
python - <<'PY'
import platform
import torch
print('python', platform.python_version())
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('torch_cuda', torch.version.cuda)
if torch.cuda.is_available():
    print('device_count', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

Fail immediately if CUDA is unavailable or if the requested device cannot be selected.

## Build configuration

Configure a release build with explicit compilers and recorded flags:

```bash
cmake -S . -B build/gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DGEO_BUILD_TESTS=ON \
  -DGEO_BUILD_BENCHMARKS=ON \
  -DGEO_BUILD_TOOLS=ON \
  -DGEO_USE_DOUBLE=ON

cmake --build build/gpu --parallel
```

The GPU implementation milestone should add dedicated targets such as:

```text
bench_v8_cuda_reference
bench_v8_cuda_planned_batch
test_v8_cuda_correctness
```

The exact configure command, compiler versions, generated cache, and effective flags must be archived.

## Correctness acceptance before timing

For every dimension, signature, side, and batch shape in the timed matrix:

1. generate deterministic inputs, parameters, targets, and output cotangents;
2. compute an independent CPU coefficient reference;
3. compare GEO CUDA reference forward output;
4. compare GEO CUDA optimized forward output;
5. compare PyTorch CUDA forward output;
6. compare parameter gradients;
7. compare one SGD update;
8. reject NaN, infinity, stale state, or unsupported shapes;
9. record maximum absolute and relative error.

The first correctness matrix should include:

- dimensions 2 through 6;
- all canonical non-degenerate signatures;
- left and right multiplication;
- batch sizes 1, 16, 64, and 256;
- dense multivectors;
- selected sparse multivectors;
- inference and training-step paths.

Float64 tolerance must be declared in the artifact. Exact modular CUDA checks should remain as an additional control where applicable.

## Timing separation

The GPU report must publish at least three timing classes.

### Resident kernel

Inputs, parameters, outputs, gradients, and optimizer state are already resident on the GPU. Measure with CUDA events and synchronize only at trial boundaries.

This isolates kernel execution and is the primary measure of GPU compute efficiency.

### Host-to-device plus compute

Include input and target transfer but keep model state resident.

This represents streaming inference or training where each batch arrives from the host.

### End-to-end invocation

Include orchestration, allocation policy, transfer, compute, synchronization, and result retrieval as defined by the benchmark command.

Do not compare resident GEO timing against end-to-end PyTorch timing.

## Warm-up and repeated trials

For every case:

- initialize CUDA context before timing;
- run enough warm-up iterations to stabilize kernels, caches, and compiled graphs;
- compile `torch.compile` paths before measured trials;
- use at least 9 measured trials;
- report median, minimum, maximum, standard deviation, and raw values;
- verify the device did not thermally throttle during the run where telemetry is available.

Use CUDA events for kernel timing. Wall-clock timing is appropriate for end-to-end measurements.

## Threading, affinity, and process controls

Record CPU affinity and thread-related variables because host orchestration can affect GPU results:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Pin the benchmark process where supported and record whether pinning succeeded.

## GPU workload matrix

Initial matrix:

| Axis | Values |
|---|---|
| Dimension | 2, 3, 4, 5, 6 |
| Batch | 1, 16, 64, 256, 1024 |
| Mode | inference, train step |
| Precision | float64 first; float32 as a separate report |
| Side | right first; left in correctness matrix and follow-up timing |
| Timing | resident, transfer+compute, end-to-end |
| Comparator | GEO reference, GEO optimized, PyTorch eager, torch.compile, hand CUDA |

Float32, mixed precision, tensor-core-oriented reformulations, and sparse kernels must be separate studies rather than silently combined with the float64 result.

## Evidence artifact

The GPU artifact should contain:

- exact source SHA and clean-tree record;
- environment and GPU telemetry;
- compiler and build commands;
- CMake cache and effective flags;
- correctness matrices and tolerances;
- raw timing trials;
- summary CSV, JSON, and Markdown;
- resident and end-to-end reports kept separate;
- executable and shared-library sizes;
- profiler captures for selected crossover cases;
- SHA-256 manifest and archive hash.

Recommended profiler evidence:

- Nsight Systems for orchestration and transfer;
- Nsight Compute for occupancy, memory traffic, instruction mix, and achieved throughput;
- selected cases at dimension 2 batch 1, dimension 6 batch 64, and dimension 6 batch 1024.

## Acceptance boundaries

A successful GPU run may support statements of the form:

> On the recorded NVIDIA GPU, software stack, precision, workload matrix, and timing class, the optimized GEO CUDA path achieved the reported geometric-mean ratio relative to the named comparator.

It must not support:

- universal GPU superiority;
- claims about untested GPUs;
- claims about tensor-core performance from float64 tests;
- substitution of peak ratios for geometric means;
- mixing resident-kernel and end-to-end measurements;
- generalization from PyTorch eager to `torch.compile`, JAX/XLA, or hand CUDA.

## Implementation sequence

1. merge the CPU methodology and optimized batch kernel after review;
2. add CUDA forward reference and optimized planned-batch kernels;
3. add CUDA parameter VJP and fused MSE/SGD path;
4. complete the physical correctness matrix;
5. add PyTorch eager CUDA comparator;
6. add `torch.compile` CUDA comparator;
7. add hand-written CUDA comparator;
8. run resident and end-to-end studies;
9. capture profiler evidence;
10. publish the acceptance record only after all required artifacts are complete.
