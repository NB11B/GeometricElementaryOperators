# GEO V8 CPU Benchmark Contract

## Status

This contract governs the CPU comparison in PR #32. It is part of the benchmark evidence and must be revised explicitly when the workload, comparator, timing method, or claim boundary changes.

## Scientific claim

The benchmark measures throughput for one matched float64 geometric-product learning workload on a CPU. It compares:

1. baseline GEO V8 dynamic-graph execution;
2. optimized GEO planned batch execution;
3. PyTorch eager tensor execution and autograd.

It does **not** establish universal superiority over PyTorch, compiled tensor frameworks, custom native kernels, GPUs, tensor cores, or unrelated model architectures.

## Locked workload semantics

- coefficient domain: IEEE-754 binary64;
- dimensions: 2 through 6;
- signatures: positive Euclidean signature for the timed matrix unless a report states otherwise;
- batch sizes: 1, 16, and 64;
- product: right geometric product, `prediction = input * parameter`;
- blade routing: XOR;
- Clifford sign table: identical mathematical rule in every backend;
- loss: one-half squared coefficient residual, reduced by batch mean for training;
- optimizer: SGD with the same learning rate and one update per timed training iteration;
- complexity: matched dense quadratic blade-pair work;
- input and target generation: deterministic and reproducible;
- inference and training checksums: recorded to prevent dead-code elimination and detect mismatched work.

Any semantic difference invalidates direct speedup claims until the case is reclassified.

## Correctness gate before timing

Every timed implementation must first pass:

- forward output agreement against an independent coefficient reference;
- parameter-gradient agreement within a declared float64 tolerance;
- one-step SGD update agreement;
- identical batch reduction semantics;
- finite output, loss, gradient, and parameter checks.

Timing must not proceed after a failed correctness gate.

## CPU execution controls

The benchmark records and, where supported, fixes:

- one compute thread;
- PyTorch intra-op and inter-op thread counts;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `OPENBLAS_NUM_THREADS=1`;
- `VECLIB_MAXIMUM_THREADS=1`;
- `NUMEXPR_NUM_THREADS=1`;
- CPU affinity using `taskset` when available;
- CPU model, logical cores, physical cores, cache topology, and reported governor;
- operating system and kernel;
- compiler path and full compiler version;
- complete CMake configure command;
- effective optimization and linker flags;
- Python and PyTorch versions;
- git branch and exact commit SHA;
- workflow run identifier.

GitHub-hosted runners are shared systems. Their measurements are useful engineering evidence, not a substitute for dedicated bare-metal replication.

## Timing protocol

Primary publication runs should use:

- at least 20 untimed warm-up iterations for each case;
- at least 9 independent measured trials per case;
- identical iteration counts within each matched case;
- a monotonic high-resolution timer;
- timing around the matched compute region only;
- separate reporting of setup/compilation/startup costs where relevant;
- no I/O inside the timed region;
- no model construction inside the timed region;
- no table generation inside the timed region.

For every case the artifact should report:

- median throughput;
- minimum and maximum throughput;
- standard deviation;
- median absolute deviation or interquartile range;
- raw trial values.

## Summary statistics

The primary aggregate metric is the geometric mean of per-case throughput ratios.

Required aggregate results:

- optimized GEO / baseline GEO inference geometric mean;
- optimized GEO / baseline GEO training geometric mean;
- optimized GEO / PyTorch eager inference geometric mean;
- optimized GEO / PyTorch eager training geometric mean.

Peak speedups are secondary observations. They may be reported only alongside the geometric mean, minimum ratio, and complete per-case matrix.

## Comparator labels

Results must use exact comparator names:

- `PyTorch eager`;
- `torch.compile` with backend and mode;
- `PyTorch C++/ATen`;
- `hand-written C++ planned GP`;
- `JAX/XLA` with platform and compilation mode;
- `oneDNN/custom CPU kernel` with implementation details.

A result against one comparator must not be generalized to another.

## Required artifact contents

- raw CSV for every backend;
- raw per-trial timing data;
- comparison CSV and JSON;
- Markdown summary;
- benchmark contract version;
- environment record;
- compiler and CMake commands;
- effective flags;
- executable sizes;
- git status and exact source SHA;
- checksums or hashes for the evidence archive;
- correctness-gate output.

## Defensible publication language

> On the recorded single-threaded CPU environment, the optimized GEO planned-batch kernel achieved the reported geometric-mean throughput ratios relative to the matched PyTorch eager implementation. These results apply to the tested float64 geometric-product workload, dimensions, batch sizes, software versions, and timing protocol. They do not establish superiority over compiled frameworks, custom native kernels, GPU implementations, or unrelated model architectures.
