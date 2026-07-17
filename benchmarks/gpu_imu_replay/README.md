# GPU IMU Replay Benchmark

Status: design and first implementation scaffold.

This benchmark carries the ESP32-C6 IMU replay contract onto CUDA without pretending that one sequential 200 Hz sensor stream is naturally a GPU workload.

The temporal recurrence remains sequential within each trajectory. GPU parallelism is introduced across independent trajectories, agents, sensors, particles, or simulation members. The benchmark therefore reports both single-stream latency and batched throughput.

## Required implementations

The final comparison will preserve the six CPU/ESP32 paths:

- `A0_geo_float_generic_gpu`
- `B0_geo_fixed_q16_generic_gpu`
- `A1_geo_float_generated_gpu`
- `B1_geo_fixed_q16_generated_gpu`
- `C_conventional_quaternion_gpu`
- `D_quantized_tinyml_gpu`

The first executable stage implements A1 and C. B1 follows immediately because the generated schedule already emits Q32 accumulator helpers. A0/B0 and D are retained as controlled expansion stages rather than being simulated by labels.

## Replay contract

- 200 Hz logical sample rate;
- 12,000 samples per trajectory;
- identical procedural IMU trace and analytic reference trajectory;
- 256 warm-up samples for streaming mode;
- 30 measured runs;
- identical gains, confidence gate, normalization boundary, output fields, quaternion error metric, NaN accounting, and named-field hashing.

The replay trace is generated once on the host and uploaded outside the kernel-only timing scope.

## GPU execution modes

### Resident replay

One CUDA thread owns one independent filter state and loops over all 12,000 samples inside one kernel launch.

This measures maximum useful batched throughput while preserving exact temporal recurrence.

Primary metrics:

- kernel microseconds per batch;
- nanoseconds per sample-trajectory;
- trajectories per second;
- samples per second;
- state bytes per trajectory;
- deterministic hash and accuracy for a captured validation trajectory.

### Streaming step

One kernel launch advances a batch of independent states by one sample. The benchmark executes 12,000 launches.

This measures the cost of using CUDA in a real-time streaming topology and separates launch overhead from arithmetic cost.

Future variants will include CUDA Graph replay and a persistent kernel.

### End-to-end

Pinned host upload, kernel execution, and result download are included.

This mode is required before any system-level speed claim. Kernel-only throughput must not be presented as end-to-end latency.

## Batch matrix

The initial matrix is:

```text
1, 32, 256, 1024, 4096, 16384 trajectories
```

The runner may reduce the maximum batch when device memory is insufficient. Batch 1 is reported as a latency control, not as the expected GPU optimum.

## Accuracy and hashing

A separate validation launch captures all 12,000 outputs for one trajectory. Accuracy, NaNs, and named-field hashes are computed outside the timed region.

Two hash classes are distinguished:

- exact device hash: bitwise determinism on the same CUDA build and device;
- semantic hash: quantized canonical output for cross-architecture comparison.

Exact ESP32 and GPU hashes are not assumed to match because CUDA FMA, square root, and division behavior may differ. A1 and C on the same CUDA build are expected to match when operation order and compiler flags are aligned.

## Timing controls

- CUDA events provide kernel-only timing;
- a steady host clock provides end-to-end timing;
- warm-up runs precede measurement;
- implementation order is rotated across runs;
- `--fmad=false`, precise division, and precise square root are used for parity experiments;
- a second performance build may enable FMA, but its results are reported separately;
- device name, compute capability, clocks, CUDA runtime, driver, compiler, commit, and build flags are preserved.

## GPU memory reporting

The report distinguishes:

- mutable state bytes per trajectory;
- shared model bytes;
- replay input bytes;
- output capture bytes;
- total allocated device bytes;
- peak device memory delta.

For TinyML, weights are shared across trajectories on GPU. They must not be falsely reported as mutable per-trajectory state.

## Acceptance stages

### G0: generated float parity

- A1 generated CUDA helper compiles and runs;
- C conventional CUDA path runs the same replay;
- A1 and C have equal displayed accuracy;
- A1 and C have equal exact device hashes in the parity build;
- zero NaNs;
- stable hashes across 30 runs.

### G1: generated fixed path

- B1 uses generated Q32 accumulators and checked Q16 boundaries;
- zero arithmetic failures;
- error remains inside the ESP32 fixed-path envelope;
- stable hash across runs.

### G2: six-way comparison

- A0/B0/A1/B1/C/D all execute the same replay contract;
- resident, streaming, and end-to-end modes are reported separately;
- batch scaling curves are preserved.

### G3: generated backend equivalence

The same schedule IR must emit both ESP32 scalar C and CUDA device helpers. Generated CUDA A1/B1 should remain within 1% of manually specialized CUDA reference kernels at useful batch sizes.

## Build target

The initial target environment is the existing CUDA 13.1 workstation with the RTX 5070 Laptop GPU. Results remain device-specific and will identify the exact GPU and software stack.
