# GPU IMU Replay Benchmark

Status: G0 resident-replay validation complete; G1 fixed-point path next.

This benchmark carries the ESP32-C6 IMU replay contract onto CUDA without pretending that one sequential 200 Hz sensor stream is naturally a GPU workload.

The temporal recurrence remains sequential within each trajectory. GPU parallelism is introduced across independent trajectories, agents, sensors, particles, or simulation members. The benchmark therefore reports both single-stream latency controls and batched throughput.

## Required implementations

The final comparison preserves the six CPU/ESP32 paths:

- `A0_geo_float_generic_gpu`
- `B0_geo_fixed_q16_generic_gpu`
- `A1_geo_float_generated_gpu`
- `B1_geo_fixed_q16_generated_gpu`
- `C_conventional_quaternion_gpu`
- `D_quantized_tinyml_gpu`

G0 implements A1 and C. G1 adds B1 because the generated schedule already emits Q32 accumulator helpers. A0/B0 and D remain controlled expansion stages rather than simulated labels.

## Replay contract

- 200 Hz logical sample rate;
- 12,000 samples per trajectory;
- identical procedural IMU trace and analytic reference trajectory;
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

Future variants include CUDA Graph replay and a persistent kernel.

### End-to-end

Pinned host upload, kernel execution, and result download are included.

This mode is required before any system-level speed claim. Kernel-only throughput must not be presented as end-to-end latency.

## Batch matrix

The validated G0 matrix is:

```text
1, 32, 256, 1024, 4096 trajectories
```

Batch 1 is reported as a latency control, not as the expected GPU optimum.

## Accuracy and hashing

A separate validation launch captures all 12,000 outputs for one trajectory. Accuracy, NaNs, and named-field hashes are computed outside the timed region.

Two hash classes are distinguished:

- exact device hash: bitwise determinism on the same CUDA build and device;
- semantic hash: quantized canonical output for cross-architecture comparison.

Exact ESP32 and GPU hashes are not assumed to match because CUDA FMA, square root, and division behavior may differ. A1 and C on the same CUDA parity build are required to match.

## Timing controls

- CUDA events provide kernel-only timing;
- warm-up runs precede measurement;
- implementation order rotates across runs;
- `--fmad=false`, precise division, and precise square root are used for parity experiments;
- a second performance build may enable FMA, but its results are reported separately;
- device name, compute capability, clocks, CUDA runtime, driver, compiler, commit, and build flags are preserved.

## GPU memory reporting

The report distinguishes:

- mutable state bytes per trajectory;
- shared model bytes;
- replay input bytes;
- output capture bytes;
- total allocated device bytes.

For TinyML, weights are shared across trajectories on GPU. They must not be falsely reported as mutable per-trajectory state.

## Physical G0 result

Environment:

- NVIDIA GeForce RTX 5070 Laptop GPU;
- compute capability 12.0;
- CUDA compiler 13.1.80;
- FMA disabled;
- 30 runs per path per batch;
- 12,000 samples per trajectory.

Validation:

- A1 and C produced trace hash `71a80f19` at every tested batch;
- their batch hashes matched at every batch;
- hashes remained stable over all 30 runs;
- displayed mean, p95, and maximum error matched;
- zero NaNs;
- validator result: `VALIDATION: PASS`.

Aggregate resident-kernel timing:

| batch | A1 mean us | C mean us | A1 vs C | A1 samples/s | C samples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 3603.513 | 3595.253 | 0.230% slower | 3.332 M | 3.339 M |
| 32 | 3594.077 | 3604.855 | 0.299% faster | 106.888 M | 106.586 M |
| 256 | 4086.034 | 4097.414 | 0.278% faster | 752.068 M | 750.047 M |
| 1024 | 4108.816 | 4091.673 | 0.419% slower | 2.992 B | 3.004 B |
| 4096 | 4111.061 | 4108.627 | 0.059% slower | 11.960 B | 11.969 B |

Interpretation:

- generated A1 reached measurement-level parity with conventional C across the complete batch matrix;
- the maximum aggregate timing separation was 0.419%;
- batch 4,096 produced about 0.08364 ns per sample-trajectory for A1;
- A1 throughput increased approximately 3,589x between batch 1 and batch 4,096 while kernel duration increased approximately 14.1%;
- this is approximately 87.6% scaling efficiency relative to ideal linear throughput scaling;
- resident kernel throughput is not an end-to-end or live-stream latency result.

Local evidence directory:

```text
benchmarks/gpu_imu_replay/evidence/gpu-20260717-184259
```

## Acceptance stages

### G0: generated float parity — complete

- A1 generated CUDA helper compiles and runs;
- C conventional CUDA path runs the same replay;
- A1 and C have equal displayed accuracy;
- A1 and C have equal exact device hashes;
- zero NaNs;
- stable hashes across 30 runs;
- measurement-level timing parity across the full batch matrix.

### G1: generated fixed path — next

- B1 uses generated Q32 accumulators and checked Q16 boundaries;
- zero arithmetic failures;
- error remains inside the ESP32 fixed-path envelope;
- stable hash across runs.

### G2: six-way comparison

- A0/B0/A1/B1/C/D all execute the same replay contract;
- resident, streaming, and end-to-end modes are reported separately;
- batch scaling curves are preserved.

### G3: generated backend equivalence

The same schedule IR emits both ESP32 scalar C and CUDA device helpers. Generated CUDA A1/B1 must remain within 1% of manually specialized CUDA reference kernels at useful batch sizes.

## Build target

The validated G0 target is CUDA 13.1 on the RTX 5070 Laptop GPU. Results remain device-specific and identify the exact GPU and software stack.
