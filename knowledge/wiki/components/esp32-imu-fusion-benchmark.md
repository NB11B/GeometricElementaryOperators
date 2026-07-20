# ESP32-C6 GEO IMU Fusion Benchmark

Status: active; clean-branch reproducibility run pending.

## Purpose

Compare generic and sparse-fused GEO orientation schedules with conventional quaternion and quantized TinyML baselines under one deterministic task contract.

## Implementations

- A0: generic floating paired-`Cl(2,0)` composition.
- B0: generic checked Q16 paired-`Cl(2,0)` composition.
- A1: sparse fused floating schedule lowered from the A0 expression.
- B1: sparse fused Q16 schedule lowered from the B0 expression.
- C: hand-specialized conventional quaternion filter.
- D: deterministic quantized neural inference fixture.

## Contract

Every path receives the same 12,000-sample, 200 Hz replay, analytic reference, 256-sample warmup, 30 measured runs, 5 ms deadline, output schema, timer, compiler optimization level, and post-run scoring.

The timed region contains only one implementation step. Replay generation, reference generation, hashing, sorting, heap inspection, and serial printing are excluded.

## Representation

For quaternion `q = w + xi + yj + zk`:

```text
A = w + y e12
B = x e1 + z e2
q -> A + iB
```

Quaternion multiplication is represented as:

```text
(A + iB)(C + iD) = (AC - BD) + i(AD + BC)
```

A0/B0 evaluate complete generic products. A1/B1 retain only operations surviving the known sparse grades and fuse product accumulation.

## Validated result

The predecessor physical run showed:

- A1 mean latency: 52.513244 us
- C mean latency: 52.563425 us
- A1 reduction from A0: 71.71%
- B1 reduction from B0: 70.89%
- zero deadline misses and NaNs
- stable hashes for all six paths
- identical A1/C displayed accuracy and output hash

The supported claim is floating GEO parity with the conventional path. A sub-percent speed lead is not claimed.

## Next compiler milestone

Replace the manually frozen A1/B1 schedules with generated schedules produced by sparsity analysis, zero elimination, fused accumulation, fixed-point range analysis, and deterministic C emission.

## Provenance

- `benchmarks/esp32_imu_baseline/`
- `benchmarks/esp32_imu_baseline/results/esp32c6-fusion-20260717.md`
- `knowledge/wiki/experiments/esp32-imu-fusion-20260717.md`
- `knowledge/wiki/tasks/TASK-20260717-004.md`
