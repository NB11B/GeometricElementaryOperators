# ESP32-C6 GEO A/B/C/D IMU replay benchmark

Status: implemented; ESP32-C6 build and physical replay evidence remain pending.

## Purpose

This component compares four local orientation-estimation paths on one deterministic 60-second, 200 Hz six-axis IMU replay:

- `A_geo_float`
- `B_geo_fixed_q16`
- `C_conventional_quaternion`
- `D_quantized_tinyml`

All paths implement the same `benchmark_impl_t` interface and are scored against the same analytic reference quaternion. Replay generation, reference generation, percentile sorting, hashing, heap inspection, and serial output are outside the timed step.

## GEO quaternion embedding

The current portable GEO kernel exposes `Cl(2,0)`, while the task requires a full 3D quaternion. The A and B paths use the standard complex 2 x 2 matrix representation of a quaternion and represent each real 2 x 2 matrix with one `Cl(2,0)` value.

For `q = w + xi + yj + zk`:

```text
q -> A + iB
A = w + y e12
B = x e1 + z e2
```

Quaternion multiplication is evaluated as:

```text
(A + iB)(C + iD) = (AC - BD) + i(AD + BC)
```

Each quaternion product therefore uses four GEO geometric products. Quaternion-vector rotation uses two such products. This preserves the full quaternion task without introducing a separate scalar quaternion-product implementation in A or B.

## A: floating GEO

`A_geo_float` uses floating `geo_cl20_mul`, `geo_cl20_add`, and `geo_cl20_sub` for quaternion multiplication and gravity-frame rotation. The correction policy, gains, confidence gate, and integration step match the conventional C path.

## B: fixed GEO

`B_geo_fixed_q16` stores quaternion and integral state in signed Q16 values. It uses checked fixed `Cl(2,0)` products and checked fixed correction/integration arithmetic. Arithmetic failure is converted to a non-finite result so the common benchmark records it through `nan_count`.

Quaternion normalization is the explicit nonlinear boundary: the Q16 quaternion is decoded, normalized with a scalar square root, and requantized. Results must therefore be described as a fixed GEO algebra/integration path with a decoded normalization boundary, not as an exclusively integer pipeline.

## Measurement contract

Each implementation receives 256 untimed warm-up samples followed by 30 measured runs of 12,000 samples. Result rows include state bytes, latency distribution, 5 ms deadline misses, quaternion error distribution, non-finite count, deterministic named-field hash, and ESP-IDF heap observations.

A startup check runs the A, B, and C paths on the same deterministic fixture for 512 steps. It requires finite results, zero B arithmetic failures, A/C agreement within 0.01 degrees, and B/C agreement within 1 degree.

## Validation boundary

The branch includes an ESP-IDF 5.3.4 ESP32-C6 compile workflow, but the pull request remains draft. A compile-only result will establish integration and toolchain compatibility, not physical timing. Hardware claims require a named ESP32-C6 board, exact commit, ESP-IDF version, clock configuration, full serial capture, and preserved CSV rows.

## Provenance

- `benchmarks/esp32_imu_baseline/main/geo_filter.c`
- `benchmarks/esp32_imu_baseline/main/geo_filter.h`
- `benchmarks/esp32_imu_baseline/main/geo_filter_self_test.c`
- `benchmarks/esp32_imu_baseline/main/benchmark_common.c`
- `benchmarks/esp32_imu_baseline/main/main.c`
- `benchmarks/esp32_imu_baseline/README.md`
- `.github/workflows/esp32-imu-benchmark.yml`
- `knowledge/wiki/tasks/TASK-20260717-002.md`
