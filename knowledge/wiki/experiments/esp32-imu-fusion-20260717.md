# ESP32-C6 IMU fusion benchmark — 2026-07-17

Status: physical run completed; result aggregation verified from the captured 181-line CSV stream. Exact firmware commit should remain paired with the local `evidence/.../commit.txt` artifact.

## Environment

- Physical ESP32-C6, chip revision 0.2.
- ESP-IDF `v5.3.4-1025-g6f6766f917-dirty`.
- SPI flash mode DIO at 80 MHz.
- Deterministic replay at 200 Hz.
- 12,000 measured samples per run.
- 30 runs per implementation.
- Six implementations, producing 180 data rows plus one CSV header.
- Startup gate: `GEO_AB_FUSION_CHECKS,status=pass`.

## Aggregate results

| Implementation | Runs | State bytes | Mean us | Run-mean SD us | P50 us | P95 us | P99 us | Max us | Mean error deg | P95 error deg | Max error deg | Misses | NaNs | Stable hash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| A0 generic GEO float | 30 | 28 | 185.601656 | 0.015496 | 184 | 192 | 192 | 198 | 6.964863 | 14.726279 | 15.840757 | 0 | 0 | yes |
| B0 generic GEO Q16 | 30 | 32 | 227.351044 | 0.009506 | 226 | 233 | 234 | 235 | 7.706916 | 16.612671 | 17.342218 | 0 | 0 | yes |
| A1 fused GEO float | 30 | 28 | 52.513244 | 0.016171 | 52 | 58 | 59 | 60 | 6.964858 | 14.726279 | 15.840757 | 0 | 0 | yes |
| B1 fused GEO Q16 | 30 | 32 | 66.180644 | 0.008783 | 66 | 72 | 73 | 74 | 7.705232 | 16.611584 | 17.338682 | 0 | 0 | yes |
| C conventional quaternion | 30 | 28 | 52.563425 | 0.013149 | 52 | 58 | 60 | 60 | 6.964858 | 14.726279 | 15.840757 | 0 | 0 | yes |
| D quantized TinyML fixture | 30 | 3,684 | 202.381383 | 0.006740 | 201 | 208 | 209 | 209 | 18.548826 | 24.185940 | 25.340490 | 0 | 0 | yes |

Minimum free heap stabilized at 361,304 bytes and the largest free block at 335,872 bytes. No implementation produced a deadline miss or non-finite quaternion output.

## Comparisons

- A1 versus A0: **3.534x throughput** and **71.71% lower mean latency**.
- B1 versus B0: **3.435x throughput** and **70.89% lower mean latency**.
- A1 versus C: A1 measured 0.050181 microseconds lower on the run average, a 0.095% difference. This is parity at the current timer and run scale, not evidence of a robust performance lead.
- A1 and C reported the same task errors and the same named-field output hash `28d82ac5` across all 30 runs.
- B1 versus C: B1 used 25.91% more mean time, while retaining a 32-byte state and a bounded Q16 accuracy difference.
- A1 versus D: **3.854x throughput** with 131.6x less state.
- B1 versus D: **3.058x throughput** with 115.1x less state.
- B1 slightly reduced the measured fixed error relative to B0 because its fused accumulators round once per lowered result rather than after every generic primitive.

## Finding

The generic GEO implementation was not limited by the geometry itself. It was limited by lowering a sparse quaternion task into repeated complete `Cl(2,0)` products, temporary values, conversions, and intermediate rounding boundaries.

The manually frozen sparse schedule removed approximately 71% of mean latency for both floating and fixed paths. The floating fused path reached conventional quaternion parity while preserving the GEO-derived expression schedule. The fixed fused path remained slower than the conventional floating path, but it became substantially faster than the quantized TinyML fixture while retaining a 32-byte state and deterministic behavior.

This run supports the kernel architecture claim that semantic fusion and sparse lowering are required to realize the computational value of the operator representation. It does not establish that every geometric workload will match hand-specialized code, nor that the TinyML fixture represents a trained production model.

## Evidence notes

The first local summarization attempt reported no CSV records because Windows PowerShell wrote the monitor log as UTF-16 while the Python reader assumed UTF-8. The raw monitor output, completion marker, 181 extracted CSV lines, and result values were intact. The runner and summarizer were subsequently hardened to accept PowerShell UTF-16 logs, emit UTF-8 CSV, and summarize the extracted CSV artifact.

## Provenance

- `benchmarks/esp32_imu_baseline/main/geo_filter.c`
- `benchmarks/esp32_imu_baseline/main/geo_filter_fused.c`
- `benchmarks/esp32_imu_baseline/main/geo_filter_self_test.c`
- `benchmarks/esp32_imu_baseline/main/benchmark_common.c`
- `benchmarks/esp32_imu_baseline/scripts/summarize_results.py`
- `benchmarks/esp32_imu_baseline/scripts/run_fusion_benchmark.ps1`
- local physical monitor capture and extracted 181-line CSV, 2026-07-17
