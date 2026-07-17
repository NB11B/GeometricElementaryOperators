# ESP32-C6 IMU Fusion Benchmark — 2026-07-17

## Status

Completed with validated physical timing. Clean-branch reproduction remains pending.

## Evidence

- Firmware commit: `e97ddad8ad472199624965ae24d87412846a70c6`
- Branch: `benchmark/esp32-imu-fused-a1-b1`
- ESP-IDF: `v5.3.4-1025-g6f6766f917-dirty`
- Target: ESP32-C6 revision 0.2 at 160 MHz
- 30 runs per implementation
- 12,000 samples per run at 200 Hz
- 181 CSV lines
- Validation: PASS

## Results

| path | mean latency us | state bytes | mean error deg |
|---|---:|---:|---:|
| A0 generic float | 185.601656 | 28 | 6.964863 |
| B0 generic Q16 | 227.351044 | 32 | 7.706916 |
| A1 fused float | 52.513244 | 28 | 6.964858 |
| B1 fused Q16 | 66.180644 | 32 | 7.705232 |
| C conventional | 52.563425 | 28 | 6.964858 |
| D TinyML fixture | 202.381383 | 3684 | 18.548826 |

All paths had zero deadline misses, zero NaNs, and stable hashes.

## Interpretation

A1 achieved measurement-level parity with C while preserving the same displayed accuracy and output hash. Fusion reduced A0 latency by 71.71%. B1 reduced B0 latency by 70.89% and retained a bounded Q16 error envelope.

The experiment isolates the primary bottleneck as missing sparse lowering and fusion rather than the paired-`Cl(2,0)` representation.

## Limitations

- Microsecond timing cannot resolve a robust 0.095% A1/C difference.
- Fixed paths decode for quaternion normalization.
- D is an inference-cost fixture, not a trained accuracy baseline.
- Implementation order was fixed rather than randomized.

## Follow-on

1. Rebuild the clean main-based branch.
2. Add cycle-counter timing and randomized order.
3. Generate A1/B1 automatically from a schedule IR.
4. Profile B1 by stage and evaluate multiple Q formats.
5. Repeat on ESP32-S3 and measure external energy per update.
