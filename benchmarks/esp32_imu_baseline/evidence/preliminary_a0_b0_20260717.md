# Preliminary ESP32-C6 A0/B0 replay evidence — 2026-07-17

Status: partial physical-board capture supplied by the user. The complete A0 run and the first eight B0 runs were visible. C and D values are from the immediately preceding complete baseline capture on the same board and benchmark contract.

## Environment

- Target: ESP32-C6, revision v0.2
- CPU: 160 MHz
- SPI flash mode: DIO, 80 MHz
- ESP-IDF: v5.3.4-1025-g6f6766f917-dirty
- Replay: 12,000 samples at 200 Hz
- Runs: 30 requested
- Fixed format: Q16
- Startup validation: `GEO_AB_CHECKS,status=pass`

## Preliminary aggregate

| implementation | state bytes | mean latency | p50 | p95 | p99 | max | mean error | p95 error | max error | deadline misses | NaNs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 generic GEO float | 28 | 185.526 us | 184 us | 192 us | 192 us | 193 us | 6.964863 deg | 14.726279 deg | 15.840757 deg | 0 | 0 |
| B0 generic GEO fixed Q16 | 32 | 228.568 us | 226 us | 236 us | 243 us | 244 us | 7.706916 deg | 16.612671 deg | 17.342218 deg | 0 | 0 |
| C conventional quaternion | 28 | 52.181 us | 52 us | 53 us | 53 us | 60 us | 6.964858 deg | 14.726279 deg | 15.840757 deg | 0 | 0 |
| D quantized TinyML fixture | 3684 | 220.371 us | 220 us | 221 us | 228 us | 229 us | 18.549 deg | 24.186 deg | 25.340 deg | 0 | 0 |

B0 latency is the mean of the eight visible run means. Its accuracy, percentiles, hash, state size, heap, deadline, and NaN fields were identical across those eight visible runs.

## Findings

- A0 reproduced C task accuracy to displayed precision while using the generic paired-`Cl(2,0)` lowering.
- A0 was approximately 3.56 times slower than C but approximately 15.8 percent faster than D.
- B0 retained bounded task error and deterministic output but was approximately 23.2 percent slower than A0 and approximately 3.7 percent slower than D.
- A0 and B0 remained far inside the 5 ms deadline and used only 28 and 32 bytes of state.
- Stable named-field hashes, zero NaNs, zero deadline misses, and stable heap observations support deterministic physical-device execution for the visible runs.

## Bottleneck interpretation

The generic A0 path evaluates five quaternion products per sample. Each quaternion product is lowered through four complete generic `Cl(2,0)` products, causing twenty full generic products per sample plus encode/decode and temporary-object overhead. B0 adds repeated conversion, checked fixed primitives, and a decoded normalization boundary.

The next controlled run therefore preserves A0/B0 and adds A1/B1 sparse fused schedules derived from the same paired-`Cl(2,0)` expression. The optimization target is lowering and fusion, not a change in filter policy or task semantics.

## Evidence boundary

This file records the supplied preliminary output but does not substitute for the complete serial log and extracted CSV. Final claims require the full 120-row A0/B0/C/D capture or a repeated complete run preserved under an exact commit identity.
