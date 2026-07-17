# ESP32-C6 GEO fused IMU schedules

Status: implemented; exact ESP-IDF compile and physical timing evidence remain pending.

## Purpose

This component isolates the cost of GEO expression lowering by preserving generic A0/B0 implementations and adding sparse fused A1/B1 implementations to the same deterministic ESP32-C6 IMU replay benchmark.

## Variants

- `A0_geo_float_generic`: complete generic floating `Cl(2,0)` products.
- `B0_geo_fixed_q16_generic`: complete checked Q16 `Cl(2,0)` products.
- `A1_geo_float_fused`: sparse floating schedule derived from the paired-`Cl(2,0)` quaternion expression.
- `B1_geo_fixed_q16_fused`: sparse Q16 schedule with prequantized constants and fused accumulators.
- `C_conventional_quaternion`: hand-specialized scalar baseline.
- `D_quantized_tinyml`: deterministic int8 inference-cost fixture.

## Lowering contract

A1/B1 retain the quaternion embedding used by A0/B0:

```text
q -> A + iB
A = w + y e12
B = x e1 + z e2
(A+iB)(C+iD) = (AC-BD) + i(AD+BC)
```

The generic path evaluates all components of four `Cl(2,0)` products per quaternion product. The fused path removes operations whose inputs are known to be zero from the grade pattern and emits the remaining scalar dependency graph directly.

A1 becoming similar to C is an expected compiler result: the geometric expression and the conventional quaternion expression lower to the same minimal arithmetic graph for this specific task. The experiment measures whether GEO can reach that graph without changing semantics.

## Fixed schedule

B1 uses Q16 state, Q16 correction/integration, fused Q32 product accumulators, checked Q16 writes, prequantized filter constants, and fast replay-boundary input quantization. Quaternion normalization remains an explicit float decode/normalize/requantize boundary.

## Measurement contract

The six variants share the same replay, analytic reference, 200 Hz update rate, 5 ms deadline, 256-sample warm-up, 30 measured runs, 12,000 samples per run, timer, result schema, and heap instrumentation.

A complete run emits 180 data rows plus one CSV header row. `scripts/summarize_results.py` requires all six implementations, exactly 30 runs each, zero deadline misses, zero NaNs, and one output hash per implementation.

## Preliminary expectation

The supplied physical A0/B0 evidence establishes the optimization target:

- A0: approximately 185.526 microseconds per sample.
- B0: approximately 228.568 microseconds per sample.
- C: approximately 52.181 microseconds per sample.
- D: approximately 220.371 microseconds per sample.

A development numerical mirror predicts B1 task error near the B0 envelope and no Q16 overflow on the deterministic replay. No A1/B1 timing claim is made before physical execution.

## Provenance

- `benchmarks/esp32_imu_baseline/main/geo_filter_fused.c`
- `benchmarks/esp32_imu_baseline/main/geo_filter_fused.h`
- `benchmarks/esp32_imu_baseline/main/geo_filter_self_test.c`
- `benchmarks/esp32_imu_baseline/main/main.c`
- `benchmarks/esp32_imu_baseline/scripts/summarize_results.py`
- `benchmarks/esp32_imu_baseline/evidence/preliminary_a0_b0_20260717.md`
- `knowledge/wiki/tasks/TASK-20260717-003.md`
