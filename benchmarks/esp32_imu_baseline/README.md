# ESP32-C6 IMU Replay Benchmark: Generic and Fused GEO

This ESP-IDF application compares generic and fused GEO orientation paths against conventional quaternion and quantized TinyML baselines on the same deterministic replay task.

## Implementations

- **A0_geo_float_generic**: floating Mahony-style orientation filter whose quaternion products and gravity rotations execute through complete generic GEO `Cl(2,0)` products.
- **B0_geo_fixed_q16_generic**: the same topology using the checked signed 32-bit Q16 GEO backend. Quaternion normalization is a decoded/requantized nonlinear boundary.
- **A1_geo_float_fused**: sparse floating schedule lowered from the same paired-`Cl(2,0)` expression. Known-zero products, temporary multivectors, and encode/decode operations are eliminated.
- **B1_geo_fixed_q16_fused**: sparse Q16 schedule with prequantized constants, fused product accumulation, checked state updates, fast input quantization, and a float normalization boundary.
- **C_conventional_quaternion**: allocation-free Mahony-style quaternion orientation filter using hand-written scalar/vector arithmetic.
- **D_quantized_tinyml**: compact int8 dense-network execution path using a 32 x 6 IMU window, 16 hidden activations, and 7 outputs. Its deterministic fixture weights measure inference cost and integration behavior; they are not trained production weights.

## GEO quaternion representation

The portable kernel exposes `Cl(2,0)`, while the task requires a full 3D quaternion. A quaternion is represented by two `Cl(2,0)` values through its standard complex 2 x 2 matrix representation.

For `q = w + xi + yj + zk`:

```text
q -> A + iB
A = w + y e12
B = x e1 + z e2
```

Quaternion multiplication is:

```text
(A + iB)(C + iD) = (AC - BD) + i(AD + BC)
```

A0 and B0 evaluate four complete generic `Cl(2,0)` products per quaternion product. A1 and B1 emit only the scalar operations that survive the known sparse grade pattern. The fused paths are manually frozen target schedules for the compiler/lowering stage; they do not change the filter gains, controller policy, replay, reference, or output contract.

## Input mode

Only deterministic replay mode is included. No physical sensor is required.

The firmware regenerates the same 60-second, 12,000-sample, 200 Hz six-axis IMU trajectory for every implementation and every run. The trajectory contains smooth three-axis rotation, deterministic sensor noise, and periodic linear-acceleration disturbance. An analytic reference quaternion is generated for accuracy scoring.

## Build and run

```powershell
cd C:\Users\nateb\Documents\GeometricElementaryOperators

git fetch origin
git switch benchmark/esp32-imu-clean-v1
git pull --ff-only origin benchmark/esp32-imu-clean-v1

. "C:\Users\nateb\esp\esp-idf\export.ps1"

cd benchmarks\esp32_imu_baseline
idf.py fullclean
idf.py set-target esp32c6
idf.py build
idf.py -p COM5 flash monitor 2>&1 |
    Tee-Object esp32-imu-fusion.log
```

The first benchmark-specific record must be:

```text
GEO_AB_FUSION_CHECKS,status=pass
```

The program performs 256 untimed warm-up samples followed by 30 measured runs of 12,000 samples for each of six implementations. A complete capture contains 180 data rows plus one CSV header row.

## Output

Serial output includes:

- implementation and run number;
- state bytes;
- mean, standard deviation, minimum, p50, p95, p99, and maximum latency;
- 200 Hz deadline misses;
- mean, p95, and maximum quaternion angular error;
- non-finite output count;
- deterministic named-field output hash;
- minimum free heap and largest free block.

Extract result rows:

```powershell
Select-String 'CSV,' .\esp32-imu-fusion.log |
    ForEach-Object {
        $_.Line -replace '^.*?CSV,', 'CSV,'
    } |
    Set-Content .\esp32-imu-fusion.csv

(Get-Content .\esp32-imu-fusion.csv).Count
```

Expected row count:

```text
181
```

Generate and validate the aggregate report:

```powershell
python .\scripts\summarize_results.py .\esp32-imu-fusion.csv `
    --markdown-out .\esp32-imu-fusion-summary.md `
    --csv-out .\esp32-imu-fusion-summary.csv
```

The summarizer fails when an implementation is missing, the run count is incomplete, a deadline or NaN count is nonzero, or an implementation produces more than one output hash across repeated runs.

## Fairness controls

All six implementations use the same samples, reference trajectory, timer, optimization flags, warm-up count, run count, output structure, and 5 ms deadline. The timed region contains only one implementation step. Replay generation, analytic reference generation, hashing, accuracy calculation, sorting, heap inspection, and serial printing remain outside the timed region.

A0, A1, and C use the same floating gains and confidence gate. B0 and B1 use the same topology and Q16 gains. A1/B1 change only the schedule lowering and conversion strategy. This isolates the performance effect of fusion without changing task semantics.

## Interpretation contract

- A0/B0 measure generic operator composition.
- A1/B1 measure sparse fused schedules derived from the same GEO expression.
- C measures a hand-specialized conventional implementation.
- D measures a deterministic quantized neural inference fixture.
- B0 and B1 are fixed algebra/integration paths with decoded quaternion normalization; they are not exclusively integer pipelines.
- Only the physical ESP32-C6 run may support timing claims.
