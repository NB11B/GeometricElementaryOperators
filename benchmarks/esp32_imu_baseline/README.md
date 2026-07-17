# ESP32-C6 IMU Replay Benchmark: A, B, C, and D

This ESP-IDF application compares two GEO implementations against conventional quaternion and quantized TinyML baselines on the same deterministic orientation-estimation task.

## Implementations

- **A_geo_float**: allocation-free Mahony-style orientation filter whose quaternion products and gravity-frame rotations are executed through the floating GEO `Cl(2,0)` kernel.
- **B_geo_fixed_q16**: the same filter topology using the signed 32-bit Q16 GEO backend for quaternion products, correction arithmetic, integration, and state storage. Quaternion normalization is the explicit nonlinear boundary and is performed through decoded scalar values before requantization.
- **C_conventional_quaternion**: allocation-free Mahony-style quaternion orientation filter using conventional scalar/vector arithmetic.
- **D_quantized_tinyml**: compact int8 dense-network execution path using a 32 x 6 IMU window, 16 hidden activations, and 7 outputs. The checked-in sparse identity/copy weights make the executable deterministic and exercise a real quantized tensor loop. They are a benchmark fixture, not trained production weights.

The TinyML fixture currently measures runtime, memory, determinism, and integration behavior. Task-quality claims must wait until trained weights are substituted without changing the inference engine or result schema.

## GEO quaternion representation

The kernel currently exposes `Cl(2,0)`, while the replay task requires a full 3D quaternion. A and B therefore use the standard complex 2 x 2 matrix representation of a quaternion and represent each real 2 x 2 matrix with one `Cl(2,0)` value.

For `q = w + xi + yj + zk`:

```text
q -> A + iB
A = w + y e12
B = x e1 + z e2
```

Quaternion multiplication is then evaluated as:

```text
(A + iB)(C + iD) = (AC - BD) + i(AD + BC)
```

Each quaternion product therefore executes four GEO geometric products plus checked GEO additions/subtractions. The A and B paths do not call a duplicate scalar quaternion-product routine.

## Input mode

Only deterministic replay mode is included. No physical sensor is required.

The firmware procedurally regenerates the same 60-second, 12,000-sample, 200 Hz six-axis IMU trajectory for every implementation and every run. The trajectory contains smooth three-axis rotation, deterministic sensor noise, and periodic linear-acceleration disturbance. An analytic reference quaternion is generated for accuracy scoring.

## Build and run

```powershell
cd benchmarks\esp32_imu_baseline
idf.py fullclean
idf.py set-target esp32c6
idf.py build
idf.py -p COM5 flash monitor
```

The program performs 256 untimed warm-up samples and then 30 measured runs of 12,000 samples for each implementation.

## Output

Serial output is CSV-compatible and includes:

- implementation and run number;
- state bytes;
- mean, standard deviation, minimum, p50, p95, p99, and maximum latency;
- 200 Hz deadline misses;
- mean, p95, and maximum quaternion angular error;
- NaN count;
- deterministic output hash;
- minimum free heap and largest free block.

Capture the run with:

```powershell
idf.py -p COM5 monitor | Tee-Object esp32-imu-abcd.log
```

Extract result rows with:

```powershell
Select-String '^CSV,' esp32-imu-abcd.log |
  ForEach-Object { $_.Line } |
  Set-Content esp32-imu-abcd.csv
```

## Fairness controls

All four implementations use the same generated samples, reference trajectory, timer, compiler flags, warm-up count, run count, output structure, and 5 ms deadline. The timed region contains only one implementation step. Replay generation, reference generation, hashing, accuracy calculation, sorting, heap inspection, and serial printing are outside that timed region.

A and C use the same filter gains and confidence gate. B uses the same topology and gains after Q16 quantization. This isolates the execution representation rather than changing the task or controller policy.
