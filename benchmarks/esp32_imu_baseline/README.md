# ESP32-C6 IMU Replay Baselines: C and D

This ESP-IDF application establishes the two non-GEO baselines that will be measured before adding GEO float (A) and GEO fixed-point (B).

## Implementations

- **C_conventional_quaternion**: allocation-free Mahony-style quaternion orientation filter using conventional scalar/vector arithmetic.
- **D_quantized_tinyml**: compact int8 dense-network execution path using a 32 x 6 IMU window, 16 hidden activations, and 7 outputs. The checked-in sparse identity/copy weights make the executable deterministic and exercise a real quantized tensor loop. They are a benchmark fixture, not trained production weights.

The TinyML fixture currently measures runtime, memory, determinism, and integration behavior. Task-quality claims must wait until trained weights are substituted without changing the inference engine or result schema.

## Input mode

Only deterministic replay mode is included. No physical sensor is required.

The firmware procedurally regenerates the same 60-second, 12,000-sample, 200 Hz six-axis IMU trajectory for every implementation and every run. The trajectory contains smooth three-axis rotation, deterministic sensor noise, and periodic linear-acceleration disturbance. An analytic reference quaternion is generated for accuracy scoring.

## Build and run

```powershell
cd benchmarks\esp32_imu_baseline
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
idf.py -p COM5 monitor | Tee-Object esp32-imu-baseline-c-d.log
```

Extract result rows with:

```powershell
Select-String '^CSV,' esp32-imu-baseline-c-d.log |
  ForEach-Object { $_.Line } |
  Set-Content esp32-imu-baseline-c-d.csv
```

## Fairness controls

Both implementations use the same generated samples, reference trajectory, timer, compiler flags, warm-up count, run count, output structure, and deadline. The timed region contains only one implementation step. Replay generation, reference generation, hashing, accuracy calculation, sorting, heap inspection, and serial printing are outside that timed region.

## Next integration

After C and D results are preserved, add:

- `A_geo_float`, using the existing floating GEO backend;
- `B_geo_fixed`, using the existing Q-format GEO backend.

They should implement the same `benchmark_impl_t` interface and emit the same result rows without changing the replay generator or metric code.
