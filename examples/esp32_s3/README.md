# ESP32-S3 hardware validation example

This ESP-IDF application exercises the portable floating-point kernel and the complete fixed GEB program executor on physical ESP32-S3 hardware. It performs deterministic correctness checks, prints memory and timing records, and then runs an allocation-free soak indefinitely.

## Requirements

- ESP-IDF 5.x with an ESP32-S3 toolchain
- an ESP32-S3 development board
- a serial connection supported by `idf.py`

## Build and run

```sh
cd examples/esp32_s3
idf.py set-target esp32s3
idf.py build
idf.py -p PORT flash monitor
```

Use `Ctrl-]` to leave the serial monitor.

## Output contract

The application emits machine-readable records in addition to the normal ESP-IDF log prefix:

```text
GEO_DEVICE,...
GEO_CHECKS,status=pass
GEO_RESULT,round=0,backend=esp32_float,operation=product_rotor_chain,...
GEO_RESULT,round=0,backend=esp32_fixed,operation=product_rotor_chain,...
```

Each soak round records:

- backend and operation;
- iteration count and microseconds per operation;
- current free heap;
- largest free block;
- minimum free heap observed by ESP-IDF.

The application aborts on any numerical mismatch, fixed-program status error, or change in free heap/largest block during a measured allocation-free round. This makes a long serial capture suitable for checking both performance stability and heap stability.

## Capture representative evidence

```sh
idf.py -p PORT monitor 2>&1 | tee esp32-s3-validation.log
```

Keep the complete build metadata, `sdkconfig`, ESP-IDF version, board model, clock configuration, and log with any published benchmark result.
