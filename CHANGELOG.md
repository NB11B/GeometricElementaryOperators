# Changelog

All notable changes are recorded here. Release compatibility follows `docs/RELEASE_POLICY.md`.

## Unreleased — 0.19.0

### Added

- optional CUDA Toolkit 13.x backend with a stable C ABI;
- batched CUDA addition, geometric product, reverse, vector dot, vector wedge, and rotor action;
- deterministic CPU/GPU equivalence tests and a CUDA-event benchmark harness;
- generated CUDA schedule kernels compiled from the same checked schedule JSON used by RTL;
- complete flat fixed-point GEB-36 program execution with checked type, register, overflow, divide, and projective-scale handling;
- fixed-point scalar/geometric Omega state, opposite-lane propagation, banked execution, and `M2(R)` routing control;
- shared cross-backend workload timing reports with P50/P95/P99 statistics;
- deterministic fixed-versus-floating numerical envelopes with component, angular, scale, overflow, and mismatch reporting;
- standalone ESP32-S3 correctness, timing, memory, and heap-stability application;
- executable fixed-point schedule generation for C and SystemVerilog;
- schedule-level fixed C/RTL nominal and overflow equivalence vectors;
- generic ARM Cortex-M DWT cycle source for the existing benchmark API;
- release, ABI, generated-artifact, and numerical compatibility policy.

### Fixed

- restored the exported `geo_struct_program_execute` and `geo_banked_execute` public symbols;
- removed signed-overflow undefined behavior from the selected-backend benchmark sink;
- rejected NaN and non-finite embedded EML inputs with explicit status;
- added fixed-product and accumulation overflow signaling to generated RTL;
- invalidated RTL outputs when overflow is asserted;
- corrected benchmark labels that overstated generated or complete backend coverage;
- corrected the numerical wedge fixture to compare the full bivector result;
- made fixed-control validation portable across every supported Q1 through Q30 configuration.

### Validation

- release-blocker changes passed the original 13-job host/sanitizer/Q-format/ESP32/RTL matrix before the expanded workflow was introduced;
- the final candidate requires the expanded host, sanitizer, Q-format, ESP32, CUDA 13 compile, generated CUDA schedule, numerical-envelope, and RTL-equivalence matrix;
- physical CUDA and ESP32-S3 results must be attached before hardware performance claims are published.

## 0.18.1

### Changed

- hardened malformed fused/static program validation;
- enforced supported Q-format bounds;
- added overflow-safe fixed-point constants and involutions;
- completed ESP32 component source coverage and standalone validation;
- made comparisons NaN-safe;
- made zero-instruction C emission C11-safe;
- aligned RTL/C fixed-point rounding;
- added generated controller simulation and synthesis coverage.

### Known post-merge findings

The following defects were found after merge and are corrected in the 0.19.0 work:

- undefined signed accumulation in the selected-backend fixed-point sink;
- missing RTL overflow signal and truncation mismatch with fixed C;
- missing established public executor symbols;
- successful NaN result from `geo_eml_log`;
- benchmark labels that implied broader path coverage than was measured.
