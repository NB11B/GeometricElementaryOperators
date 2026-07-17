# Changelog

All notable changes are recorded here. Release compatibility follows `docs/RELEASE_POLICY.md`.

## Unreleased — 0.19.0

### Added

- optional CUDA Toolkit 13.x backend with a stable C ABI;
- batched CUDA addition, geometric product, reverse, vector dot, vector wedge, and rotor action;
- deterministic CPU/GPU equivalence tests and a CUDA-event benchmark harness;
- operation-specific CUDA transfer accounting, checked batch/grid sizing, logical kernel bandwidth, and transfer-bandwidth reporting;
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

### Changed

- the established `bench_complete` executable name is retained while its visible description and reports identify it as a selected-path benchmark;
- CUDA CSV operation names now match the shared workload names: `addition`, `geometric_product`, `reverse`, `vector_dot`, `vector_wedge`, and `rotor_action`.

### Validation required before release

- GCC and Clang release builds in float and double precision;
- ASan/UBSan, including selected-path and numerical benchmark execution;
- Q1, Q8, Q16, Q24, and Q30 fixed-point validation;
- standalone ESP32 component and ESP32-S3 application builds;
- CUDA Toolkit 13.x float/double compilation and generated-schedule tests;
- fixed-C/RTL simulation and synthesis equivalence;
- physical CUDA and ESP32-S3 evidence before publishing hardware performance claims.

Hosted runner startup failures are not counted as test failures or passes. The release remains untagged until the exact candidate commit has complete evidence.

## 0.18.2 — pending merge and tag

### Fixed

- restored the legacy caller-owned `geo_optimized_witness_t` layout;
- restored exported `geo_struct_program_execute` and `geo_banked_execute` symbols while retaining `_impl` entry points;
- removed signed-overflow undefined behavior from the selected-path fixed-point benchmark sink;
- checked benchmark conversions, result kinds, and executor statuses before consuming outputs;
- rejected NaN and both infinities in `geo_eml_log` without changing caller output storage;
- added rounded-product and final-wide-sum overflow signaling to generated RTL while preserving valid cancellation cases;
- defined serialized controller completion, sticky-overflow, abort, dependency, `done`, and `result_valid` behavior;
- corrected benchmark labels and recorded the exact operation/path matrix in stdout, JSON, and Markdown;
- added sanitizer benchmark execution, archive-symbol checks, a mandatory Windows ABI consumer, and pipeline failure propagation.

### Release status

The hotfix is isolated in PR #13 and is not yet tagged. Its full release gate is documented in `docs/v0.18.2-release-gate.md`.

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

The following defects were found after merge and are addressed by the pending v0.18.2 hotfix, then carried forward into v0.19:

- undefined signed accumulation in the selected-path fixed-point sink;
- missing RTL overflow signal and truncation mismatch with fixed C;
- missing established public executor symbols;
- caller-owned optimizer ABI growth;
- successful NaN result from `geo_eml_log`;
- benchmark labels that implied broader path coverage than was measured.
