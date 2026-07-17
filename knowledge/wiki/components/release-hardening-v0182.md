# v0.18.2 release hardening

## Status

Verified locally and independently reviewed. Publication and the exact pushed-head GitHub CI gate remain active process requirements.

## Durable behavior

The v0.18.2 hotfix restores public executor symbols and the legacy caller-visible `geo_optimized_witness_t` layout, makes terminal count an explicit folding input, hardens selected-path benchmark arithmetic and labeling, rejects non-finite embedded-logarithm inputs, and gives the generated fixed-point product/controller RTL an explicit tested overflow protocol.

The release surface is isolated from CUDA and v0.19 backend integration. No CUDA source or v0.19 feature was changed while correcting the exact-head safety-test compile blocker.

## Validation boundary

Local GCC, Clang, sanitizer, Q-format, ESP32 component, report/schema, Icarus, Yosys, MSVC, and archive-symbol gates passed. Independent review returned PASS with no BLOCKER or HIGH finding.

This evidence establishes a reviewed merge candidate; it does not replace the required GitHub Actions run against the exact pushed commit. Run #215 failed before checkout and therefore contributes no product-code result.

## Provenance

- `docs/v0.18.2-release-gate.md`
- `.github/workflows/ci.yml`
- `include/geo/optimizer.h`
- `include/geo/folding.h`
- `src/folding.c`
- `src/executor_api.c`
- `src/eml_embedded.c`
- `benchmarks/bench_complete.c`
- `tools/generate_rtl.py`
- `tests/test_release_hardening.c`
- `tests/test_abi_consumer.c`
- `tests/test_safety.c`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
