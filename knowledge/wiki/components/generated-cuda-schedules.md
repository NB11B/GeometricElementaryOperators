# Generated CUDA schedule launchers

## Status

Verified for the reviewed v0.19 Stage 1 candidate on 2026-07-16.

## Contract

`tools/generate_cuda_schedule.py` emits CUDA kernels and C-linkage launchers for
five compiled schedules. The launchers consume device-resident Cl(2,0) arrays,
write the complete schedule-root multivector, and accept an optional CUDA
stream represented as `void *`.

The generated vector-wedge schedule writes the complete blade
`(0, 0, 0, e12_coefficient)`. This differs intentionally from the public
`geo_cuda_cl20_vector_wedge_batch` API, which writes only the scalar e12
coefficient. Callers must not treat these output layouts as interchangeable.

## Launch validation

A nonzero launch checks pointers and block-size arithmetic, computes the block
count without unsigned wraparound, queries the active device, and rejects
geometry exceeding `maxThreadsPerBlock` or `maxGridSize[0]`. CUDA device and
property query failures are returned unchanged. A zero-count launch is a
successful no-op before pointer or device queries.

Real-GPU regressions verify that excessive block and grid dimensions return
`cudaErrorInvalidConfiguration`; a following synchronization would expose an
accidental rejected launch.

## Remaining validation boundary

Release float and double tests passed on an RTX 5070 Laptop GPU with CUDA 13.1.
Compute Sanitizer coverage is unresolved because Compute Sanitizer 2025.4
could not attach in the available environment.

## Provenance

- `tools/generate_cuda_schedule.py`
- `tests/test_cuda_schedules.cu`
- `tests/test_generators.py`
- `include/geo/cuda.h`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
