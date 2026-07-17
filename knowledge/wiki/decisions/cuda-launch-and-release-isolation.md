# CUDA launch validation and v0.19 release isolation

## Active-device launch limits

Status: accepted and verified for the reviewed Stage 1 implementation.

Generated CUDA launchers validate block and x-grid geometry against properties
of the active device. They do not assume that 1024 threads per block or the
host `UINT_MAX` represents the actual launch contract. Query errors propagate
as CUDA errors so a missing or invalid device is not misreported as a geometry
failure.

The zero-count path remains a successful no-op and does not require a current
device. This preserves the pre-existing API behavior.

## Output-contract distinction

Status: accepted and verified in the public and generated headers.

The public vector-wedge batch API returns scalar e12 coefficients. Generated
schedules always return complete Cl(2,0) roots, so generated vector wedge writes
`(0, 0, 0, e12_coefficient)`. Documentation, rather than an ABI change, makes
the distinction explicit.

## Release isolation

Status: active constraint.

The v0.19 integration branch contains PR #13 code through `0280f0d`, but its
base does not contain the later v0.18.2 provenance commit `4469433`. Therefore
v0.19 work must not be used as evidence that v0.18.2 was accepted, merged, or
tagged. The branch will need reconciliation with the eventual accepted v0.18.2
head after exact-head hosted CI executes successfully.

## Provenance

- `tools/generate_cuda_schedule.py`
- `include/geo/cuda.h`
- `tests/test_cuda_schedules.cu`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
