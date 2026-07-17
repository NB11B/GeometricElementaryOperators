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

`origin/main` now contains the PR #13 merge `0280f0d`, and the v0.19 branch is
based on that merged code. Neither branch contains the later local validation-
provenance commit `4469433`. The merge explicitly deferred release tagging
because hosted jobs never executed. Therefore v0.19 work must not be used as
evidence that v0.18.2 passed hosted validation or was release-tagged; the code
merge and release acceptance remain distinct states.

## Provenance

- `tools/generate_cuda_schedule.py`
- `include/geo/cuda.h`
- `tests/test_cuda_schedules.cu`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
