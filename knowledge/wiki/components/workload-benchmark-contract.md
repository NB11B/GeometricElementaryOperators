# Workload benchmark contract

Status: verified for the reviewed v0.19 Stage 2 implementation.

## Contract

`tools/workload_report.py` treats benchmark output as a strict capability
contract rather than accepting whatever rows an executable happens to emit.
The CPU manifest contains exactly 27 operation/backend/timing-scope tuples. The
CUDA manifest contains exactly 11 tuples: six public batched API operations and
five generated schedules. Generated reverse is intentionally absent because no
generated reverse schedule exists.

The CUDA public API rows use `timing_scope=host_end_to_end`. Their host-clock
measurement includes allocation, transfers, launch, synchronization, and
release performed by the public API. Generated schedule rows use
`timing_scope=device_kernel`; CUDA events measure launchers operating on
device-resident buffers. These scopes are intentionally not interchangeable.

The reporter forwards one deterministic seed and validates the reported
precision, iteration count, warmup count, seed, batch, and selected operation.
`--require-cuda` turns a missing CUDA executable or device-skip exit into a
failure. Optional mode still permits CPU-only reporting.

CUDA rows report validated upload, download, and logical-kernel byte counts.
Dot and wedge account for their scalar public output versus complete generated
Cl(2,0) output, and rotor action accounts for its third generated input. Byte
arithmetic is checked before allocation. Reverse-only execution does not
allocate generated-schedule resources.

## Validation

The reviewed Stage 2 candidate passed:

- the eight-test Python workload-report contract suite;
- a WSL/GCC Release double build with 33/33 CTests and the workload-report
  target;
- the WSL generator suite;
- CUDA 13.1 Release float and double builds, each with 36/36 executable CTests
  after excluding the known Windows generator-toolchain mismatch;
- required-CUDA reports in both precisions, each producing 49 aggregate rows:
  27 CPU rows plus 22 CUDA rows for batches 32 and 257, with all 11 CUDA
  capabilities and zero mismatches; and
- a reverse-only real-GPU run on an RTX 5070 Laptop GPU.

An independent reviewer returned **PASS** after CPU configuration validation
and reverse-only resource allocation were corrected.

## Limitations

- GitHub Actions did not execute because of the repository billing/spending
  gate, so hosted exact-head validation is not claimed.
- Compute Sanitizer did not attach locally and remains unpassed.
- The Windows `generators` CTest was excluded because of its compiler-path and
  MSVC harness mismatch; the same generator suite passed under WSL.

## Provenance

- `benchmarks/bench_cuda.cu`
- `benchmarks/bench_workloads.c`
- `tools/workload_report.py`
- `tests/test_workload_report.py`
- `CMakeLists.txt`
- `docs/v0.19-validation.md`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
