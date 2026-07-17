# Exact workload capability and timing contract

Status: accepted and verified for v0.19 Stage 2.

## Decision

The unified workload reporter validates an exact, version-controlled set of
operation/backend/timing-scope capabilities. Missing, extra, or duplicate rows
are errors. Reported run configuration and CUDA transfer/logical byte counts
must also match the invocation and operation contract.

Public CUDA APIs and generated CUDA schedules remain separate capabilities:

- public APIs are measured as `host_end_to_end` because allocation, transfer,
  synchronization, and release are part of those calls;
- generated schedules are measured as `device_kernel` on resident buffers;
- generated reverse is not advertised because that launcher does not exist.

Required-CUDA reporting fails when the CUDA executable is absent or reports a
device skip. CPU-only reporting remains available only when CUDA is optional.

## Rationale

An inferred or permissive manifest can silently convert a missing backend path
into a green report. Precise timing scopes prevent unlike measurements from
being presented as equivalent. Exact configuration and byte-count checks keep
reproducibility and transfer-cost claims tied to the command that actually ran.

## Consequences

- Adding or removing a benchmark capability requires an intentional manifest
  and contract-test update.
- Report consumers may compare rows only within a compatible timing scope.
- The absence of generated reverse remains explicit rather than being filled
  by a private benchmark-only kernel.
- Reverse-only public execution avoids unnecessary generated CUDA resources.

## Provenance

- `tools/workload_report.py`
- `benchmarks/bench_cuda.cu`
- `benchmarks/bench_workloads.c`
- `tests/test_workload_report.py`
- `docs/v0.19-validation.md`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
