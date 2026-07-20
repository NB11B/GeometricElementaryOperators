# Unresolved Knowledge Questions

## TASK-20260717-004

- Does the standalone main-based ESP-IDF project reproduce the predecessor output hashes exactly?
- What cycle-level A1/C ratio is observed under randomized implementation order?
- Can automatically generated A1/B1 schedules match the manually frozen hashes and timing within 1%?
- Which fixed-point format minimizes B-path latency while preserving the required task error envelope?
- How do the generated schedules perform on ESP32-S3 and in externally measured energy per update?

## TASK-20260717-005

- Should the IR remain polynomial-only or be generalized immediately to typed expression DAGs?
- How should per-intermediate Q formats and overflow envelopes be represented?
- Should common-subexpression elimination occur before or after backend-specific accumulator formation?
- Can the same schedule IR emit scalar C, ESP32-S3 SIMD, CUDA, and RTL without changing task semantics?

## Provenance

- `knowledge/wiki/tasks/TASK-20260717-004.md`
- `knowledge/wiki/tasks/TASK-20260717-005.md`
