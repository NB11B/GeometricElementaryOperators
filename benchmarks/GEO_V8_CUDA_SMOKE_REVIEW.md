# GEO V8 CUDA Smoke Review

Physical smoke execution on the NVIDIA GeForce RTX 5070 Laptop GPU completed successfully for the native CUDA benchmark, PyTorch eager CUDA comparator, and three-trial protocol.

## Accepted smoke gates

- native CUDA benchmark executable completed;
- PyTorch eager CUDA comparator completed;
- three GEO trials and three PyTorch eager trials completed;
- aggregation completed with `GEO_GPU_BENCHMARK_PROTOCOL: PASS`;
- earlier physical correctness matrix passed 200 cases.

## Preliminary observation

The planned CUDA forward path is materially faster than the reference forward path in resident inference across much of the smoke matrix. This is descriptive only; the smoke run uses three trials and ten iterations and is not publication evidence.

## Required corrections before the nine-trial acceptance run

1. The reference forward routine currently allocates, copies, and frees the device signature on each invocation. That setup work contaminates resident-kernel timing. The signature must be uploaded once with the device plan and reused.
2. The native benchmark currently labels both training rows as reference and planned, but both dispatch to the same planned MSE/SGD implementation. The duplicate reference-training rows must be removed or replaced by a genuinely independent reference training path.
3. End-to-end checksums should be derived from the result produced by the end-to-end invocation rather than the persistent resident buffer.

Until these corrections are implemented and the correctness/smoke gates are rerun, no nine-trial GPU performance result should be treated as accepted.
