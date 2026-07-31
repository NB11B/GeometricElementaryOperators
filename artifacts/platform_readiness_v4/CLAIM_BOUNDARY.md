# GEO Rank-8 Platform Baseline Claim Boundary

## Supported

GEO Compact rank 8 is the current platform baseline.

In the recovered five-seed rank calibration, it achieved
practical quality parity with ordinary rank-8 low-rank
factorization while reducing trainable parameters by
approximately 15.14% relative to the dense model.

The platform provides:

- 8,401,888 trainable parameters
- 12 GEO compact projection replacements
- GEO-owned explicit PyTorch autograd
- CUDA tensor execution
- reference/runtime numerical parity
- finite nonzero U, V, and alpha gradients
- optimizer updates
- model serialization
- canonical checkpoint compatibility fixtures
- durable experiment tooling
- real-model PyTorch-CUDA systems measurements

## Not Yet Supported

- recovery of all original trained checkpoint weights
- a custom compiled implicit-linear CUDA kernel
- compiled-kernel performance claims
- source-shifted transfer conclusions
- general GEO superiority over ordinary low rank

## Future Gate

A custom compiled implementation is tracked separately as:

GEO_R8_COMPILED_KERNEL_GATE
