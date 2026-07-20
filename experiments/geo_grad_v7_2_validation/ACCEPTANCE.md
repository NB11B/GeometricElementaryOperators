# GEO Gradient V7.2 Non-CUDA Validation Acceptance

## Accepted code head

`30275afaeec5f1b70b3b16289f996f633f5300f5`

The acceptance-record commit changes documentation only.

## Scope accepted

This milestone validates the complete declared CPU scope of the GEO V7 native differentiation runtime and the V7.1 `geo_grad` tool.

CUDA kernels, physical GPU evidence, distributed training, mixed-precision training, and arbitrary model-file graph parsing beyond the registered V7 runtime remain outside this acceptance boundary.

## Independent mathematical validation

The production geometric-product implementation is not used as the sole oracle. The V7.2 core suite independently reconstructs:

- blade swap parity;
- metric-overlap signs;
- coefficient routing by XOR;
- dense geometric products;
- coefficient-space left and right VJPs.

The suite validates production results against this independent implementation across:

- dimensions 1 through 6;
- all 27 canonical non-degenerate signatures;
- left and right parameter placement;
- four deterministic dense trials per signature;
- analytic JVP/VJP adjoint identity;
- central finite-difference directional diagnostics.

## Graph-runtime validation

Validated runtime behavior includes:

- chained graphs;
- diamond fan-out and fan-in;
- exact branch cotangent accumulation;
- 128-node maximum-capacity execution;
- explicit node-129 rejection;
- forward-before-backward lifecycle enforcement;
- backward-before-optimizer lifecycle enforcement;
- stale-state invalidation;
- transactional rejection of non-finite values;
- 10,000 forward/backward/update lifecycle cycles.

## Optimizer validation

The suite independently checks:

- SGD gradient values and parameter updates;
- Adam first-step bias correction and parameter updates;
- deterministic optimizer-state replay;
- invalid learning-rate and beta rejection.

## CLI validation

The `geo_grad` validation executes 54 complete end-to-end cases:

```text
27 canonical signatures x 2 action sides
```

Every case performs:

- model validation;
- native GEO graph construction;
- native SGD training;
- recovery of known multivector coefficients;
- checkpoint serialization and parse;
- prediction over every basis blade;
- independent prediction comparison;
- generated C-header export.

Additional CLI coverage includes malformed models, invalid signatures, unsupported versions and optimizers, malformed datasets, NaN and infinity rejection, corrupted and mismatched checkpoints, invalid C symbols, and byte-identical repeated seeded Adam training.

## Hosted validation

### GEO gradient V7.2 validation — run `29709219440`

All jobs passed:

- GCC Release;
- GCC Debug;
- Clang Release;
- Clang Debug;
- GCC AddressSanitizer and UndefinedBehaviorSanitizer;
- Valgrind full leak and invalid-access check;
- Windows MSVC Release;
- complete host build and CTest regression.

### Shared regression workflows

All shared workflows passed on the same accepted head:

- GEO gradient tool V7.1 — run `29709219441`;
- GEO native autodiff V7 — run `29709219448`;
- Geometric operator kernel V5.1 — run `29709219444`;
- GEO native gradients V1 — run `29709219442`.

## Runtime markers

```text
GEO_GRAD_V7_2_CORE_VALIDATION: PASS
GEO_GRAD_V7_2_CLI_VALIDATION: PASS
```

## Final status

```text
GEO_GRAD_V7_2_NON_CUDA_VALIDATION: PASS
external_autograd=NONE
cuda=DEFERRED
```
