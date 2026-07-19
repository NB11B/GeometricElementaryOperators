# GEO Gradient Tool V7.1 Acceptance

## Accepted code head

`1375efcb7329956cc194e2d8d9930d4ffba5cc14`

The acceptance-record commit changes documentation only.

## Tool accepted

`geo_grad` is the first standalone developer tool built on the GEO V7 internal differentiation runtime.

Accepted commands:

```text
geo_grad init-example <model.geo> <train.csv>
geo_grad check <model.geo>
geo_grad train <model.geo> <train.csv> <checkpoint.txt>
geo_grad predict <model.geo> <checkpoint.txt> <input.csv> <output.csv>
geo_grad export-c <model.geo> <checkpoint.txt> <header.h> <symbol>
```

The accepted model family learns a trainable multivector geometric-product map using either right action `y = x * w` or left action `y = w * x`.

The tool supports dimensions one through six, non-degenerate diagonal signatures, GEO-native SGD, GEO-native Adam, deterministic CSV datasets, resumable parameter and optimizer checkpoints, prediction, and generated C parameter headers.

## Native execution contract

The training command executes:

```text
model validation
-> GEO V7 graph construction
-> GEO forward
-> GEO backward
-> GEO cotangent accumulation
-> GEO SGD or Adam
-> GEO checkpoint
```

No external autograd engine, machine-learning framework, numerical-gradient approximation, surrogate forward recomputation, or silent fallback is used.

## Hosted validation

### GEO gradient tool V7.1 — run `29708735769`

All jobs passed against the accepted code head:

- GCC release configure/build/test: PASS;
- Clang release configure/build/test: PASS;
- GCC AddressSanitizer plus UndefinedBehaviorSanitizer: PASS;
- Windows MSVC release configure/build/test: PASS;
- full host build and complete CTest regression with tools enabled: PASS.

### Shared regression workflows

The following existing workflows also passed unchanged on the same head:

- GEO native gradients V1 — run `29708735760`;
- Geometric operator kernel V5.1 — run `29708735776`;
- GEO native autodiff V7 — run `29708735752`.

## End-to-end CLI validation

The CLI acceptance test performs the complete workflow:

1. creates an example model and dataset;
2. checks analytic JVP/VJP adjoint identity;
3. compiles and executes a V7 forward/backward graph;
4. trains a `Cl(2,0)` multivector transformation;
5. verifies substantial loss reduction;
6. reloads the checkpoint;
7. predicts from all four basis inputs;
8. verifies the learned first prediction against the declared target coefficients;
9. exports a self-contained C header;
10. rejects a malformed metric signature.

Expected marker:

```text
GEO_GRAD_CLI_TEST: PASS init=PASS check=PASS train=PASS predict=PASS export=PASS invalid_model=REJECTED no_external_autograd=TRUE
```

## Claim boundary

V7.1 is a usable standalone GEO-native gradient and training tool for one complete multivector geometric-product model family. It does not yet provide arbitrary graph parsing, multiple trainable parameters, mini-batch gradient accumulation, constrained rotor optimization, CUDA-native training, or distributed execution.

These are tool-layer extensions above the accepted GEO-owned differentiation runtime. Unsupported model families and invalid contracts fail explicitly.

## Final status

```text
GEO_GRAD_V7_1: PASS
```
