# `geo_grad`: GEO-Native Gradient Tool V7.1

`geo_grad` is a standalone command-line tool built directly on the GEO V7 differentiation runtime. It performs model checking, training, prediction, checkpointing, and C-header export without calling an external machine-learning or autograd framework.

## Supported model

V7.1 intentionally begins with one complete trainable model family:

```text
y = x * w
```

or

```text
y = w * x
```

where `x`, `y`, and trainable parameter `w` are multivectors in one declared `Cl(p,q)` coefficient space.

The tool supports dimensions one through six, all non-degenerate diagonal signatures with entries `+1` or `-1`, left or right geometric-product action, and either GEO-native SGD or GEO-native Adam.

## Model file

The model file is a deterministic line-oriented format:

```text
version=7.1
model=multivector_gp
dimension=3
signature=1,1,-1
side=right
optimizer=adam
learning_rate=0.01
epochs=100
beta1=0.9
beta2=0.999
epsilon=1e-8
seed=1
```

Unknown keys, unsupported model families, invalid signatures, invalid optimizer settings, and unsupported dimensions are rejected explicitly.

## Commands

Create a complete example model and training set:

```bash
geo_grad init-example model.geo train.csv
```

Check the native gradient contract and compile the V7 graph:

```bash
geo_grad check model.geo
```

Train and save a resumable checkpoint:

```bash
geo_grad train model.geo train.csv checkpoint.txt
```

Run inference from coefficient rows:

```bash
geo_grad predict model.geo checkpoint.txt input.csv output.csv
```

Export the learned multivector as a self-contained C header:

```bash
geo_grad export-c model.geo checkpoint.txt trained_model.h trained_model
```

## Dataset contract

For dimension `n`, each multivector contains `2^n` coefficients in blade-mask order.

A training row contains:

```text
input coefficients, target coefficients
```

A prediction row contains only input coefficients.

Blank lines and lines beginning with `#` are ignored. Every numeric row must have exactly the expected coefficient count. Non-finite values are rejected.

## Checkpoint contract

A checkpoint records:

- dimension;
- signature;
- left/right action;
- parameter coefficients;
- optimizer step;
- Adam first moments;
- Adam second moments.

Checkpoint metadata must match the model file before it can be loaded.

## Native execution path

The training command executes:

```text
parse and validate model
-> construct GEO V7 graph
-> set input and target values
-> GEO forward
-> GEO backward
-> GEO SGD or Adam step
-> save GEO parameter and optimizer state
```

There is no PyTorch, JAX, TensorFlow, external tape, surrogate forward path, numerical-gradient approximation, or silent fallback.

## Verification

`geo_grad check` performs an analytic JVP/VJP adjoint-identity test for the configured geometric-product side and compiles and executes a complete V7 forward/backward graph.

Expected marker:

```text
GEO_GRAD_CHECK: PASS ... adjoint_identity=PASS ... unsupported_fallbacks=0 no_external_autograd=TRUE
```

The end-to-end CLI test creates an example, checks it, trains it, predicts from the learned checkpoint, exports a C header, and verifies malformed-model rejection.

Expected marker:

```text
GEO_GRAD_CLI_TEST: PASS init=PASS check=PASS train=PASS predict=PASS export=PASS invalid_model=REJECTED no_external_autograd=TRUE
```

## Scope boundary

V7.1 is a usable standalone gradient and training tool for trainable left/right multivector geometric-product maps. It is not yet a general parser for arbitrary V7 graphs, mini-batch gradient accumulation, shuffled data loading, multiple parameters, constrained rotor updates, CUDA-native training, or distributed execution.

Those capabilities can now be added above the same GEO-owned differentiation runtime without introducing an external autograd dependency.
