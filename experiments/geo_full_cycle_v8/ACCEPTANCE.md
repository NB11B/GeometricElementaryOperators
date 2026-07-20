# GEO Full-Cycle Runtime V8 Acceptance

## Accepted code head

`d6d4767773432ffb7b5e5065a78fee77146d22cc`

The acceptance-record commit changes documentation only.

## Accepted full-cycle capability

V8 owns the declared non-CUDA CPU cycle:

```text
strict JSON graph model
-> deterministic GEO IR
-> dynamically allocated native graph
-> forward inference
-> native reverse differentiation
-> minibatch gradient accumulation
-> SGD or Adam
-> transactional constraint projection
-> checkpoint/resume
-> stateful inference
-> C parameter export
```

No external autograd engine, tensor framework, surrogate forward path, finite-difference training gradient, or silent fallback participates in execution.

## Runtime scope accepted

- arbitrary directed acyclic graphs built from registered V8 nodes;
- multiple named inputs, targets, constants, parameters, and states;
- graphs larger than the V7 128-node bound;
- addition, scaling, geometric product, reversion, grade projection, grade involution, Clifford conjugation, Hadamard product, tanh, sigmoid, Euclidean normalization, and squared norm;
- independent leaf-gradient accumulation across minibatches;
- SGD and Adam with shared optimizer step and per-parameter moments;
- Euclidean-unit, metric-unit-vector, and even-versor constraints;
- simultaneous recurrent-state commits;
- streaming CSV training and prediction;
- checkpoint save/load and byte-identical resume;
- generated C parameter headers.

## Hosted evidence

### GEO full-cycle V8 — run `29710549060`

All jobs passed against the accepted code head:

- GCC Release: PASS;
- GCC Debug: PASS;
- Clang Release: PASS;
- Clang Debug: PASS;
- GCC AddressSanitizer plus UndefinedBehaviorSanitizer: PASS;
- Valgrind full leak and invalid-access gate: PASS;
- Windows MSVC Release: PASS;
- complete host build and CTest regression: PASS;
- all committed V8 JSON examples compiled and executed through `geo_cycle check`: PASS.

### Shared regression workflows

The following existing workflows also passed unchanged on the same head:

- Geometric operator kernel V5.1 — run `29710549034`;
- GEO native gradients V1 — run `29710549036`;
- GEO native autodiff V7 — run `29710549051`;
- GEO gradient tool V7.1 — run `29710549055`;
- GEO gradient V7.2 validation — run `29710549059`.

## Core mathematical and lifecycle validation

`test_full_cycle_v8` validates:

- all 27 canonical non-degenerate signatures in dimensions one through six;
- a composed multi-parameter graph containing geometric product, tanh, sigmoid, Hadamard gating, reversion, grade involution, Clifford conjugation, grade projection, addition, normalization, and squared loss;
- independent central-directional finite-difference diagnostics for multiple parameters;
- exact minibatch gradient accumulation;
- a dynamically grown 602-node graph;
- recurrent state update and continued native optimization;
- unit Euclidean, unit metric-vector, and even-versor projection;
- transactional preservation when a constraint projection fails.

Expected marker:

```text
GEO_FULL_CYCLE_V8_TEST: PASS signatures=27 arbitrary_graph=PASS multiple_parameters=PASS nonlinear_adjoints=PASS batching=PASS stateful_recurrence=PASS dynamic_nodes=602 constraints=PASS external_autograd=NONE cuda=DEFERRED
```

## Model compiler and tool validation

`test_geo_cycle_v8.py` validates:

- strict arbitrary JSON model validation and deterministic IR generation;
- exact one-step recovery of a batched multivector GP parameter;
- exact recovery of two independently routed parameters;
- nonlinear multilayer training and prediction;
- stateful recurrent-cell training and sequence inference;
- constrained parameter training;
- a separately generated 327-node JSON graph;
- a 2,048-row streaming dataset with minibatches;
- byte-identical ten-step checkpoint resume versus uninterrupted twenty-step Adam training;
- checkpoint load, prediction, C export, malformed JSON, invalid graph references, invalid constraints, corrupt IR, and unsupported operations.

Expected marker:

```text
GEO_CYCLE_V8_CLI_TEST: PASS arbitrary_json=PASS multiple_parameters=PASS multilayer=PASS batching=PASS streaming_rows=2048 recurrent=PASS checkpoint_resume=BYTE_IDENTICAL export=PASS constraints=PASS dynamic_nodes=327 adversarial_cases=8 external_autograd=NONE cuda=DEFERRED
```

## Precise recurrent boundary

V8 supports stateful forward execution, state persistence, reset controls, checkpointed state, and one-step truncated recurrent differentiation. It does not automatically propagate gradients backward through an unbounded sequence of previously committed states. Long-horizon backpropagation can be represented by explicitly unrolling the sequence as a larger acyclic graph; an automatic BPTT/checkpoint planner is not accepted in this milestone.

## LGM boundary

V8 supplies the scalable native graph, multi-parameter, state, batching, and training substrate required to construct geometric grammar architectures. The committed `lgm_stack.json` is an architecture example. This acceptance does not claim that a language-scale corpus has been ingested, that a production Large Grammar Model has been trained, or that language benchmarks have been passed.

## Deferred scope

- CUDA-native forward/backward/optimizer kernels and physical GPU evidence;
- automatic long-horizon BPTT and activation checkpoint planning;
- distributed training;
- mixed precision;
- application-specific LGM corpus construction, training, and evaluation.

## Final status

```text
GEO_FULL_CYCLE_V8_CPU: PASS
```
