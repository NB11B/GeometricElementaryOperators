# GEO Full-Cycle Runtime V8

V8 removes the fixed single-model boundary from the earlier `geo_grad` tool. It provides a dynamically sized, named GEO computation graph with native forward execution, native reverse differentiation, minibatch gradient accumulation, optimizer state, recurrent state, constrained parameter updates, checkpointing, prediction, and C export.

## Full cycle

```text
JSON GEO model
-> deterministic GEO IR
-> native graph construction
-> forward inference
-> scalar loss
-> native reverse traversal
-> accumulated parameter gradients
-> SGD or Adam update
-> optional constraint projection
-> checkpoint
-> continued training or inference
```

No external autograd system or machine-learning framework is used.

## Components

- `include/geo/full_cycle_v8.h`: public dynamic graph ABI (`0x00080000`).
- `src/full_cycle_v8.c`: forward, reverse, optimizer, constraints, and recurrent-state runtime.
- `tools/geo_model.py`: strict JSON model validator and deterministic IR compiler.
- `geo_cycle`: streaming training, resume, prediction, checkpoint, and C-export runner.

## Registered differentiable operations

- addition and scalar scaling;
- geometric product;
- reversion;
- grade projection;
- grade involution;
- Clifford conjugation;
- coefficientwise Hadamard product;
- coefficientwise `tanh` and sigmoid;
- Euclidean normalization;
- half squared coefficient norm.

Every registered operation has a native forward and native adjoint rule. Unknown operations fail model validation; there is no fallback.

## Multiple parameters, layers, and large graphs

The graph grows dynamically rather than stopping at the V7 128-node bound. Parameters, inputs, targets, constants, and state leaves are named and may be reused throughout arbitrary directed acyclic graphs. The acceptance suite executes a graph with more than 600 nodes, while the JSON compiler and runner test a separately generated graph with more than 300 nodes.

This is the architecture needed to construct larger geometric grammar stacks. It is not by itself a trained Large Grammar Model or proof of language-scale quality.

## Batching and streaming

`geo_cycle` reads CSV data one row at a time. A training row contains every input multivector followed by every target multivector. The runtime accumulates native gradients across `batch_size` samples and applies their mean in one optimizer step. Dataset size is not stored in memory by the runner.

For stateful models, set `dataset.reset_column=true`. The first CSV value is then `1` to reset model state before a sequence or `0` to continue the current sequence.

## Stateful models

A state leaf is bound to one graph node through `state_updates`. After a forward pass, `geo_v8_commit_states` copies all state updates simultaneously. Checkpoints preserve current state.

The current runner performs **one-step truncated recurrent differentiation**: gradients flow through the current graph evaluation but not backward through an unbounded history of previously committed states. Full backpropagation through time can be represented by explicitly unrolling time steps in a larger acyclic graph; an automatic BPTT planner is not claimed in V8.

## Constraints

Trainable parameters may declare:

- `unit_euclidean`;
- `unit_vector_metric`;
- `even_versor`.

Updates are transactional. All candidate parameter values and optimizer moments are computed first, constraints are checked and projected, and only then are accepted values committed. A failed projection preserves the previous parameter state.

`even_versor` projects odd grades out and normalizes a candidate only when its product with its reverse is scalar within the declared numerical tolerance. This is a practical constrained update contract, not a general manifold optimizer for every rotor group.

## Commands

```bash
python tools/geo_model.py validate model.json
python tools/geo_model.py compile model.json model.geoir

geo_cycle check model.geoir
geo_cycle train model.geoir train.csv checkpoint
geo_cycle train model.geoir train.csv resumed-checkpoint checkpoint
g eo_cycle predict model.geoir checkpoint input.csv output.csv
geo_cycle export-c model.geoir checkpoint model.h model_symbol
```

The extra space in `g eo_cycle` above should not be used; the actual command is `geo_cycle predict`. It is shown separately here to avoid accidental command execution in rendered documentation.

## Examples

- `batched_gp.json`: batched multivector operator learning.
- `recurrent_grammar_cell.json`: a stateful geometric grammar cell.
- `lgm_stack.json`: a multi-input, multi-parameter geometric grammar stack demonstrating the scalable architecture.

## Claim boundary

V8 covers the non-CUDA CPU full-cycle architecture: arbitrary registered graphs, multiple parameters and layers, native adjoints, streaming minibatches, stateful one-step recurrence, constrained parameters, checkpoint/resume, prediction, and C export.

Deferred work includes CUDA kernels and physical GPU evidence, automatic long-horizon BPTT/checkpoint scheduling, distributed execution, mixed precision, and an application-specific trained LGM corpus and evaluation program.
