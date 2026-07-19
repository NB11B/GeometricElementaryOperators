# GEO Native Autodiff V7

## Purpose

V7 makes differentiation a native GEO execution capability rather than an adapter around an external autograd engine.

The runtime owns the complete training path:

```text
GEO typed graph
-> GEO forward execution
-> GEO reverse traversal
-> GEO cotangent accumulation
-> GEO optimizer state transition
-> GEO parameter update
```

No PyTorch, JAX, TensorFlow, framework tape, surrogate forward recomputation, finite-difference gradient, or silent fallback participates in the V7 C runtime or its acceptance test.

## Architectural contract

A `geo_v7_program_t` is a caller-owned, allocation-free, bounded computation graph. Nodes are added in topological order and compiled against one declared dimension, diagonal signature, coefficient domain, and pairing.

The initial pairing contract is the Euclidean pairing over stored blade coefficients:

```text
<X,Y> = sum_A X_A Y_A
```

This makes reverse values explicit coefficient-space cotangents while retaining all Clifford metric and noncommutative signs inside each operator and its adjoint.

The V7 ABI is:

```text
0x00070000
```

The initial graph bound is 128 nodes and dimensions one through six.

## Native node set

V7 supplies forward and reverse rules for:

- inputs;
- trainable multivector parameters;
- constants;
- addition;
- scalar scaling;
- geometric product;
- reversion;
- grade projection;
- half squared coefficient norm.

The geometric-product reverse rule delegates to the accepted GEO-native VJP kernel. Reversion and grade projection are lowered as their exact coefficient-space adjoints. Shared nodes accumulate cotangents at branch joins during the GEO-owned reverse traversal.

Unsupported node kinds fail compilation. A non-scalar selected loss fails backward explicitly.

## Native optimizer ownership

V7 includes allocation-free optimizer state inside parameter nodes:

- SGD;
- Adam with first and second moments, bias correction, and checked finite updates.

Optimizer calls require a successful GEO forward and GEO backward pass. Parameter updates invalidate saved forward and backward state, requiring a new native forward pass before the next backward pass.

## Validation design

The acceptance test checks:

1. **Composed adjoint identity** across all 27 canonical non-degenerate signatures in dimensions one through six for

   ```text
   loss(grade_project(reverse(input * weight)))
   ```

   using an analytic GEO JVP as the independent directional derivative.

2. **Branch accumulation** where one geometric-product result is consumed twice. The shared node must receive the sum of both reverse contributions.

3. **End-to-end native SGD training** in `Cl(2,1)`. A complete multivector parameter is learned from all basis-blade inputs in one analytically chosen optimizer step.

4. **End-to-end native Adam training** with repeated GEO forward, GEO backward, and GEO optimizer steps.

5. **Failure contracts** for uncompiled execution, backward-before-forward, optimizer-before-backward, invalid grade, non-scalar loss, post-compile graph mutation, non-finite values, and transactional rejected writes.

Expected marker:

```text
GEO_V7_NATIVE_AUTODIFF_TEST: PASS signatures=27 branch_accumulation=PASS sgd_training=PASS adam_training=PASS no_external_autograd=TRUE
```

## Claim boundary

V7 establishes a real internal reverse-mode runtime and optimizer for the declared node set. It does not yet claim every GEO grammar operation has a registered adjoint rule, nor does it claim a tuned CUDA backward implementation, dynamic control-flow tape, checkpointing planner, distributed training, mixed-precision training, or physical GPU gradient evidence.

The extension path is explicit: add a typed forward rule and a typed adjoint rule to the V7 node registry, then validate the resulting compiled program through the same analytic adjoint and end-to-end training gates. There is no external-autograd fallback path.
