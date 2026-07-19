# GEO Native Gradients V1

## Purpose

This milestone removes the need to recompute GEO forward expressions in an external autograd system merely to obtain a backward pass. The operator kernel now exposes native forward-mode and reverse-mode derivative primitives whose signs, operand order, signature, side of action, and blade routing are computed by the same GEO semantics as the forward kernel.

The gradient contract is defined over the ordinary Euclidean inner product of the stored coefficient vector. This is the contract expected when a multivector is represented as a tensor of blade coefficients in a machine-learning runtime.

## Generic geometric product

For

```text
Y_c = sum_(a xor b = c) s(a,b;g) L_a R_b,
```

the native reverse-mode vector-Jacobian product is

```text
bar_L_a = sum_b bar_Y_(a xor b) s(a,b;g) R_b
bar_R_b = sum_a bar_Y_(a xor b) s(a,b;g) L_a.
```

The native forward-mode Jacobian-vector product is

```text
dY = dL R + L dR.
```

Public entry points:

```c
geo_operator_status_t geo_operator_gp_f64_jvp(...);
geo_operator_status_t geo_operator_gp_f64_vjp(...);
```

## Fixed and sparse operator plans

A fixed contribution row has the form

```text
y_(a xor J) += x_a s(a,J;g) c_J
```

for right action, with the sign arguments reversed for left action. Its input VJP is the exact transpose route

```text
bar_x_a += bar_y_(a xor J) s(a,J;g) c_J.
```

Public entry point:

```c
geo_operator_status_t geo_operator_apply_f64_vjp(...);
```

## Trainable sparse geometric operators

`geo_operator_apply_parametric_f64` reuses a validated GEO plan for dimension, signature, side, and blade topology while accepting caller-owned `double` parameters. Parameter `i` is the real coefficient of the corresponding planned blade.

The matching VJP returns both input and parameter cotangents:

```text
bar_x_a     += bar_y_(a xor J_i) s_i theta_i
bar_theta_i += bar_y_(a xor J_i) s_i x_a.
```

Public entry points:

```c
geo_operator_status_t geo_operator_apply_parametric_f64(...);
geo_operator_status_t geo_operator_apply_parametric_f64_vjp(...);
```

Initializing the real parameter array from the integer plan coefficients reproduces `geo_operator_apply_f64` exactly. Parameters can then be updated continuously by an external optimizer without changing the certified blade topology.

## Validation method

The tests do not use finite-difference gradients. They verify the exact adjoint identity

```text
<bar_y, J delta_x> = <J^T bar_y, delta_x>
```

and, for parametric operators and the bilinear geometric product,

```text
<bar_y, J(delta_x, delta_theta)>
  = <bar_x, delta_x> + <bar_theta, delta_theta>.
```

The deterministic training test uses all basis blades as inputs, learns a complete Cl(2,1) multivector weight through the native geometric-product VJP, and reaches the exact target coefficient vector in one analytically chosen gradient step.

## Current coverage

- 1,528 fixed-blade VJP cases across dimensions 2 through 6, all 25 canonical non-degenerate signatures, every blade, and both sides;
- 50 sparse-plan VJP cases across the same signature matrix and both sides;
- 50 trainable parametric VJP cases;
- generic geometric-product JVP/VJP duality across dimensions 1 through 6 and all 27 canonical signatures in that range;
- integer-plan versus parametric-forward equality at initialization;
- transactional rejection tests for invalid metrics and invalid output aliasing;
- one complete native-gradient training step with no external autograd recomputation.

## Local validation performed

The isolated operator source and gradient test were compiled and executed with:

- GCC 14.2.0, release warnings-as-errors: PASS;
- Clang 17.0.0, release warnings-as-errors: PASS;
- GCC AddressSanitizer plus UndefinedBehaviorSanitizer: PASS.

Expected terminal marker:

```text
GEO_OPERATOR_GRADIENT_TEST: PASS fixed_vjp=1528 sparse_vjp=50 parametric_vjp=50 gp_signatures=27 native_training=PASS
```

The pull-request workflow also builds and runs the existing operator-kernel test beside the new gradient test under GCC, Clang, ASan, and UBSan.

## Claim boundary

This V1 milestone supplies exact native derivative primitives for the public `f64` fixed-operator and geometric-product paths. It is not yet a general expression-tape engine, an optimizer, a PyTorch/JAX binding, or a physical CUDA gradient acceptance result. Modular and Q-format execution remain forward/exact-inference domains; a conventional real-valued training phase can be compiled or quantized into those domains after training.
