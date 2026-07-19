# GEO Native Gradients V1 Acceptance

## Accepted code head

`7c26684734bd4de9b27bae277df9d13becca3efc`

The later acceptance-record commit changes documentation only.

## Hosted validation

Two pull-request workflows executed successfully against the accepted code head.

### GEO native gradients V1 — run 29707299447

- GCC native gradient configure/build/test: PASS;
- Clang native gradient configure/build/test: PASS;
- GCC AddressSanitizer and UndefinedBehaviorSanitizer configure/build/test: PASS;
- existing `operator_kernel` and new `operator_gradients` CTest targets both executed in every job.

### Geometric operator kernel V5.1 — run 29707299419

The complete pre-existing V5.1 host acceptance workflow also passed:

- Python tests: PASS;
- operator pipeline: PASS;
- independent certificates: PASS;
- C kernel configure/build/test: PASS;
- evidence upload: PASS.

This confirms that the native-gradient additions preserve the accepted V5.1 operator pipeline and its independent certificate gate.

## Gradient test marker

```text
GEO_OPERATOR_GRADIENT_TEST: PASS fixed_vjp=1528 sparse_vjp=50 parametric_vjp=50 gp_signatures=27 native_training=PASS
```

## Accepted coverage

- fixed-plan VJP: 1,528 exhaustive fixed-blade cases;
- sparse-plan VJP: 50 cases;
- parametric sparse VJP: 50 cases;
- generic geometric-product JVP/VJP: all 27 canonical non-degenerate signatures across dimensions 1 through 6;
- native training: complete Cl(2,1) multivector weight recovered without external autograd recomputation;
- invalid metric and invalid alias rejection: transactional outputs preserved.

## Methodological boundary

No finite-difference gradient is used as the acceptance oracle. Gradient correctness is checked through exact analytic JVP/VJP adjoint identities over the stored coefficient vectors.

The accepted implementation is the native C/f64 gradient foundation. A framework binding, general expression tape, optimizer, tuned CUDA backward kernel, and physical CUDA gradient evidence remain subsequent integration stages.

## Final status

```text
GEO_NATIVE_GRADIENTS_V1: PASS
```
