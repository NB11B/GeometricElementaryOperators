# GEO Native Autodiff V7 Acceptance

## Accepted code head

`998770aeacc4eb0e5c1c6e7e2bbac3a9a2eea541`

The acceptance-record commit changes documentation only.

## Native capability accepted

V7 owns the complete declared training path internally:

```text
GEO graph compilation
-> GEO forward execution
-> GEO reverse traversal
-> GEO cotangent accumulation
-> GEO SGD or Adam state transition
-> GEO parameter update
```

The implementation contains no external autograd engine, surrogate forward recomputation, finite-difference gradient, or silent fallback.

Accepted differentiable node kinds:

- input;
- trainable multivector parameter;
- constant;
- addition;
- scalar scaling;
- geometric product;
- reversion;
- grade projection;
- half squared coefficient norm.

The selected reverse pairing is the Euclidean pairing over stored blade coefficients. Clifford metric signs, blade routing, and noncommutative operand order remain part of the forward operator and native adjoint rules.

## Hosted validation

### GEO native autodiff V7 — run `29708352916`

All jobs passed against the accepted code head:

- GCC release configure/build/test: PASS;
- Clang release configure/build/test: PASS;
- GCC AddressSanitizer plus UndefinedBehaviorSanitizer: PASS;
- Windows MSVC release configure/build/test: PASS;
- full host build and complete CTest regression: PASS.

The full-host gate initially exposed a stale safety test that attempted to write the removed `geo_optimized_witness_t.terminal_count` field. The test was restored to the established ABI contract, where terminal count is supplied explicitly to `geo_program_fold_constants`. The complete host build and test matrix then passed.

### GEO native gradients V1 — run `29708352923`

The operator-level native JVP/VJP compiler and sanitizer matrix passed unchanged.

### Geometric operator kernel V5.1 — run `29708352908`

The existing V5.1 host gate passed unchanged, including:

- Python regression tests;
- operator pipeline generation;
- independent certificate verification;
- C operator-kernel build and test;
- evidence upload.

## V7 scientific checks

The V7 test validates:

- composed analytic JVP/VJP adjoint identity across all 27 canonical non-degenerate signatures in dimensions one through six;
- exact cotangent accumulation at graph branch joins;
- end-to-end internal SGD training in `Cl(2,1)`, recovering a complete multivector parameter;
- repeated internal Adam training;
- explicit rejection of uncompiled execution, backward-before-forward, optimizer-before-backward, invalid grades, non-scalar losses, post-compile mutation, and non-finite writes;
- transactional preservation of stored values on rejected updates.

No finite-difference gradient is used as the acceptance oracle.

Expected runtime marker:

```text
GEO_V7_NATIVE_AUTODIFF_TEST: PASS signatures=27 branch_accumulation=PASS sgd_training=PASS adam_training=PASS no_external_autograd=TRUE
```

## Claim boundary

V7 establishes a real GEO-owned reverse-mode graph runtime and optimizer for the declared node registry. It does not yet claim registered adjoint rules for every higher-level GEO grammar operation, a tuned CUDA backward backend, dynamic control-flow recording, checkpointing, distributed execution, mixed-precision training, or physical GPU gradient evidence.

Unsupported differentiation paths fail explicitly. There is no external-autograd fallback.

## Final status

```text
GEO_NATIVE_AUTODIFF_V7: PASS
```
