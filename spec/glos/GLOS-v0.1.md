# GEO/LGM Structured Operator Specification (GLOS)

**Version:** 0.1.0  
**Status:** Engineering draft integrated for GEO tensor-runtime work

This directory adopts the GEO/LGM Structured Operator Specification as the semantic contract for portable, typed, differentiable, certifiable, and hardware-lowerable operators.

The normative source reviewed for this integration defines each operator as a first-class object declaring:

- typed inputs and outputs;
- governing algebra;
- parameters and constraints;
- forward execution semantics;
- differentiation mode and ownership;
- saved-state policy;
- numerical behavior;
- symmetry or transformation behavior;
- required tests and certificates;
- explicit hardware lowerings and fallback policy.

## Integration rule for the current GEO runtime

The executable C/CUDA ABI remains the source of truth for implementation behavior. GLOS descriptors document that behavior and are validated against tests; they do not authorize a backend substitution.

For the current deep-learning tensor operators:

- `gradient_status: native` is required when GEO implements the VJP directly;
- `gradient_status: bridge` is reserved for an explicitly labeled reference-recompute implementation;
- a bridge descriptor MUST NOT be used to claim native GEO backward execution;
- fallback is `forbidden` for production GEOSDP training;
- CPU/CUDA parity and gradient certificates remain required before release.

## Initial conformance target

The first live descriptor is `operators/geo_tensor_linear.operator.yaml`. It describes the existing GEO tensor-linear forward and exact native VJP surface rather than the bridge-gradient example from the draft specification.

The full upstream engineering draft should be retained as the normative design reference when the operator registry, schema validator, certificate format, algebra registry, and Sidequest/LGM packaging layers are split into their own repository or stabilized in GEO.
