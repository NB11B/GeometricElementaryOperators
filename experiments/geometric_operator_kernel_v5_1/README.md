# Geometric Operator Kernel V5.1

This milestone converts the accepted identity engine into a reusable operator compiler and a bounded C kernel. It covers the planned V4.3 through V5.1 sequence without entering RTL, analog mapping, or the Large Grammar Model runtime.

## Milestone map

### V4.3 — executable fixed-blade specialization

For a fixed blade `J`, left or right multiplication is compiled to a signed permutation:

```text
source blade a -> target blade a xor J
factor = coefficient * gp_sign(a, J)
```

The acceptance pipeline compares the generic geometric product, the specialized contribution plan, and the extracted matrix for every basis blade, both sides, dimensions 2 through 6, and every canonical `(p,q)` signature class.

### V4.4 — sparse fixed multivectors

The IR accepts a canonical sparse constant:

```json
{
  "fixed_multivector": {
    "terms": [
      {"blade": 0, "coefficient": 1},
      {"blade": 3, "coefficient": -2},
      {"blade": 15, "coefficient": 1}
    ]
  }
}
```

Duplicate blades are combined, zero terms are removed, and an empty result is rejected. Multiplication lowers to a sum of signed permutations.

### V4.5 — linear operator extraction

Every fixed left or right product is emitted as a coefficient-space matrix. The analyzer records:

- sparsity and density;
- row and column occupancy;
- monomial/permutation structure;
- determinant for monomial matrices;
- rank and invertibility modulo 65521;
- parity preservation;
- grade-transfer maps;
- deterministic matrix hashes.

### V4.6 — dimension-general engine

The native path covers dimensions 2 through 6 and all canonical non-degenerate diagonal signature classes. The pipeline derives `I^2`, `I^-1`, dual-square behavior, and sparse-constant expansion from the metric rather than hard-coding dimension four.

### V4.7 — theorem schemas

Concrete exact instances are grouped into parameterized schemas carrying dimension, `(p,q)`, `I^2`, sign, family, and source classification.

### V4.8 — independent proof certificates

Each exact theorem instance receives a standalone certificate containing the normalized specification, difference polynomial, symbols, hashes, and final zero/nonzero claim. `tools/geo_operator_certificate_verify.py` imports no discovery, compiler, or V5.1 engine module and independently reconstructs the sparse integer polynomial.

### V5.0 — stable C operator kernel

Public ABI:

```text
include/geo/operator_kernel.h
src/operator_kernel.c
```

The kernel provides:

- generic geometric product;
- specialized fixed-blade and sparse-fixed-multivector execution;
- dense extracted-matrix execution;
- exact modular `int32_t` mode;
- configurable fixed-point mode;
- `geo_real_t` mode;
- alias-safe outputs;
- no internal heap allocation.

### V5.1 — embedded and edge profile

`include/geo/operator_embedded.h` exposes compile-time limits and confirms:

- maximum dimension 6;
- maximum 64 basis blades;
- fixed caller-owned plans;
- constant plan storage suitable for normal MCU flash/RODATA mapping;
- no runtime parser;
- no dynamic allocation;
- deterministic contribution bounds;
- fixed stack/storage bounds.

## Host acceptance

```powershell
git switch research/geometric-operator-kernel-v5-1-acceptance
git pull --ff-only origin research/geometric-operator-kernel-v5-1-acceptance

& .\benchmarks\geo_operator_kernel\scripts\run_geo_operator_v5_1_host_gate.ps1 `
    -ExpectedBranch research/geometric-operator-kernel-v5-1-acceptance
```

Expected final markers:

```text
V4_3_SPECIALIZATION: PASS
V4_4_FIXED_MULTIVECTOR: PASS
V4_5_LINEAR_OPERATOR: PASS
V4_6_DIMENSION_MATRIX: PASS
V4_7_THEOREM_SCHEMAS: PASS
V4_8_CERTIFICATES: PASS
V5_0_KERNEL: PASS
V5_1_EMBEDDED: PASS
GEO_OPERATOR_V5_1_PIPELINE: PASS
GEO_OPERATOR_KERNEL_TEST: PASS
GEO_OPERATOR_V5_1_HOST_GATE: PASS
```

## Physical CUDA acceptance

```powershell
& .\benchmarks\geo_operator_kernel\scripts\run_geo_operator_v5_1_cuda_clean.ps1 `
    -ExpectedBranch research/geometric-operator-kernel-v5-1-acceptance `
    -Device 0 `
    -Assignments 1024 `
    -Archive
```

Expected final marker:

```text
GEO_OPERATOR_V5_1,status=complete
```

## Evidence boundary

The exact polynomial backend certifies the declared integer identities. Host and CUDA gates validate implementation equivalence over deterministic finite-field assignments. Timing data is implementation evidence, not a universal performance claim. V5.1 does not include FPGA/RTL generation, ASIC or analog mapping, or a Large Grammar Model runtime.
