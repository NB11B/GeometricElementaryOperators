# Geometric Identity Engine V4.2

V4.2 promotes fixed basis-blade constants into a versioned native engine and expands the accepted V4.1 `Cl(2,2)` duality milestone across signatures, contractions, mixed-grade multivectors, structural relation quotienting, and specialized hardware lowering.

## Native engine

The V4.2 path does not install import-time hooks into the accepted legacy modules.

- `tools/geo_identity_v4_2_exact.py` owns validation and exact characteristic-zero sparse polynomial evaluation.
- `tools/geo_identity_v4_2_compiler.py` owns the fixed-blade DAG node, Python finite-field evaluation, and host/CUDA emission.
- `tools/geo_identity_v4_2_engine.py` provides duality, contraction, canonicalization, and signed-permutation lowering macros.
- `tools/geo_identity_v4_1_ir.py` is a compatibility facade over the native V4.2 implementation on this branch.

The fixed-blade node remains:

```json
{"fixed_blade":{"blade":15,"coefficient":1}}
```

## Signature matrix

The default dimension-four matrix is:

| algebra | signature | `I^2` |
|---|---|---:|
| `Cl(4,0)` | `[1,1,1,1]` | `+1` |
| `Cl(3,1)` | `[1,1,1,-1]` | `-1` |
| `Cl(2,2)` | `[1,1,-1,-1]` | `+1` |
| `Cl(1,3)` | `[1,-1,-1,-1]` | `-1` |
| `Cl(0,4)` | `[-1,-1,-1,-1]` | `+1` |

For a non-degenerate diagonal signature,

```text
I^2 = (-1)^(n(n-1)/2) product_i g_i
I^-1 = I^2 I
```

The corpus therefore tests metric-dependent dual squares rather than assuming `I^-1 = I` globally.

## Variable supports

- `v`: grade 1
- `B`: grade 2
- `T`: grade 3
- `E`: grades 0, 2, 4
- `O`: grades 1, 3
- `M`: grades 0 through 4

This separates identities that require homogeneous inputs from identities that survive bilinear extension to even, odd, or full multivectors.

## Contraction conventions

```text
A_r lcontract B_s = <reverse(A_r) B_s>_(s-r), r <= s
A_r rcontract B_s = <A_r reverse(B_s)>_(r-s), r >= s
```

Both extend bilinearly by grade decomposition. The corpus includes exact contraction-plus-duality round trips and searches candidate wedge/contraction dual formulas over both signs. Only exact zero-polynomial candidates are admitted.

## Structural quotient

`geo_identity_v4_2_relation_audit.py` normalizes relations under:

- side exchange;
- simultaneous sign reversal;
- variable-to-role renaming;
- additive operand ordering;
- pseudoscalar-square class.

Left and right duals remain distinct unless the normalized expressions actually coincide.

## Hardware lowering

Multiplication by a fixed basis blade is lowered to a signed permutation:

```text
source blade a -> target blade a xor J
factor = coefficient * gp_sign(a, J)
```

The lowering tool emits JSON evidence, a Markdown benchmark report, and a C++ header for left/right multiplication by `I` and `I^-1` across the signature matrix. The Python timing is a sanity benchmark, not a hardware performance claim.

## Host gate

```powershell
git switch research/geometric-identity-engine-v4-2
git pull --ff-only origin research/geometric-identity-engine-v4-2

python .\tools\geo_identity_v4_2_grammar_preflight.py `
    --config .\experiments\geometric_identity_engine_v4_2\config.json `
    --output-json .\local-evidence\v4-2\grammar-preflight.json `
    --markdown-out .\local-evidence\v4-2\grammar-preflight.md

& .\benchmarks\geo_identity_search\scripts\run_identity_v4_2_host_gate.ps1 `
    -ExpectedBranch research/geometric-identity-engine-v4-2 `
    -PythonChecks 128
```

Accepted host markers:

```text
V4_2_GRAMMAR_PREFLIGHT: PASS signatures=5 relations=53
44 tests: PASS
V4_2_CORPUS: PASS signatures=5 identity_definitions=160 controls=20 statements=360
V4_2_VALIDATION: PASS statements=360 identities=320 controls=40
V4_2_RELATION_AUDIT: PASS definitions=180 families=72 reduction=108
V4_2_LOWERING: PASS reports=20 blade=15
V4_2_HOST_GATE: PASS
V4_2_ALL: PASS
```

## Physical CUDA gate

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_2_cuda_clean.ps1 `
    -ExpectedBranch research/geometric-identity-engine-v4-2 `
    -Device 0 `
    -Assignments 65536 `
    -CpuChecks 128 `
    -MaxRelationsPerSignature 0 `
    -Archive
```

Accepted evidence:

```text
benchmarks/geo_identity_search/evidence/identity-20260718-102312
benchmarks/geo_identity_search/evidence/identity-20260718-102312.zip
benchmarks/geo_identity_search/evidence/identity-20260718-102312.zip.sha256.txt
```

Accepted physical result:

```text
statements: 360
assignments per statement: 65,536
total exact modular assignments: 23,592,960
identity rows: 320 PASS without witnesses
control rows: 40 PASS with exact witnesses
summed kernel time: 0.006936 s
aggregate rate: 3,401,553,878.727 assignments/s
VALIDATION: PASS
GEO_IDENTITY_V4_2,status=complete
```

The complete acceptance record is in `ACCEPTANCE.md`.

## Evidence boundary

Exact zero integer difference polynomials certify selected identities in the declared dimension, signature, and grade supports. Host/CUDA execution validates the generated finite-field implementation. Counterexample controls must retain nonzero exact polynomials and produce finite-field witnesses. No novelty claim follows from relation survival alone.

The paper is intentionally unchanged in V4.2.
