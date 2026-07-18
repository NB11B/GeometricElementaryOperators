# V4.2 Acceptance Record

Status: **physically accepted**

Accepted branch: `research/geometric-identity-engine-v4-2`

Accepted physical evidence directory:

```text
benchmarks/geo_identity_search/evidence/identity-20260718-102312
```

Archive:

```text
benchmarks/geo_identity_search/evidence/identity-20260718-102312.zip
benchmarks/geo_identity_search/evidence/identity-20260718-102312.zip.sha256.txt
```

## Scope

V4.2 validates native fixed-basis-blade constants, signature-general pseudoscalar duality, left/right contraction expansion, contraction-duality relation search, mixed-grade multivectors, relation quotienting, and fixed-blade signed-permutation lowering.

The accepted signature matrix is:

- `Cl(4,0)`
- `Cl(3,1)`
- `Cl(2,2)`
- `Cl(1,3)`
- `Cl(0,4)`

The finite-field statement matrix used primes `65521` and `65519`.

## Host acceptance

- grammar preflight: PASS
- signatures: 5
- grammar relations: 53
- regression and V4.2 tests: 44 PASS
- identity definitions: 160
- control definitions: 20
- generated statements: 360
- exact identity rows: 320
- exact control rows: 40
- normalized relation families: 72
- relation presentation reduction: 108
- fixed-blade lowering reports: 20
- manifest-driven Python evaluator: PASS
- generated host header: PASS
- final marker: `V4_2_ALL: PASS`

## Physical CUDA acceptance

Hardware and toolchain:

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- compute capability: 12.0
- driver: 592.01
- CUDA compiler: 13.1.80
- MSVC: 19.44.35222.0
- CMake: 4.2.1
- Ninja: 1.13.0

Execution result:

- statements: 360
- assignments per statement: 65,536
- CPU checks per statement: 128
- total exact modular assignments: 23,592,960
- 320 identity rows completed without witnesses
- 40 counterexample rows produced exact witnesses
- host smoke: PASS
- dynamic corpus validation: PASS
- summed kernel time: 0.006936 s
- aggregate assignment rate: 3,401,553,878.727 assignments/s
- evidence archive and SHA-256 manifests: generated
- final marker: `GEO_IDENTITY_V4_2,status=complete`

## Interpretation boundary

Characteristic-zero certification comes from exact symbolic polynomial equality. CUDA execution validates the generated finite-field implementation and the declared controls; it does not independently prove arbitrary identities over characteristic zero.

The accepted relations are convention- and signature-audited engine results. Acceptance does not constitute a novelty claim.

## Post-run cleanup correction

The accepted physical run completed before a cleanup-only PowerShell failure. The failure came from using `git ls-files --error-unmatch` under native-command error promotion when the generated header was intentionally untracked. The cleanup now uses non-failing `git ls-files -- <path>` detection, restores tracked headers, and directly removes untracked generated headers. No scientific rerun is required for this correction.
