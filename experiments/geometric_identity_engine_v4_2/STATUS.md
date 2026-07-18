# V4.2 status

## Implemented

- Native fixed-blade exact backend and compiler.
- Host/CUDA shared-header emission.
- Signature matrix for Cl(4,0), Cl(3,1), Cl(2,2), Cl(1,3), and Cl(0,4).
- Metric-derived pseudoscalar square and inverse.
- Left and right duality.
- Left and right contraction expansions.
- Exact contraction-duality sign and order search with structural-zero exclusion.
- Homogeneous, even, odd, and full multivector supports.
- Structural relation quotienting.
- Signed-permutation lowering for fixed-blade products.
- Grammar preflight, deterministic corpus generation, validation, manifest compilation, tests, host gate, CUDA runner, and orchestration script.

## Acceptance state

V4.1 is merged and accepted. V4.2 is implemented but remains a draft milestone until the local host and CUDA gates pass.

## Fast host smoke

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_2_all.ps1 `
    -ExpectedBranch research/geometric-identity-engine-v4-2 `
    -MaxRelationsPerSignature 8 `
    -PythonChecks 32 `
    -LoweringIterations 500
```

## Full host gate

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_2_all.ps1 `
    -ExpectedBranch research/geometric-identity-engine-v4-2 `
    -MaxRelationsPerSignature 0 `
    -PythonChecks 128 `
    -LoweringIterations 2000
```

## CUDA gate

```powershell
& .\benchmarks\geo_identity_search\scripts\run_identity_v4_2_cuda_clean.ps1 `
    -ExpectedBranch research/geometric-identity-engine-v4-2 `
    -Device 0 `
    -Assignments 65536 `
    -CpuChecks 128 `
    -MaxRelationsPerSignature 0 `
    -Archive
```

Expected final markers include `V4_2_HOST_GATE: PASS` and `GEO_IDENTITY_V4_2,status=complete`.

The technical paper is intentionally deferred.
