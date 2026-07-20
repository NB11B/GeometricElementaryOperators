# V5.1 Acceptance Record

Status: **accepted**.

Validated implementation commit: `aeee0d18e2f51a7a24b85b80add3d6f07386456c`.

Host acceptance completed with 49 tests, 25 signatures, 1,528 fixed-blade cases, 50 sparse cases, 50 matrices, and 50 independent certificates.

CUDA acceptance completed on an NVIDIA GeForce RTX 5070 Laptop GPU with compute capability 12.0, CUDA 13.1.80, and MSVC 19.44.35222.0.

The CUDA matrix executed 100 operator cases with 1,024 assignments per case, for 102,400 total comparisons and zero mismatches.

Final markers:

```text
GEO_OPERATOR_V5_1_HOST_GATE: PASS
GEO_OPERATOR_V5_1_CUDA: PASS dimensions=2-6 signatures=25 cases=100
GEO_OPERATOR_V5_1,status=complete
```

Evidence:

```text
benchmarks/geo_operator_kernel/evidence/operator-20260718-123214
benchmarks/geo_operator_kernel/evidence/operator-20260718-123214.zip
benchmarks/geo_operator_kernel/evidence/operator-20260718-123214.zip.sha256.txt
```

No paper update is part of this milestone. V5.2, V5.3, and V6.0 remain deferred.
