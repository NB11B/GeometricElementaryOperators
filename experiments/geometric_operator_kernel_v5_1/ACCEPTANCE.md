# V5.1 Acceptance Record

Status: **prepared; local host and physical CUDA evidence pending**.

The acceptance branch contains the V4.3 through V5.1 implementation and gates. This record must only be changed to accepted after both markers are observed from the same branch head:

```text
GEO_OPERATOR_V5_1_HOST_GATE: PASS
GEO_OPERATOR_V5_1,status=complete
```

## Required host matrix

- dimensions: 2 through 6;
- canonical signature classes: 25;
- every fixed blade, both left and right multiplication: 1,528 specialization cases;
- sparse fixed multivector, both sides: 50 cases;
- deterministic exact integer comparisons from the configured iteration matrix;
- independent certificates: 50;
- extracted coefficient-space matrices: 50;
- C host fixed-blade cases: 1,528;
- C host sparse cases: 50;
- real, exact-modular int32, and fixed-Q execution paths;
- embedded limits: dimension 6, 64 blades, 64 fixed terms, no heap, no runtime parser.

## Required physical CUDA matrix

- dimensions: 2 through 6;
- canonical signature classes: 25;
- fixed-pseudoscalar and sparse-fixed-multivector operators;
- both left and right multiplication;
- 100 CUDA operator cases;
- generic and specialized results equal for every configured assignment;
- zero mismatches.

## Required evidence

- pipeline JSON and Markdown artifacts;
- independent certificate manifest and verifier logs;
- C kernel test log;
- host build and test logs;
- CUDA device identity and toolchain versions;
- CUDA generic-versus-specialized comparison log;
- SHA-256 manifest;
- optional archive and archive hash.

No paper update is part of this milestone. V5.2, V5.3, and V6.0 remain explicitly deferred.
