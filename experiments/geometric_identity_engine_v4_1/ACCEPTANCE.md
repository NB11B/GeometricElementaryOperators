# V4.1 fixed-blade duality acceptance

## Scope

- dimension: 4
- signature: `Cl(2,2)` with diagonal metric `[1, 1, -1, -1]`
- pseudoscalar: `I = e1234`
- exact contract: `I^2 = 1`, therefore `I^-1 = I`
- right dual convention: `dual_r(A) = A I^-1 = A I`
- fixed-blade IR node:

```json
{"fixed_blade":{"blade":15,"coefficient":1}}
```

## Characteristic-zero gate

The selected identity rows were accepted only when the exact blade-wise integer difference polynomial was zero. Counterexample controls were accepted only when the exact difference polynomial was nonzero.

## Host gate

- 37 regression and V4.1 tests: PASS
- identity definitions: 10
- control definitions: 4
- identity rows over two primes: 20
- control rows over two primes: 8
- total statements: 28
- primes: 65521 and 65519
- manifest-driven Python checks: PASS

## Physical host/CUDA gate

- evidence directory: `benchmarks/geo_identity_search/evidence/identity-20260718-050956`
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- compute capability: 12.0
- CUDA toolkit: 13.1
- host compiler: MSVC 19.44
- assignments per statement: 131,072
- total exact modular assignments: 3,670,016
- identity rows without witnesses: 20 of 20
- controls with exact witnesses: 8 of 8
- host smoke: PASS
- dynamic corpus validation: PASS
- summed kernel time: 0.000811 s
- aggregate rate: 4,526,681,429.874 assignments/s
- final marker: `GEO_IDENTITY_V4_1_DUALITY,status=complete`

## Interpretation

The symbolic zero-polynomial result is the characteristic-zero certificate for each selected identity in the declared scope. The host and CUDA runs validate the executable lowering and finite-field behavior. The controls verify that false statements remain distinguishable and produce exact witnesses. This milestone does not claim discovery of a new mathematical identity.

## Known non-blocking issue

An existing discovery test emits a Python `ResourceWarning` for an unclosed CSV reader. It did not change test status or V4.1 acceptance and is tracked for cleanup in V4.2.
