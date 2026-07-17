# Fixed GEB-36 typed numerical envelope

Status: verified for the independently reviewed v0.19 Stage 3 implementation.

## Runtime and build contract

`GEO_FIXED_FRACTION_BITS` is a CMake cache setting constrained to 1 through 30
for signed 32-bit storage. CMake publishes the selected value through the
kernel so the library, tests, and report executable compile against one Q
format.

`geo_fixed_geb36_execute` covers the frozen 36-target interface and returns an
authoritative typed result. The manifest contains 26 Cl(2,0), five scalar, four
projective, and one unipotent result. Executor working values are initialized,
and results are staged in locals before successful assignment. Safety
regressions cover invalid target IDs, checked addition overflow, reversion of
`INT32_MIN`, and null output; failure must leave a non-null destination byte-for-
byte unchanged.

## Numerical envelope

`bench_numerical` constructs deterministic fixtures directly as raw Q-format
integers, then decodes those exact values for the floating GEB reference. This
avoids comparing fixed execution against the pre-quantization values that it
never received. Fixture ranges are Q-aware. The Q1 path retains both signs and
usable vector relationships despite its very small representable set.

Before the requested run, the executable replays 64 fixtures from the requested
seed and requires positive and negative components, distinct general and vector
operands, a mixed-sign general multivector, and nonparallel vectors. A seed that
does not satisfy this diversity contract is rejected.

Every target has an explicit LSB budget. Exact constructors, involutions,
projections, addition/subtraction, dual, parity projection, and unipotent
translation use zero LSB. Rounded operations use one to four LSB according to
their arithmetic depth; vector inverse projective uses a documented 48-LSB
envelope because denominator rounding is amplified by division. Projective
values are normalized componentwise in double precision. The Q30 matrix uses a
double floating reference intentionally so one fixed-point LSB remains
observable.

`tools/numerical_report.py` enforces the exact 36 target/name/kind manifest and
an executable-reported precision, Q-format, sample-count, and seed handshake.
It rejects missing, extra, duplicate, or mistyped rows; malformed or incomplete
accounting; non-finite metrics; any overflow, status failure, kind failure, or
mismatch; and any operation that does not complete every bounded sample.

## Validation

- The Python reporter contract suite passed 9/9 tests.
- Clang ASan/UBSan Q1, Q8, Q16, Q24, and Q30 runs each completed 1,024 samples
  for all 36 targets with zero failure counts. The executor ran all five Q
  formats; root independently repeated the original five-Q matrix and repeated
  Q1 and Q30 after review corrections.
- MSVC Release float and double each passed 33/33 non-generator CTests, compiled
  `fixed_geb36` cleanly under `/W4`, and completed a strict Q16 report with
  10,000 samples for each of 36 targets (360,000 total) and zero failure counts.
- An executor GCC Release double build passed 34/34 CTests and its
  `numerical_report` target.
- The initial independent review returned **FAIL** for constant-sign/Q1 corpus
  degeneracy and trusted precision metadata. After Q-aware fixture/diversity
  and compiled-precision corrections, the independent reviewer returned
  **PASS**.

## Limitations

- The floating GEB reference shares low-level primitives with fixed/compiled
  execution and is not a wholly independent oracle.
- Q30 LSB accuracy is intentionally assessed with a double reference; this is
  not a float-reference Q30 claim.
- The Windows `generators` CTest was excluded locally; the portable generator
  suite passed previously under WSL.
- Hosted GitHub Actions and Compute Sanitizer remain unavailable or unpassed
  and are not counted as Stage 3 validation.

## Provenance

- `CMakeLists.txt`
- `include/geo/fixed_geb36.h`
- `src/fixed_geb36.c`
- `benchmarks/bench_numerical.c`
- `tools/numerical_report.py`
- `tests/test_fixed_geb36.c`
- `tests/test_numerical_report.py`
- `.github/workflows/ci.yml`
- `docs/v0.19-validation.md`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
