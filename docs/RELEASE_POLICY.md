# Release and compatibility policy

This policy defines the compatibility contract for the portable C kernel, fixed-point backend, CUDA backend, embedded adapters, generated schedules, and RTL artifacts.

## Versioning

The project uses semantic versioning for tagged releases:

- **major**: intentional source or ABI break, incompatible numerical contract, or incompatible generated-artifact schema;
- **minor**: backward-compatible public API additions, new backends, new targets, or compatible schedule/schema extensions;
- **patch**: correctness, safety, documentation, test, or performance fixes that preserve the published contract.

A release tag must identify one immutable commit on `main`. Release notes must list the merge commit, supported toolchains, known hardware used for validation, and the exact generated artifacts attached to the release.

## Stable public C surface

Headers under `include/geo` are public unless a header explicitly states otherwise. Public function names and public structure prefixes are ABI-sensitive.

The following rules apply within a major release:

- existing public functions retain their symbol, calling convention, argument order, and return semantics;
- existing enum values retain their numeric value;
- existing public structure fields are not reordered or removed;
- new structure fields may only be appended when the structure is part of a published ABI;
- implementation entry points with an `_impl` suffix do not replace the established public symbol;
- a deprecated symbol remains available for at least one minor release and is identified in release notes and headers.

The build must include a link-level regression that calls established public executor symbols. Release validation should additionally inspect the built archive or shared object with `nm`, `dumpbin`, or an equivalent symbol tool.

## Precision and numerical behavior

`geo_real_t` is selected at compile time:

- `GEO_USE_DOUBLE=ON`: IEEE-754 `double` where the target toolchain provides it;
- `GEO_USE_DOUBLE=OFF`: IEEE-754 `float` where the target toolchain provides it.

Floating-point equivalence is tolerance-based. Every benchmark result must be paired with the tolerance, maximum absolute error, maximum relative error, and mismatch count for the same fixtures.

The signed 32-bit fixed backend supports `GEO_FIXED_FRACTION_BITS` from 1 through 30. Its published arithmetic contract is:

- multiplication and division round half away from zero;
- an unrepresentable intermediate or final result returns an overflow status;
- divide by zero returns an explicit status;
- checked APIs do not silently saturate or truncate;
- a failed program instruction leaves its destination register unchanged;
- fixed C and generated RTL use the same instruction order, rounding rule, and overflow rule;
- an RTL result marked overflow is invalid and its data outputs are driven to zero.

Changing any of these fixed/RTL rules requires a major release unless an earlier implementation violated the documented contract and the change is explicitly classified as a corrective patch.

## Supported build classes

Every release candidate must preserve these independent build classes:

1. portable C11 without CUDA;
2. portable C11 with `geo_real_t` as `float`;
3. portable C11 with `geo_real_t` as `double`;
4. sanitizer builds using AddressSanitizer and UndefinedBehaviorSanitizer where supported;
5. ESP-IDF component build;
6. supported Q-format validation builds;
7. generated RTL simulation and synthesis;
8. optional CUDA build when `GEO_BUILD_CUDA=ON`.

CUDA is optional. Enabling or disabling it must not change the portable C ABI. The v0.19 CUDA backend requires CUDA Toolkit 13.x and a CMake version new enough to support the selected CUDA language standard.

## Toolchain evidence

Release notes must record the actual compiler/tool versions used. The expected validation matrix includes:

- GCC and Clang host builds;
- float and double configurations;
- ASan/UBSan;
- Q1, Q8, Q16, Q24, and Q30 fractional-bit configurations;
- ESP-IDF component validation and, for hardware claims, a named ESP32-S3 board and ESP-IDF version;
- Icarus Verilog simulation and Yosys synthesis for generated RTL;
- CUDA Toolkit 13.x compile validation and, for GPU performance/correctness claims, a named GPU, driver, toolkit update, and Compute Sanitizer result;
- Cortex-M hardware claims accompanied by MCU, core clock, CMSIS/vendor SDK version, and DWT availability.

A compile-only environment is not evidence of device execution. Hardware-specific release claims must clearly distinguish compile validation from physical-device validation.

## Generated artifacts

Generated C, JSON schedules, SystemVerilog, testbenches, and manifests must be reproducible from checked-in generators and checked-in input specifications.

A release that attaches generated artifacts must include:

- the generator command;
- the source commit;
- width/fraction parameters;
- input schedule filenames;
- a manifest or checksums;
- simulation and synthesis results when RTL is included.

Hand-editing generated release artifacts without updating the generator is not permitted.

## Benchmarks

Benchmark labels must describe only the path actually measured. Terms such as “complete,” “generated,” or “all backends” may be used only when the corresponding paths are present in that run.

Published results must include:

- fixture seed and input domain;
- warmup and repetition counts;
- precision and Q-format;
- operation and backend;
- batch size where applicable;
- minimum, median/P50, P95, P99, and maximum timing or enough raw samples to reproduce them;
- correctness/error statistics;
- transfer-inclusive and kernel-only CUDA timing as separate values;
- device, compiler, clock, and build-mode metadata.

Anti-optimization sinks must use defined arithmetic. Benchmark code is part of the sanitizer and undefined-behavior release gate.

## Release gate

A release candidate may be tagged only when:

- required CI and local tests pass on the exact candidate commit;
- no known release-blocking correctness, ABI, undefined-behavior, or overflow mismatch remains;
- public symbols expected by the previous compatible release are present;
- generated artifacts are reproducible;
- release notes distinguish tested configurations from untested configurations;
- hardware results, when claimed, are attached or linked as evidence.

If hosted CI is unavailable because of runner, billing, or service conditions, the release remains untagged until equivalent evidence is produced and reviewed. An empty or unstarted workflow is not a passing result.
