# External validation blockers for the v0.18.2/v0.19 sequence

## Compute Sanitizer attach timeout

Status: unresolved environmental/tool failure.

Compute Sanitizer 2025.4 launched both the CUDA test and a one-kernel probe but
timed out before attaching. It reported `No attachable process found` and could
not obtain an exit code. Options including a longer launch timeout, forced
blocking launches, and a one-launch limit did not resolve the failure.

This is not evidence of a kernel sanitizer failure, but it is also not a passed
sanitizer run. CUDA Release float/double correctness tests remain valid local
evidence; sanitizer coverage remains outstanding.

## GitHub Actions pre-checkout billing gate

Status: unresolved external-system failure.

Exact-head run `29550381265` for the v0.18.2 candidate failed every job before
checkout. GitHub returned empty step lists and reported that jobs were not
started because of account payment or spending-limit state. Repository code did
not execute, so the run cannot be used as negative or positive code evidence.

PR #13 was later merged into `origin/main` as `0280f0d`, but its merge message
explicitly defers release tagging because hosted validation never executed.
The code is merged; hosted validation and a v0.18.2 release tag remain
outstanding until the account gate is cleared.

## Provenance

- `knowledge/wiki/tasks/TASK-20260716-001.md`
- `.github/workflows/ci.yml`
