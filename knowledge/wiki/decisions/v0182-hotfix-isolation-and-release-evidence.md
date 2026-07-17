# Isolate v0.18.2 and require exact-head release evidence

## Decision

Status: accepted and verified on 2026-07-16.

The v0.18.2 release-hardening branch contains only compatibility, correctness, validation, and release-evidence changes. CUDA and broader v0.19 backend work remain outside the hotfix and must be integrated only after the hotfix gate is satisfied.

Local validation and independent review are required candidate evidence. They do not authorize merge or tagging by themselves: GitHub CI must execute against the exact pushed candidate and pass. A workflow run with no checkout, no steps, and no repository logs is classified as an infrastructure startup failure, neither a code failure nor a code pass.

## Rationale

Branch isolation keeps ABI and correctness remediation reviewable without allowing feature work to mask regressions. Exact-head evidence prevents successful results from a different commit from being attributed to the release candidate.

## Consequences

- PR #13 remains draft, unmerged, and untagged while exact pushed-head CI is absent.
- The v0.19 integration branch must be rebased or merged onto the accepted hotfix before its expanded backend gate is evaluated.
- GitHub Actions run #215 is retained as failure-process evidence but cannot satisfy or refute any product acceptance criterion.

## Provenance

- `docs/v0.18.2-release-gate.md`
- `.github/workflows/ci.yml`
- `knowledge/wiki/tasks/TASK-20260716-001.md`
- `knowledge/wiki/failures/github-actions-run-215-pre-checkout.md`
