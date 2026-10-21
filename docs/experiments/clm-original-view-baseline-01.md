# Historical evaluation: clm-original-view-baseline-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Scope and dates

- Kind: **evaluation-only baseline; not counted as an SFT stage**.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-27T09:07:13.471Z (field `recorded_at`).
- Training start/end, optimizer-update count, and export count: not recorded in the reviewed evaluation summary; represented as null, not zero, in the manifest.
- Model/configuration overview: CLM original-view baseline. Exact model variant and provider configuration are not restated without a reviewed supporting summary.

## Recorded result

- 253 completed decisions: 85 F08 development + 168 retention.
- Development: **31/85** correct.
- Retention: **76/168** correct.
- Confidence threshold 0.95: **14 accepted; 5 errors**.
- Original stage status: `terminal_unqualified_quality_reviewed`.
- Original qualification: `false`.
- Confidence gate and original quality gates: failed.

This report does not establish training, final held-out qualification, or a successful model release. Cohort counts are not actual F08 consumer counts. Its sampled PDH dedicated-memory measurement must not be directly ranked against another run's CUDA allocator peak.

## Local evidence fingerprint

Repository-relative local source, **not published with this report**:

- `runtime/clm-original-view-baseline-01/root-state.json` — SHA-256 `5dae787915b9ec4d44e1e1de59ad61a4468474314841a905b6e3436dc1b4d4b0`.

No raw examples, per-sample predictions, weights, logs, account paths, or sealed-partition data are included.

[Historical experiment index](README.md)
