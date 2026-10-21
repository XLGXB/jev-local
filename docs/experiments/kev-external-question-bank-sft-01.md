# Historical experiment: kev-external-question-bank-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T17:59:07.531Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; 455 training groups; admitted external question-bank mixture.

- Planned epochs: 1. An incomplete run did not necessarily finish these epochs.
- Train: 5,890 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **737**.
- Actual exports: **1**.
- Original verdict: `terminal_failed_no_retry`.

Historical partial saved-record summary: 157/168 overall; CI 72/72; retention 85/96. Original gate comparison: limited improvement false, separate-final eligibility false; whole_run_eligible false.

## Limits and failure evidence

Stopped at native public request identity mismatch. Saved-record aggregation was recorded historically; it was not recomputed for this publication. Diagnostic/external-invariance/comparison/final-verification outputs were missing; 47 external forwards remained unrun.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-external-question-bank-sft-01/root-terminal-verdict.json` — SHA-256 `3845562e2f6dde9dd137bb27e99390d2f479977cea175366aa5e2f11281402a6`.
- `runtime/kev-external-question-bank-sft-01/contract.json` — SHA-256 `44a5e47ae2656c7c83c2d505395245d5e2be3c4a217196c351e3f40d257e0d6d`.

[Historical experiment index](README.md)
