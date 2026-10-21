# Historical experiment: kev-retention-question-bank-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T11:05:20.267Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; one-epoch design; microbatch 1, accumulation 8; 566 planned updates, only 3 recorded completed updates.

- Planned epochs: 1. An incomplete run did not necessarily finish these epochs.
- Train: 4,528 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **3**.
- Actual exports: **0**.
- Original verdict: `terminal_native_failed_progress_publication`.

No trained export, selected reload, candidate development result, or final quality evaluation was available. Quality status: not_evaluated_incomplete_training.

## Limits and failure evidence

Progress-publication infrastructure failure. The incomplete run is not evidence of semantic rejection or model inefficiency. Planned counts are not actual counts.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-retention-question-bank-sft-01/root-verdict.json` — SHA-256 `6f5ac5f5cc8b0d5c699ac9a41885c3b855f14eea9ecd6a0cc1315abc2900b14d`.
- `runtime/kev-retention-question-bank-sft-01/contract.json` — SHA-256 `6354769a208c9f612107fa55eacb8aca318245caeaa6d3ebbf73c6d61e627812`.

[Historical experiment index](README.md)
