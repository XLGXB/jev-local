# Historical experiment: laya-full-option-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-27T03:32:27.019Z (field `recorded_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Laya multilingual. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

Original Laya multilingual initializer; head/type_emb/scorer only; encoder/action head/temperature frozen; AdamW LR 5e-5, weight decay 0.01; seed 42; microbatch 1, accumulation 8; FP32, no AMP/TF32; full encoding lengths 1536/320/64.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,124 rows / atoms (452 F08 + 672 replay).
- F08 development: 85 atoms.
- Retention: 168 atoms (72 CI + 96 generic).
- Actual F08 consumer cohort: 48 requests. It is not the development/retention denominator.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **564**.
- Actual exports: **1**.
- Original verdict: `terminal_encoding_failure_after_training_and_reload`.

564 updates and one export were recorded. Reload: 253 records, maximum absolute logit delta 0. Compact terminal summary records a separate read-only evaluation with 48 consumer reports; complete semantic totals are not restated here. Candidate qualification UNQUALIFIED.

## Limits and failure evidence

The native process failed during encoding after training and reload. Later read-only evaluation is not a successful native terminal run and not new training. No omitted semantic count is treated as zero.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/laya-full-option-sft-01/root-state.json` — SHA-256 `86aef4293cf7da5b7add85c8bd5c2d12767f8393ed270161003c0f7b32c20e6f`.
- `runtime/laya-full-option-sft-01/contract.md` — SHA-256 `6d582937989945444916d4169911f8432f35aa672d9159839b84fdf40743821d`.

[Historical experiment index](README.md)
