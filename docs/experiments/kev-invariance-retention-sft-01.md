# Historical experiment: kev-invariance-retention-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-26T13:10:29.817Z (field `recorded_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA only with frozen initializer heads; head LR 0, LoRA LR 2e-5; AdamW; microbatch 1, accumulation 8; activation checkpointing; invariance views retain original mass.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,366 rows; atoms: 1,622.
- Oracle development: 84 atoms.
- Original retention development: 168 atoms.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **812**.
- Actual exports: **1**.
- Original verdict: `terminal_invariance_mechanics_passed_quality_rejected_no_retry`.

Oracle development: 54 to 81/84. Original development: 138 to 144/168. CI: 69/72. Generic retention: 75/96. Oracle observed improvement true; original-retention improvement false; both separate-final eligibility flags false; qualification unqualified.

## Limits and failure evidence

Mechanics passed, quality rejected; no promotion or repeat authorization.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-invariance-retention-sft-01/root-verdict.json` — SHA-256 `274d2563375ebfcf6890b478de257057dd45ef6a7abbef077d96779fcbe721d2`.
- `runtime/kev-invariance-retention-sft-01/contract.json` — SHA-256 `3b8ad388ac1de691a86c0c3e8953aadcda96901f4b5c30c9140d8e5805d1d87b`.

[Historical experiment index](README.md)
