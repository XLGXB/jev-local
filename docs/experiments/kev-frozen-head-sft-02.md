# Historical experiment: kev-frozen-head-sft-02

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-26T10:50:39.597Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA only with frozen initializer heads; head LR 0, LoRA LR 2e-5; AdamW; microbatch 1, accumulation 8; activation checkpointing; 372 trainable tensors / 10,822,656 elements.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,366 rows; atoms: 1,622.
- Oracle development: 84 atoms.
- Original retention development: 168 atoms.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **812**.
- Actual exports: **1**.
- Original verdict: `terminal_frozen_head_ablation_mechanics_passed_quality_rejected_no_retry`.

Oracle development: 81/84. Original development: 152/168. CI: 69/72. Generic retention: 83/96. Content-filter retention: 8/16. quality_gates_pass false; qualification unqualified.

## Limits and failure evidence

Frozen-head ablation mechanics passed, quality rejected. No promotion.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-frozen-head-sft-02/root-verdict.json` — SHA-256 `afcd0be6b06a1113eea5df6debc6849f4788835d3e7fdceffb4b556cd16c4ee1`.
- `runtime/kev-frozen-head-sft-02/contract.json` — SHA-256 `4ef9f76f487491ec8349ce3f1b48fc948d04eac2c0ec0393f5eb6c134045239b`.

[Historical experiment index](README.md)
