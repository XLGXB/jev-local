# Historical experiment: kev-coverage-checkpointed-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-26T07:25:43.387Z (field `created_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; activation checkpointing; 203 updates per epoch.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,366 rows; atoms: 1,622.
- Oracle development: 84 atoms.
- Original retention development: 168 atoms.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **812**.
- Actual exports: **1**.
- Original verdict: `completed_native_mechanics_pass_quality_rejected_no_promotion_no_retry`.

Oracle development: baseline 54/84; final 78/84. Original development: 155/168; CI 72/72; generic retention 83/96. Content-filter retention: baseline 14/16; final 8/16. Quality gates false; qualification unqualified.

## Limits and failure evidence

Mechanics passed. The retained receipt also flags a gate-feasibility inconsistency: baseline 22/24 with required +4, but maximum possible gain +2. This historical contract issue is disclosed, not repaired retrospectively and not used to promote the candidate.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-coverage-checkpointed-sft-01/root-verdict.json` — SHA-256 `e6c5fcc4a6a2f18eaf5d6b0bb0afc29b60824fcf3534372c642bf53dedc7b89c`.
- `runtime/kev-coverage-checkpointed-sft-01/contract.json` — SHA-256 `5f1a96c7740858235b51e2c231d82e9100c2bc4671bfd8065b1baba702d3520d`.

[Historical experiment index](README.md)
