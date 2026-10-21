# Historical experiment: kev-retention-question-bank-sft-02

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T12:08:53.154Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; retention question-bank intervention.

- Planned epochs: 1. An incomplete run did not necessarily finish these epochs.
- Train: 4,528 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **566**.
- Actual exports: **1**.
- Original verdict: `terminal_mechanics_passed_quality_failed`.

Development: 154/168. CI: 72/72. Retention: 82/96. Content-filter retention: 8/16. Original quality gates failed; limited observed improvement and separate-final eligibility: false.

## Limits and failure evidence

Mechanics passed. Do not replace the original retention gates with aggregate accuracy.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-retention-question-bank-sft-02/root-verdict.json` — SHA-256 `133d3b8f7a20babb8e4b2979a83f6fab44822b4a17baa9289d7e125f77d001d9`.
- `runtime/kev-retention-question-bank-sft-02/contract.json` — SHA-256 `1b90a95e54f70f26f9adb6a9250e69d631ec0a8929fbeef9f76f3ce1b9d51084`.

[Historical experiment index](README.md)
