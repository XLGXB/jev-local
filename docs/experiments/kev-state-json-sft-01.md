# Historical experiment: kev-state-json-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-24T23:03:33.433Z (field `created_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; state representation jevlocal.state-json.v1.

- Planned epochs: 2. An incomplete run did not necessarily finish these epochs.
- Train: 384 rows; atoms: not recorded in the reviewed contract summary.
- Development: 96 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **96**.
- Actual exports: **2**.
- Original verdict: `accepted_observed_synthetic_development_only`.

Synthetic development: baseline 54/96; selected 93/96. Limited observed improvement: true. Eligible for separate final evaluation: true.

## Limits and failure evidence

The original acceptance is synthetic-development-only. Eligibility is permission to design a separate evaluation, not proof that it occurred, general semantic qualification, or default promotion.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-state-json-sft-01/root-verdict.json` — SHA-256 `7ab2aafd881d10b01ca914e60a544e6aaa9e0d81af01afae38dc1a5791222ea1`.
- `runtime/kev-state-json-sft-01/contract.json` — SHA-256 `262593a7dc11596af4c7fa230489bb0eb2de58d2d1e9318e0a8dba6ffda9cad4`.

[Historical experiment index](README.md)
