# Historical experiment: kev-mixed-gpu-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-24T21:18:54.148Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8.

- Planned epochs: 2. An incomplete run did not necessarily finish these epochs.
- Train: 384 rows; atoms: not recorded in the reviewed contract summary.
- Development: 96 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **96**.
- Actual exports: **2**.
- Original verdict: `training_mechanics_and_limited_dev_improvement_verified_final_evaluation_gate_not_met`.

Development: baseline 53/96; selected 85/96. Limited observed improvement: true. Eligible for separate final evaluation: false.

## Limits and failure evidence

Two epoch exports were recorded; this is not final semantic qualification.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-mixed-gpu-sft-01/root-verdict.json` — SHA-256 `763f151c5707e47e3be60dcf1d83d2e89a049217d413160541a05d123547f060`.
- `runtime/kev-mixed-gpu-sft-01/contract.json` — SHA-256 `4f7812928d29772b9068ae23f4d14117b26a6fdf391e9099e3abf3928914cdc3`.

[Historical experiment index](README.md)
