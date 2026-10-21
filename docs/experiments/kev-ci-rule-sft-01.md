# Historical experiment: kev-ci-rule-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T02:45:19.805Z (field `created_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8.

- Planned epochs: 2. An incomplete run did not necessarily finish these epochs.
- Train: 672 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **168**.
- Actual exports: **2**.
- Original verdict: `accepted_mechanics_and_limited_dev_improvement_only`.

Development: baseline 97/168; selected 138/168. CI: 47 to 64/72. Retention: 50 to 74/96. Limited observed improvement: true; separate-final eligibility: false.

## Limits and failure evidence

Mechanics and limited development improvement were accepted, not full quality or promotion.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-ci-rule-sft-01/root-verdict.json` — SHA-256 `d6f1d258d4f116fe5a3772b2ff06fb42275671e8d8fdea8e8a2b8524c891f6be`.
- `runtime/kev-ci-rule-sft-01/contract.json` — SHA-256 `3235c616a006a98d1aebaf848986bacd2a93d3d13e8eee72742265ae21a5960f`.

[Historical experiment index](README.md)
