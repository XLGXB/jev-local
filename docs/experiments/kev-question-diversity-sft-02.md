# Historical experiment: kev-question-diversity-sft-02

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T09:27:00.672Z (field `at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; controlled question-diversity intervention.

- Planned epochs: 1. An incomplete run did not necessarily finish these epochs.
- Train: 4,528 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **566**.
- Actual exports: **1**.
- Original verdict: `terminal_native_verified_quality_failed`.

Development: initializer 138/168; candidate 151/168. The preceding robustness comparison was also 151/168. Raw NLL: initializer 0.976093739897632; robustness comparison 0.9168681157715908; candidate 0.7330713173686705. Incremental nonregression gate: false; qualified/promoted/separate-final eligible: false.

## Limits and failure evidence

Lower NLL did not repair the retention-family regression. This is a development-informed comparison, not a held-out qualification result.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-question-diversity-sft-02/root-verdict.json` — SHA-256 `710c0f3d9ee08e649556cef68718e0af964ff8131525f09028d2fc4f093db022`.
- `runtime/kev-question-diversity-sft-02/contract.json` — SHA-256 `cf9fbe4b2d82c31b96daf2f2ad7df9d60a59adef8ab59e038bbf31415a1d10db`.

[Historical experiment index](README.md)
