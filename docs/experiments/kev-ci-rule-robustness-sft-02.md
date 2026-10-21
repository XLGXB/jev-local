# Historical experiment: kev-ci-rule-robustness-sft-02

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-25T07:10:28.688Z (field `created_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; fresh optimizer from the frozen CI-stage initializer.

- Planned epochs: 1. An incomplete run did not necessarily finish these epochs.
- Train: 4,528 rows; atoms: not recorded in the reviewed contract summary.
- Development: 168 atoms.
- A separate retention partition count is not recorded here; reported development subgroups are not additional examples.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **566**.
- Actual exports: **1**.
- Original verdict: `terminal_verified_mechanics_quality_gate_failure`.

Development: 138 to 151/168. CI: 72/72. Retention: 79/96. Content-filter retention: 8/16, versus baseline 14/16. Limited observed improvement: false; separate-final eligibility: false.

## Limits and failure evidence

Mechanics passed via supplemental independent verification. Retention-family regression remains a failed quality gate; aggregate improvement does not override it.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-ci-rule-robustness-sft-02/root-verdict.json` — SHA-256 `10d21d1f7f28faaa6c37897855f98be82e848efd306000fad39a4bd3cb15fb54`.
- `runtime/kev-ci-rule-robustness-sft-02/contract.json` — SHA-256 `7f02f859dd04a8a42feba5ffe6be21b79aac60235c846f1083e5e513e3004db7`.

[Historical experiment index](README.md)
