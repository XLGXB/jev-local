# Historical experiment: laya-sample-order-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-27T04:29:23.712Z (field `recorded_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Laya multilingual. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

Original Laya multilingual initializer; head/type_emb/scorer only; deterministic sha256-sort-v1 sample-order intervention; AdamW LR 5e-5; seed 42; microbatch 1, accumulation 8; FP32, no AMP/TF32.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,124 rows / atoms (452 F08 + 672 replay).
- F08 development: 85 atoms.
- Retention: 168 atoms (72 CI + 96 generic).
- Actual F08 consumer cohort: 48 requests. It is not the development/retention denominator.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **564**.
- Actual exports: **1**.
- Original verdict: `native_completed_postverified_semantic_gates_failed`.

Original-train fit 558/1124. F08 development 34/85. Retention 57/168. Actual F08 consumers 15/48. Confidence threshold 0.95: accepted 0. Candidate UNQUALIFIED.

## Limits and failure evidence

Mechanical/postverification completion does not repair failed semantic gates. The receipt rejects further seed/order retries; it does not prove all samplers ineffective.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/laya-sample-order-sft-01/root-state.json` — SHA-256 `72eff5ba79a8a8915e78e996bb903a01700f7b7342b940d420e1186c01429131`.
- `runtime/laya-sample-order-sft-01/contract.md` — SHA-256 `1e5eea35c94006d46fb2af2c2d32c7a3eae0483e7a59bd0462acfac77e9d5920`.

[Historical experiment index](README.md)
