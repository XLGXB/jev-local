# Historical experiment: laya-question-bank-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-27T09:32:38.715Z (field `recorded_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Laya multilingual. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

Original Laya multilingual initializer; all 22 encoder blocks plus original heads, 162 optimizer tensors / 125,101,057 elements; same four-epoch recipe as encoder-blocks stage; only 1,152 prescribed training question views changed; labels/facts/partitions unchanged.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,124 rows / atoms (452 F08 + 672 replay).
- F08 development: 85 atoms.
- Retention: 168 atoms (72 CI + 96 generic).
- Actual F08 consumer cohort: 48 requests. It is not the development/retention denominator.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **564**.
- Actual exports: **1**.
- Original verdict: `terminal_mechanically_verified_semantically_unqualified`.

Original-train fit 1075/1124. F08 development 58/85. Retention 129/168: CI 61/72 and generic 68/96. Actual F08 consumers 31/48. Confidence threshold 0.95: accepted 104, errors 9. Limited improvement true; separate-final eligible false; qualification UNQUALIFIED.

## Limits and failure evidence

Epoch-1 control drift: max logit delta 4.754974126815796 and 34 development argmax mismatches. Cause is unresolved; do not attribute the final difference to question wording alone. Development-informed design, not blind evaluation.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/laya-question-bank-sft-01/root-state.json` — SHA-256 `22cc034f55b76a6fc7e5dd6081f02e6f4c9825e1cb8af0d36e9cef5577bfb28e`.
- `runtime/laya-question-bank-sft-01/contract.md` — SHA-256 `bb6122e6152cff80a20f2789ffc9cf05345e3c68e5ed73f492efd76a531d2bd4`.

[Historical experiment index](README.md)
