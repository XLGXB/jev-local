# Historical experiment: kev-bound-wording-sft-01

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-26T20:15:16.171Z (field `recorded_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA only with frozen initializer heads; head LR 0, LoRA LR 2e-5; AdamW; microbatch 1, accumulation 8; activation checkpointing; four admitted wording-source categories; 812 planned updates.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 1,366 rows; atoms: 1,622.
- Oracle development: 84 atoms.
- Original retention development: 168 atoms.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **528**.
- Actual exports: **0**.
- Original verdict: `terminal_wall_budget_exhausted_528_updates_no_quality_verdict_no_retry`.

528 completed optimizer updates, zero exports. Recorded wall time 3600.218 seconds. quality_observed false; semantic_rejection false; promotion_authorized false.

## Limits and failure evidence

Wall-budget exhaustion, not a recorded VRAM failure or semantic rejection. The terminal-failure postcheck did not establish terminal_failure_evidence_valid=true; no completed-candidate quality verdict is available.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-bound-wording-sft-01/root-verdict.json` — SHA-256 `8f82595eccd9be8f851306933fc05a5935d9a5fb37147a86d7c1f764c0f3fbf9`.
- `runtime/kev-bound-wording-sft-01/contract.json` — SHA-256 `2bef8533d62cd6ad08982dcfbeb6ea03597471cbda5c8dca834492d1ae056045`.

[Historical experiment index](README.md)
