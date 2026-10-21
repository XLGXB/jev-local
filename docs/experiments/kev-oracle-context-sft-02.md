# Historical experiment: kev-oracle-context-sft-02

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Identity and dates

- Kind: actual SFT stage, including incomplete or rejected runs.
- Published: 2026-10-21.
- Original summary timestamp: 2026-09-26T03:51:07.116Z (field `adjudicated_at`). This is a receipt/adjudication timestamp, not an inferred training start.
- Training start/end timestamps: not recorded in this report.
- Model family: Kev-0.8B. Exact per-run weight inventories are not published; this overview does not establish artifact reproducibility or provider parity.

## Configuration and partition counts

LoRA and decision heads; AdamW; LoRA LR 2e-5, head LR 5e-5; microbatch 1, accumulation 8; explicit oracle-context route.

- Planned epochs: 4. An incomplete run did not necessarily finish these epochs.
- Train: 144 rows; atoms: 336.
- Oracle development: 84 atoms.
- Original retention development: 168 atoms.

These are aggregate provenance/partition statistics, not public training examples. No raw questions, sample IDs, per-sample outputs, weights, or logs are included. Historical source admission is not a blanket redistribution license or a new source approval.

## Recorded result

- Actual optimizer updates: **168**.
- Actual exports: **1**.
- Original verdict: `terminal_reload_parity_failure_and_development_quality_rejection`.

Oracle development: baseline 54/84; final 81/84. Original development: baseline 138/168; final 140/168. Oracle observed improvement true but separate-final eligibility false; qualification unqualified.

## Limits and failure evidence

Export occurred, but reload parity failed and development quality was rejected. Neither export presence nor oracle accuracy establishes a usable candidate.

No original threshold was changed for this retrospective. Export, reload, a development improvement, and eligibility for a future evaluation are distinct claims. None authorizes deployment, promotion, resumed training, or access to sealed partitions.

## Local evidence fingerprints

The following repository-relative paths identify retained local evidence. **The runtime artifacts themselves are not published with this report.** The contract records planned settings, not proof that every planned operation occurred; actual counts and verdict above come from the terminal summary. SHA-256 values identify summary bytes only, not a reproducible training package.

- `runtime/kev-oracle-context-sft-02/root-verdict.json` — SHA-256 `e282a2d453632a5ff23042c6968010cd8180ecee6031105b3954630b0fc01c1a`.
- `runtime/kev-oracle-context-sft-02/contract.json` — SHA-256 `834a5b2f805f22e5a51047e423759ea7c92addabd6e304b62bfc7382a27e1bfe`.

[Historical experiment index](README.md)
