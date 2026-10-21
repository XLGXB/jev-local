# Retrospective experiment reports

Published: **2026-10-21**. Reviewed: **2026-10-21T10:51:08Z**.

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

## Scope and reading rules

This bounded inventory covers **19 actual SFT stages** identified from retained local summaries: 14 Kev and 5 Laya. Incomplete and rejected runs remain included with their original verdicts. One CLM evaluation-only baseline and a mechanical/preparation appendix are separate. No claim is made that every historical experiment has been found.

The reviewed entry points were `docs/PROJECT_STATUS.md`, `docs/NATIVE_VALIDATION.md`, `.codex-handoff/current.md`, runtime top-level directory names, and directly relevant compact terminal summaries/contracts/recipe summaries. Original timestamps below are embedded receipt/adjudication fields, not filesystem dates or invented training starts. This retrospective's publication date is distinct from the original September 2026 events.

The 19 SFT summaries record **9,254 optimizer updates and 20 exports** in aggregate. These are bookkeeping sums only: not quality, speedup, benefit, successful-training, or qualified-model totals. They exclude two one-update mechanical probes and all prospective counts.

## SFT reports

| Stage | Original receipt UTC | Actual updates | Actual exports | Publication commit |
|---|---|---:|---:|---|
| [kev-mixed-gpu-sft-01](kev-mixed-gpu-sft-01.md) | 2026-09-24T21:18:54.148Z | 96 | 2 | [`8b4289d`](https://github.com/XLGXB/jev-local/commit/8b4289d860f09d2ffb329cf93932c11ebde409cb) |
| [kev-state-json-sft-01](kev-state-json-sft-01.md) | 2026-09-24T23:03:33.433Z | 96 | 2 | [`61df05d`](https://github.com/XLGXB/jev-local/commit/61df05d0274db05a74af5f28dd7faa73cb12b2ee) |
| [kev-ci-rule-sft-01](kev-ci-rule-sft-01.md) | 2026-09-25T02:45:19.805Z | 168 | 2 | [`c406653`](https://github.com/XLGXB/jev-local/commit/c40665322a32861812c46e0cd8fbb395de11501a) |
| [kev-ci-rule-robustness-sft-02](kev-ci-rule-robustness-sft-02.md) | 2026-09-25T07:10:28.688Z | 566 | 1 | [`cf4d4d6`](https://github.com/XLGXB/jev-local/commit/cf4d4d68bd3fab3b8cb99ba034ddd97ed6c496dd) |
| [kev-question-diversity-sft-02](kev-question-diversity-sft-02.md) | 2026-09-25T09:27:00.672Z | 566 | 1 | [`d3ff18c`](https://github.com/XLGXB/jev-local/commit/d3ff18c6bf218c2aa09294219f693f6b11d79361) |
| [kev-retention-question-bank-sft-01](kev-retention-question-bank-sft-01.md) | 2026-09-25T11:05:20.267Z | 3 | 0 | [`a57e975`](https://github.com/XLGXB/jev-local/commit/a57e97583c12115a28f7fda1e7aec6d99c62b96f) |
| [kev-retention-question-bank-sft-02](kev-retention-question-bank-sft-02.md) | 2026-09-25T12:08:53.154Z | 566 | 1 | [`0d1c428`](https://github.com/XLGXB/jev-local/commit/0d1c428698131737c371f21b165d7365f01fdef2) |
| [kev-external-question-bank-sft-01](kev-external-question-bank-sft-01.md) | 2026-09-25T17:59:07.531Z | 737 | 1 | [`a8a0f37`](https://github.com/XLGXB/jev-local/commit/a8a0f37a17735fd4373cc6b828d5873fc09dab45) |
| [kev-oracle-context-sft-02](kev-oracle-context-sft-02.md) | 2026-09-26T03:51:07.116Z | 168 | 1 | [`3232942`](https://github.com/XLGXB/jev-local/commit/3232942989f4934e067145da5c7dc93f73a03b4c) |
| [kev-mixed-retention-sft-01](kev-mixed-retention-sft-01.md) | 2026-09-26T04:31:28.766Z | 504 | 1 | [`71d0541`](https://github.com/XLGXB/jev-local/commit/71d05411cd3f9f60ad847daa837e27c76bbb5798) |
| [kev-coverage-checkpointed-sft-01](kev-coverage-checkpointed-sft-01.md) | 2026-09-26T07:25:43.387Z | 812 | 1 | [`e315a8e`](https://github.com/XLGXB/jev-local/commit/e315a8e767ebd3bf387427bdde3a96387f8c6f11) |
| [kev-frozen-head-sft-02](kev-frozen-head-sft-02.md) | 2026-09-26T10:50:39.597Z | 812 | 1 | [`f167ced`](https://github.com/XLGXB/jev-local/commit/f167ced704037fed1e01bc9aad872ea6b9aa6149) |
| [kev-invariance-retention-sft-01](kev-invariance-retention-sft-01.md) | 2026-09-26T13:10:29.817Z | 812 | 1 | [`6298c47`](https://github.com/XLGXB/jev-local/commit/6298c47b70722078cbf6083c415d5220be1fb9d3) |
| [kev-bound-wording-sft-01](kev-bound-wording-sft-01.md) | 2026-09-26T20:15:16.171Z | 528 | 0 | [`3a2ffbb`](https://github.com/XLGXB/jev-local/commit/3a2ffbb6245a9d4bace1301bbe73504bfc55d84a) |
| [laya-full-option-sft-01](laya-full-option-sft-01.md) | 2026-09-27T03:32:27.019Z | 564 | 1 | [`a09d4d0`](https://github.com/XLGXB/jev-local/commit/a09d4d054217b1633042837a866c6044090dde28) |
| [laya-sample-order-sft-01](laya-sample-order-sft-01.md) | 2026-09-27T04:29:23.712Z | 564 | 1 | [`7c2e0ae`](https://github.com/XLGXB/jev-local/commit/7c2e0ae3647e96a68d42d3aaca674df3dbbd4086) |
| [laya-last-block-sft-01](laya-last-block-sft-01.md) | 2026-09-27T05:13:05.702Z | 564 | 1 | [`8bd36ca`](https://github.com/XLGXB/jev-local/commit/8bd36ca695cfa0373ef8f2b831c86b203a2bd027) |
| [laya-encoder-blocks-sft-01](laya-encoder-blocks-sft-01.md) | 2026-09-27T05:59:03.369Z | 564 | 1 | [`e1940cb`](https://github.com/XLGXB/jev-local/commit/e1940cbc0ca2d2f28fe9f429c4e3cbf3f76b7a64) |
| [laya-question-bank-sft-01](laya-question-bank-sft-01.md) | 2026-09-27T09:32:38.715Z | 564 | 1 | [`b28fdc5`](https://github.com/XLGXB/jev-local/commit/b28fdc52fce120fbb54f1260a9fb4d95bdc43510) |

## Other evidence

- [CLM original-view baseline](clm-original-view-baseline-01.md): evaluation-only, not SFT.
- [Mechanical probes, zero-update failures, and unexecuted designs](mechanical-and-unexecuted.md): no additional completed SFT claims.
- [Machine-readable manifest](manifest.json): explicit source-summary fingerprints, actual operation counts, original verdicts, dates, and unknowns as null.
- [Existing native-validation scope](../NATIVE_VALIDATION.md): model lineage and synthetic smoke limits; zero-update smoke is not training improvement.

## Provenance and reproducibility limits

Each report lists repository-relative local summary paths and SHA-256 fingerprints. **Those runtime artifacts are not published with these reports.** A fingerprint is a binding to retained evidence, not a complete public reproducibility package or permission to obtain private source files.

Only configuration overviews, partition counts, aggregate metrics, original verdicts/failures, and summary fingerprints are published. Raw questions, sample IDs, per-sample labels or predictions, training/evaluation inputs, sealed partitions, weights/checkpoints, raw/private logs, account paths, and old task-sharing links are not included. This pass did not inspect or hash sealed data. No new redistribution or source-training authorization is inferred from a historical admission.

Datasets, exact per-run model inventories, runtime environments, and full native receipts are intentionally absent from this publication. A reader cannot independently reconstruct or validate the original training runs from these reports alone. Unknown fields remain unknown; a missing summary field does not become zero. These reports are not instructions to resume an experiment or to open calibration, final-test, or sealed data.

Comparisons retain their original denominators and gates. Development/retention accuracy is not an actual-consumer denominator. An exported model, reload parity, limited synthetic-development improvement, future-evaluation eligibility, and semantic qualification are distinct. A failed or incomplete run is not retrospectively repaired by publication. Memory metrics with different measurement sources are not directly comparable.

## Publication and commit boundaries

One actual SFT stage has one report, so its entire historical result can be committed together. The index/manifest form shared retrospective metadata; the CLM baseline and mechanical appendix remain distinct from SFT. Original event dates are evidence fields only: do not backdate commits, imply earlier public maintenance, or split one experiment into multiple supposed runs to inflate history.
