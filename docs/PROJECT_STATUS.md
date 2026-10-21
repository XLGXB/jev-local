# Project status

Updated: **2026-10-21**. The initial source was uploaded to [XLGXB/jev-local](https://github.com/XLGXB/jev-local) at [b31492c](https://github.com/XLGXB/jev-local/commit/b31492c131315f73daad3f8f06a6bb430c1d93e2). This update adds public-readiness navigation and a new local CPU demo; it is not new training, broad model qualification or browser acceptance. Historical experiments occurred in September 2026; individual summaries preserve their original dates, pins and limits.

## Full goal and acceptance boundary

The complete **F01–F20 local Jev + Codex goal remains ACTIVE and incomplete**. The retained objective labels are:

| ID | Goal label | ID | Goal label |
|---|---|---|---|
| F01 | fast-jev-compaction | F11 | jev-ultrafast |
| F02 | Winnow | F12 | json-render |
| F03 | jev-codex-router | F13 | agent-desktop |
| F04 | jev-review | F14 | jev-browser |
| F05 | Canny | F15 | jev-search |
| F06 | Blink | F16 | Reticle |
| F07 | typesafe-mcp | F17 | NanoJev |
| F08 | jev-mcp | F18 | open-alternative-jev |
| F09 | SemDecide | F19 | pi-jev |
| F10 | jev-curate | F20 | hermes-jev-skills |

These labels preserve scope, not a completed-feature checklist. The module summary below does not replace their original criteria or infer a one-to-one implementation mapping. Where representative consumer/semantic evidence has not been established, acceptance remains **pending**, not passed by file existence.

Keep four layers separate: **implementation** (mechanism/contracts), **consumer** (a particular pinned real host/worker), **semantics** (quality, retention, calibration/rejection), and **recovery/performance** (measured scenario). Production consumers and host qualification have not all passed. No default promotion, general autonomous execution or speed gain follows from isolated evidence.

## Implemented modules and scoped consumers

| Area | Implemented surface | Acceptance still separate |
|---|---|---|
| Canonical data | `jevlocal` schema, semantic keys, explicit permission/provenance bridge, frozen four-way partition checks | Label truth, source rights, representative quality |
| Native training/inference | `decision_ft` backends, evaluation/calibration, explicit represented routes; synthetic Laya/Kev native smoke | Semantic qualification, general training improvement, export/provider parity |
| Serving/qualification | `decision_deploy` strict `/v1/decide`, runtime identity and artifact binding | Supported production host integration and actual permission |
| Browser | `fastloop.service.invoke`, target lock, durable effects, grants, uncertain-outcome review | Authorized tasks, native-qualified predictor, broader recovery/performance |
| Context/evidence | Lexical baseline, explicit active native scoring, passive shadow/scoped replay and optional MCP | Broad connected-host coverage and semantic relevance; citation integrity is not truth |
| Explicit advice | F03 routing, F04 diff risk, F05 completion evidence, F06 bounded navigation, F08 source-bound claim support | Full objective acceptance, representative quality and downstream qualification |
| Other consumers | SemDecide, static JSON UI choice, opt-in Qwen relevance aid | Route-specific quality/calibration, platform/resource and production gates |
| Pi/F19 slice | Official Pi SDK extension, owned worker, typed advice and active-tool suggestion | Full F19, interactive recovery, OS-level egress isolation, semantics/default promotion |

See [integration](INTEGRATION.md), [native validation](NATIVE_VALIDATION.md), and [Pi](../pi/README.md) for exact contracts. Some tests use fake predictors; no current full-suite success is inferred from source existence.

## Public demonstration and readiness evidence

The [evidence-ranking demo](../demos/evidence_ranking/README.md) uses **Qwen/Qwen3-Reranker-0.6B**, revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, for real local **CPU** inference on 24 newly authored public synthetic questions with 72 candidates. Top-1 accuracy was **20/24**, compared with **14/24** for the lexical baseline. The measured subsets were direct 12/12, paraphrase 8/8, and lexical-distractor **0/4**; all four distractor gold candidates ranked third. See the [saved local result](../demos/evidence_ranking/results/local-cpu-20261021.json).

This is zero-training, small synthetic evidence. It is not representative domain acceptance, calibration, training improvement or a demonstrated time saving. The failures are part of the result, not omitted exceptions.

[Historical experiment summaries](experiments/README.md) cover **19 actual SFT runs: 14 Kev and 5 Laya**, with CLM classified separately as baseline and mechanical/unexecuted work kept in appendices. They publish reviewed configuration/metrics/summary hashes, not original private samples, weights or raw logs. Historical dates must not be relabeled recent ongoing maintenance. An SFT run count is not a count of qualified models.

The [CI workflow](../.github/workflows/ci.yml) configures dependency-light standard-library checks on Windows and Linux; a workflow file does not prove a hosted run passed. The authorized manual cloud-CPU reproduction has **not yet been executed**. Contribution guidance and issue forms are available, but independent community adoption/usage evidence has not been established; Stars/Forks cannot substitute for it.

## Latest retained training quality evidence: not qualified

Local compact summaries: `runtime/laya-question-bank-sft-01/root-state.json` and `runtime/clm-original-view-baseline-01/root-state.json`, recorded 2026-09-27. They remain local-only.

| Historical stage | Original dev decisions | Retention decisions | Confidence at 0.95 | Verdict |
|---|---:|---:|---|---|
| LW question-bank SFT | 58/85 | 129/168 | 104 accepted, 9 errors; failed | `UNQUALIFIED` |
| CLM original-view baseline | 31/85 | 76/168 | 14 accepted, 5 errors; failed | Unqualified; original gates failed |

Dev/retention totals are not real F08 consumer denominators. LW separately records **31/48** correct actual F08 consumers, with its consumer gate failed. Its limited-improvement flag does not override failed dev/confidence/consumer gates; separate final evaluation was ineligible. Mechanical training/export/reload verification is not semantic acceptance. Neither terminal stage may be blindly restarted or promoted.

### Full-model gradient diagnostic

`runtime/full-model-gradient-repeat-02/result.json` records one selected 1,207-token sample, one model construction, six forwards/backwards, **zero optimizer updates and zero exports**. Three default-attention repeats had identical logits but differing gradients; three explicit math-route repeats matched one another exactly for this sample. Complete model-state hashes before/after matched.

This is bounded repeatability evidence, not quality improvement, a general determinism proof, proof of all historical drift causes, or permission for new training. It does not repair failed gates. This organization pass did not rerun it or inspect sealed examples/labels.

## Recovery and resource evidence

- **Isolated browser recovery, 2026-09-27:** `runtime/browser-dispatch-crash-recovery-01/root-verdict.json` reports 8 passed, 0 skipped. A real loopback Chromium effect occurred once; an interrupted run remained `OUTCOME_UNKNOWN` without replay, then closed through review. Predictor/qualification were test doubles. This is not user authorization, native semantics, arbitrary host recovery or independent enumeration of all browser descendants.
- **Pi, 2026-09-26:** [Pi README](../pi/README.md) separates official SDK loading, native parity, worker allocation, normal EOF and pending-session drain. W04 measured three model/tokenizer loads and five forwards, with reference probability delta 0; late advice was discarded before replacement. These scoped lifecycle/resource results do not prove arbitrary task replay, interactive restart, OS-level egress isolation or end-to-end speed.
- **Native smoke, 2026-09-23:** [Native validation](NATIVE_VALIDATION.md) records synthetic inference/gradient/reload and in-process ASGI evidence. Zero-update smoke is not training improvement.

Stage wall times include cold loads/harness work; do not relabel them inference latency or speedup over an unmeasured baseline.

## Remaining acceptance

- Retain all F01–F20 objectives and dependencies; judge against original criteria, not module/file counts.
- Establish authorized, representative, leakage-controlled semantic quality, retention, calibration and rejection evidence. Failed candidates remain unqualified.
- Complete supported production-host/consumer qualification. Explicit CLI, stdio, SDK slices and connected-host operation are distinct; universal Codex interception is not claimed.
- Verify applicable recovery, resource, egress and performance scenarios without replaying uncertain effects or reopening terminal stages without a newly authorized scope.
- Preserve the Apache-2.0 project-code decision and applicable third-party obligations; review any newly distributed model/data assets separately. See [licensing](LICENSING.md) and [publishing](PUBLISHING.md).

This documentation edit itself performs no training, installation, service registration, commit or push. The initial source upload and new local CPU demo are distinct recorded activities described above. Historical `runtime/` paths identify retained local records, not files bundled with published source. Hosted/manual cloud results remain pending until their actual receipts exist.
