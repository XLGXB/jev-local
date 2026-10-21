# Historical mechanical probes and unexecuted designs

Published on 2026-10-21 as a retrospective of retained historical summaries. No training, model inference, evaluation, or metric recomputation was run for this publication. This does not imply an earlier public release or maintenance history.

These entries are not additional completed SFT experiments or semantic-quality successes. The two one-update probes are classified as mechanical micro-probes and excluded from the 19-stage SFT total.

## One-update mechanical probes

| Stage | Original receipt UTC | Actual updates | Export evidence | Original status / scope |
|---|---|---:|---|---|
| `kev-cpu-typed-probe-01` | 2026-09-24T18:49:37.015Z (`at`) | 1 | Export/reload recorded; numeric export count not established in the reviewed root summary | `cpu_mixed_typed_update_export_reload_mechanics_passed`; post-reload max logit delta 0 |
| `kev-gpu-probe-01` | 2026-09-24T20:44:37.300Z (`at`) | 1 | Export/reload recorded; numeric export count not established in the reviewed root summary | `passed_gpu_update_export_reload_mechanics`; post-reload max logit delta 0 |

These bounded mechanics do not measure representative semantic quality, training improvement, or acceleration. No sample inputs or per-sample results are published.

## Zero-update failures and preparation-only stages

| Stage | Original summary UTC | Actual updates / exports | Retained result |
|---|---|---:|---|
| `kev-ci-rule-robustness-sft-01` | 2026-09-25T06:09:34.066Z (`created_at`) | 0 / 0 | `terminal_tokenizer_admission_failed_native_training_not_admitted` |
| `kev-question-diversity-sft-01` | 2026-09-25T08:40:04.733Z (`at`) | 0 / 0 | `terminal_pre_gate_failure`; root receipt explicitly says no model, optimization, export, or dev inference ran |
| `kev-oracle-context-sft-01` | 2026-09-25T22:21:31.617Z (`adjudicated_at`) | 0 / 0 | `terminal_startup_failure_no_retry` |
| `kev-coverage-retention-sft-01` | 2026-09-26T05:55:21.376Z (`created_at`) | 0 / 0 | `terminal_native_oom_before_first_optimizer_update_no_retry`; no trained candidate quality |
| `kev-frozen-head-sft-01` | 2026-09-26T09:54:37.565Z (`recorded_at_utc`) | 0 / 0 | `terminal_checkpointing_head_policy_mismatch_before_training_no_retry` |
| `laya-f08-task-sft-01` | not recorded in the reviewed root summary | 0 / 0 | `terminal_capacity_rejected_no_retry`; model load and forwards also zero |
| `kev-bound-wording-durable-sft-01` | 2026-09-26T21:09:29.104Z (`created_at`) | not executed; actual counts not recorded | `CPU_runner_snapshot_full_verifier_integration_verified_native_approval_pending` |

The durable-wording contract's **812 updates, 5 exports, 4 epoch snapshots, and 7200-second budget are prospective**, not historical execution. Its integration receipt explicitly records the absence of execution-freeze, launch intent, start, and epoch snapshots. This is stronger than inferring an execution result merely from an unlocated file; no prospective number is added to actual-operation totals.

Incomplete runs with nonzero updates (including the three-update retention-bank failure and 528-update wording timeout) have their own SFT reports; they are not duplicated as extra experiments here.

## Full-model gradient repeatability diagnostic

`full-model-gradient-repeat-02/result.json` has original status `completed`, but is a **diagnostic, not training**:

- One model build / weight load; six encoder/decision forwards and six backwards.
- Zero optimizer constructions, optimizer updates, and exports.
- Original experiment timestamp: not recorded in the reviewed result.

This bounded repeatability evidence is not semantic qualification, proof of all historical drift causes, or authorization to repeat a terminal run. The earlier repeat-01 is a failed diagnostic predecessor, not an additional training success.

## Local evidence fingerprints

These repository-relative references identify local summaries; **the runtime artifacts themselves are not published**. SHA-256 identifies summary bytes, not data or model weights.

- `runtime/kev-cpu-typed-probe-01/root-verdict.json` — SHA-256 `f35cfe36a5030f61a1c48f4f24c9dcba9affe9fb006480998ca213ac27a09ce0`.
- `runtime/kev-gpu-probe-01/root-verdict.json` — SHA-256 `92d6c2264e387234813168982c8ab65022368db65acfe4b8dfa32c6674e23a03`.
- `runtime/kev-ci-rule-robustness-sft-01/root-verdict.json` — SHA-256 `fddee4a147e22c728e3a0541e5c77da6f432b5116c1bde7a74fd050d5ed05ddd`.
- `runtime/kev-question-diversity-sft-01/root-verdict.json` — SHA-256 `81d6f6fa6a0d82d0f04c7603d585e9557213604d2fe5aff22b6c460b12711c34`.
- `runtime/kev-oracle-context-sft-01/root-verdict.json` — SHA-256 `f2b6008744d55a5fb51c54733bc52452e09f43ce772e81ec73a690bba573c229`.
- `runtime/kev-coverage-retention-sft-01/root-verdict.json` — SHA-256 `ac8283bdf2f495922e1f7b11ed18ac4ef18c93b6f33333fa01112dd47c10e643`.
- `runtime/kev-frozen-head-sft-01/root-verdict.json` — SHA-256 `00fbacb0a12d85d21c0e4e59735ee1af7cdf6341dcb0d63fb1646487b905f7fa`.
- `runtime/laya-f08-task-sft-01/root-state.json` — SHA-256 `acfb83cfb7c9d0f286a179a24104844cc2afa305018a950f3c3c0d1900d4942b`.
- `runtime/kev-bound-wording-durable-sft-01/root-integration-verdict.json` — SHA-256 `a232351e200c44f552e947303c6f1c468a260e239cd98bc96e35a168fbe486cf`.
- `runtime/full-model-gradient-repeat-02/result.json` — SHA-256 `ee243b89241bb4495bc9dcafa14bb20f0e5137cb832c69ec2de28d8ad38445b2`.

[Historical experiment index](README.md)
