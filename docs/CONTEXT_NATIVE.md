# Explicit context-bound native Kev

This is an experimental, uncalibrated development route. It is not semantic
qualification or execution, merge, completion, or review authority.

## Supported path

1. Validate a context policy with decision_ft.context_backend.validate_context.
2. A separately frozen training runner may explicitly warm-start a native
   KevBackend and wrap it in ContextKevBackend, passing the exact three encoder
   source hashes. Do not change native global defaults.
3. Train using the wrapper's forward and train_groups. Its export writes the
   native files plus context_policy.json. The runner must also write the existing
   state_representation.json and local_checkpoint.json with genuine training
   provenance. A legacy checkpoint is not context-trained merely because its
   weights can be used as a warm start.
4. Build an explicit config and call
   jevlocal.integration.context_native.build_predictor(config).
5. Pass the resulting predictor to NativeReviewJudge or NativeCompletionJudge,
   then to the existing review_diff or verify_completion judge_factory.

Existing CLI and legacy route behavior are unchanged. There is no implicit
fallback and no new CLI option in this slice.

## Policy and identity

Current experiment policy is schema_version=1, max_state=768, max_branch=1408,
max_packed=2432, max_encoding_tokens=1024. Policy values are exact integers.
The state range and row/packed equations mirror the pinned Kev training_context
implementation. max_packed limits one encoded record, not the sum of a batch.
Training batch size, accumulation, GPU memory and wall-time budgets require a
separate execution freeze.

Encoding calls the genuine model.encode with explicit max_state, max_branch
and strict=True. It never delegates to the legacy 384-token encode path.
Full evidence is retained; truncation, row/record overflow, experimental budget
overflow and candidate reordering are rejected. Cache keys bind context digest
and represented atom fingerprint. The exposed policy is a detached copy.

The new config extends the old represented route config with context and
expected_context_sources, and uses route=context_native_uncalibrated_v1.
The native spec adds context_sha256. Complete checkpoint inventory has ten
files, including the two representation/context markers. The runtime identity
includes context policy, marker, both new source hashes, existing fourteen
source hashes, complete checkpoint hashes, representation binding, spec and
environment. Adapter hashes include this route and the existing two bridges.

Admission validates actual sources and loaded origins, complete checkpoint
inventory, both strict markers, original base snapshot hashes, process-start
offline flags and cached HF paths, and safe head metadata before constructing
the model. Predict checks identity before and after work, validates raw replies,
and always clears its encoding cache. Temperature remains 1 and calibration is
None.

## Evidence scope

The focused tests use CPU doubles. Real review_diff and verify_completion are
called, but scripted predictions are not learned model quality. Synthetic
receipt fixtures are not execution attestations.

The prior capacity experiment measured 504 atoms (1008 encodings across both
option-isolation modes): maximum state 639, maximum record 781. Its terminal
384-state/512-experimental-budget rejection remains unchanged. Under identical
pinned encoder/tokenizer/input sources, the explicit 768-state/1024-record policy
is statically feasible: the encoder uses these limits only for the strict gate
and state slice; every recorded state fits without slicing and every record
fits without truncation. This is source-bound reuse of saved measurements, not
a repeated tokenizer run, GPU memory evidence, training success, or semantic
qualification.

A separate frozen optimizer experiment is required next. Development,
calibration and test partitions must not be read as training data.

## Explicit zero-update inference adaptation

A separately frozen consumer experiment may also warm-start a retained,
represented checkpoint and call `ContextKevBackend.export` without any optimizer
update. This is an inference-context adaptation, **not context training** and not
semantic qualification. The original initializer remains read-only.

The stage runner must byte-preserve the original `state_representation.json`,
including its original epoch and training-freeze hash. It must record the genuine
native export provenance and the initializer lineage separately; a new execution
freeze must never replace the historical training-freeze field. Compare adapter
and head tensors before/after export and check genuine reload predictions under
the identical representation and context policy. Do not infer weight equality
from a successful file copy or from the presence of a marker.

Build a fresh ten-file identity and fingerprint, including the actual new context
marker and source/checkpoint hashes. The existing strict config admission is
unchanged. No failed candidate is made admissible by this procedure, and no
semantic quality gate is lowered. An adaptation without executed reload evidence
remains an unverified experimental artifact.
