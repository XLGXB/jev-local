# Independent JevLocal integration

The implementation is in `D:\Projects\JevLocal`, outside the novel project. The old project is never modified or used as a deployment destination.

## Explicit active native context route

`jevlocal context rank_evidence` and `suggest_skills` accept an optional
`--native-runtime <operator-owned JSON file>`. Without this flag, the existing
labelled lexical baseline remains active. The JSON has exactly these keys:

```json
{
  "mode": "native",
  "backend": "laya",
  "checkpoint": "D:/Projects/JevLocal/models/laya-multilingual/1c5edc17a7acd8701df6fc341c0d179f1c62c982",
  "device": "cuda",
  "amp": "none",
  "weights_dtype": "fp32"
}
```

Only `laya`/`kev`, `cpu`/`cuda`, and the shown FP32/no-AMP profile are accepted.
Checkpoint directories must exist on an explicitly trusted local drive, without
UNC paths or linked/reparse-point ancestors. No download or provider fallback is
performed. Start the process with `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, and
`TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` already set. Already-imported HF offline caches
are checked too. For Kev, point `HF_HOME`/`HF_HUB_CACHE` to the already populated
project-local matching-base cache. Use the corresponding isolated environment;
do not alter global environment or install packages to launch this route.

Example argument sequence for the project launcher or configured environment:
`python -m jevlocal.integration.cli context rank_evidence --input request.json --native-runtime runtime.json --output new-result.json`.
Input keeps the same documented context-tool arguments; executable callbacks
are never read from JSON. Original input is not overwritten. Output files are
exclusive-create. The `route` field identifies backend, runtime fingerprint,
context-adapter SHA, rubric version, `calibrated=false`, and `shadow=true`.
Here the legacy `shadow=true` means non-authoritative model advice; this active
route does change returned scores/ranking. It does **not** mean the lexical
primary result is preserved. The separate passive entry below does preserve it.

For an explicitly configured MCP stdio process, optionally set
`JEVLOCAL_CONTEXT_NATIVE_RUNTIME` to an absolute runtime-config JSON path in that
process's environment. `JEVLOCAL_CONTEXT_STORE` remains required. Runtime paths
and callbacks are not exposed in MCP tool arguments. This does not install or
register MCP globally, launch a daemon, or provide Codex-wide interception.

Each candidate is independently scored with a Noul relevance question, using
the existing sanitizer before inference. Scores are **not** a distribution over
candidates, are uncalibrated, and do not authorize execution or deletion. One
predictor belongs to one scorer; reuse the scorer rather than sharing its model
with concurrent callers. Model inference is serialized inside that scorer.
Candidate ranking can be native; cited-line selection remains lexical.
`filter_tool_output` and `restore_lines` retain their deterministic behavior and
do not accept `--native-runtime`.

No partial native result or implicit lexical/cloud fallback is returned on
failure. Preserve caller-owned original material and report the failure. A
budget shortage returns `requires_larger_budget` before storing excerpts; route
provenance counts toward the compact-JSON character budget. Existing backends
reject token truncation. This is a mechanics entry point, not a domain-quality
qualification, calibrated rejection policy, or training loop.

## Explicit passive rank-evidence CLI

```text
python -m jevlocal.integration.cli context rank_evidence --input request.json --output new-primary.json --shadow-runtime runtime.json --shadow-output new-observations.json --shadow-timeout 60
```

`--shadow-runtime` and `--native-runtime` are mutually exclusive. Passive mode is
only supported for `rank_evidence` and requires a new, separate `--shadow-output`.
No flag means no observer/model import. The runtime JSON uses the same exact
native configuration above; the process uses the caller's current interpreter,
never a different Python/model/provider fallback. Choose the corresponding
isolated Laya or Kev environment and already populated local cache explicitly.

The parent validates/redacts the explicit input using the existing rank boundary,
freezes the safe query/candidates, starts one finite hidden child, and runs the
existing lexical ranking exactly once. Primary stdout is flushed and an optional
primary file is written before waiting for the observer. Neither model failure,
missing dependencies, capacity rejection nor timeout replaces that lexical
answer. This is not a fallback: lexical ranking is the chosen primary route from
the start. Its existing budget/status behavior and sanitized text storage remain
unchanged. The original input is never overwritten.

The child uses the existing `ShadowHost` and native Noul relevance rubric, one
independent question per candidate. It sets process-only offline/safe-loading
flags before model imports, denies socket connections, and has no action callback.
Temporary IPC holds only the sanitized snapshot and operator configuration;
payload/result are each bounded to 1 MiB. A larger valid lexical input can still
succeed while shadow availability reports capacity failure. Redaction remains
best-effort, not permission to feed arbitrary private material.

`--shadow-timeout` is finite and in `(0,60]`, including observer startup. Late
results are discarded. After its deadline the parent kills its owned child and
performs a bounded one-second reap attempt; this is not a hard real-time guarantee
for filesystem operations or operating-system cleanup. Cleanup failure retains
the live process handle and its owned temporary directory for another `close`,
records `reaped/closed/temporary_cleaned`, and the CLI reports failure rather than
claiming safe completion. No daemon or replacement worker is started.

The final sidecar is serialized to a same-directory owned staging file and
published with an exclusive hard link; unsupported filesystems or an existing
destination fail rather than fall back to overwrite. A failed staging write does
not publish partial JSON. The CLI reports sidecar persistence/cleanup errors on
stderr with exit 2, without printing a second primary JSON or replacing the
already published result. Native unavailability alone preserves the primary's
normal exit status. Parent validation rejects changed snapshot/request bindings,
wrong IDs/counts, inconsistent runtime identity, malformed probabilities,
duplicate JSON keys and nonfinite values.

Sidecar fields distinguish:

- `host_ranking/host_top`: what the lexical host actually returned.
- `shadow_ranking/shadow_top`: uncalibrated native relevance estimates, never a
  verified label or an authorized action.
- `availability`: observed and total candidate counts, independent of truth-label
  availability. An unlabeled candidate remains in this denominator.
- `mechanical_feedback`: independent checks of candidate IDs, source/content
  hashes, quoted lines/offsets and optional hash-checked stored text. `verified`
  means citation integrity, **not** relevance correctness; semantic relevance
  remains unknown and no `verified_labels` are generated.
- Frozen snapshot/primary-result hashes, source/text hashes and quote/line bindings,
  native receipts/full probabilities/runtime fingerprints, and child lifecycle.
  The final sidecar excludes raw query/candidate text; use caller-owned input and
  the optional sanitized store for replay.

All records retain `training_allowed=false`, `execution_authorized=false`, and
`cloud_data_used=false`. No inference is made about unexecuted alternatives or
model quality from host success or ranking agreement. This is a real explicit CLI
entry, not automatic coverage of daily Codex activity or a semantic-labeling loop.

## Passive local shadow and actual feedback

`jevlocal.integration.shadow.LocalShadow(predictor).observe(envelope)` freezes
the same four-field pre-action envelope, including candidate order. Supply one
exclusively owned native predictor and already authorized/sanitized input. It
returns an in-memory `ShadowObservation`; `receipt()` contains typed decisions,
request digest and runtime identity, not raw state or candidate descriptions.
Prediction errors produce `status=unavailable` without a live-route fallback.
`LocalShadow` itself does not start a background worker or automatically intercept
any Codex/cloud call. For an explicit embedding host, the bounded coordinator
below supplies scheduling and result deadlines. Live Codex-wide task hookup
remains separate integration work.

The host executes its approved task exactly once through its existing controller.
After collecting actual evidence, call
`assess_feedback(observation, evidence_bytes, verifier=host_check, verifier_id="...")`.
The explicit trusted Python verifier receives the original snapshot and bounded
evidence bytes, checks postconditions **without replaying the action**, and returns:

```python
{
    "request_sha256": request_digest(original_envelope),
    "outcome": "success",  # or failure / unknown
    "executed": {"action": "write"},
    "verified_labels": {"action": "write"},  # empty unless independently proven
}
```

The verifier must verify actual evidence and its binding; it must not merely
echo labels supplied by a model or a `verified=true` field. Its ID, source-code
hash and evidence hash are recorded. These hashes aid provenance, not protection
against a malicious host program. Verifiers are not accepted as JSON or MCP
tool arguments. Public receipts remain `training_allowed=false` and
`execution_authorized=false`; record collection does not authorize training.

Three distinct comparisons are retained: independently established label
correctness, agreement with the actually executed action, and the observed
outcome of that action. Execution success alone never labels a prediction
correct. An unexecuted local alternative has `local_action_outcome=unobserved`;
only independently decidable labels can establish its answer correctness.
Missing labels and unavailable local predictions remain `correct=null`.
Action outcome and proposition truth are independent: `outcome=unknown` may
coexist with independently verified labels. This intentionally supersedes the
older coupled rejection rule. A known label does not turn an unknown action
outcome into success, invent an executed action, or authorize training.
No cloud output is used in this stage. Write receipts only to caller-owned,
authorized storage; the library itself neither scans nor persists files.

## Read-only replay of passive context artifacts

```text
python -m jevlocal.integration.cli replay-context --input rank-request.json --primary primary.json --shadow observations.json --output new-replay-report.json
```

This explicit entry reads exactly three caller-selected files (each at most
4 MiB), rejects duplicate/nonfinite/deeply nested JSON and validates the v1
passive sidecar. It reproduces the lexical primary in memory without persisting
text, verifies candidate/snapshot/request hashes, native receipt types and
runtime consistency, and checks derived host/native rankings and availability
denominators. It does **not** rerun inference or any prior action. The special
minimal `observer_internal_error` CLI receipt has no replayable observation and
is rejected, rather than upgraded into one.

The `store_dir` recorded in input is data, not authority to read that location.
Supply `--store-dir <explicit-existing-store>` separately to recheck current
stored citation bytes; a missing/corrupt requested store rejects replay. Without
that option, `store_check=not_requested` even when the historical primary says
`stored=true`. Output publication is exclusive and atomic as for shadow sidecars.
Default context ranking and passive inference behavior remain unchanged.

For an embedding trusted host, `context_replay.read_replay_files(...)` (or
`load_replay(arguments, primary, sidecar, store_dir=...)`) returns a frozen replay.
Its `report()` excludes raw query/candidate text. Use
`replay.assess(candidate_id, evidence_bytes, verifier=host_check, verifier_id=...)`
to feed an actually recorded observation through the existing `assess_feedback`
contract above. The host must verify the exact proposition, candidate object,
content version and evidence scope. A callback is never selected from JSON, an
artifact path, model output or CLI argument. No stored label or ranking agreement
automatically becomes truth. If the observer produced no receipt for a candidate,
assessment rejects instead of fabricating a prediction or runtime fingerprint.

The feedback result is a separate record with snapshot/primary/sidecar and raw
artifact hashes; the original report remains unlabeled. `training_allowed`,
`execution_authorized`, `cloud_data_used` remain false. This is observation replay,
not a third training schema, automatic sample promotion or all-task collector.

`status=consistent` establishes supplied-artifact consistency only, not historical
authenticity. Legacy v1 sidecars have no independently authenticated event time or
unique invocation ID; identical snapshots cannot prove distinct executions.
Reports explicitly retain `origin_authenticated=false`, `event_time_known=false`,
and distinguish current replay time from capture time. Runtime fingerprints and
recorded lifecycle flags are consistency-checked claims, not independently
authenticated process history. Trusted caller provenance and future event capture
remain separate requirements; replay never invents missing original evidence.

## Bounded explicit host coordinator

`jevlocal.integration.shadow_host.ShadowHost(predictor, max_pending=4)` owns one
native `LocalShadow` and one serialized worker thread. Do not share that predictor
with another worker, scorer, or direct caller. The embedding host supplies an
already authorized/sanitized four-field envelope **before** taking its own action:

```python
host = ShadowHost(predictor, max_pending=4)
try:
    ticket = host.begin(pre_action_envelope, timeout_s=5.0)
    # Existing trusted host/controller performs its authorized action exactly once.
    # This coordinator has no action/exec callback and never dispatches that action.
    evidence = collect_program_evidence_without_replaying_action()
    receipt = host.feedback(ticket, evidence, verifier=host_check, verifier_id="host-v1")
finally:
    shutdown = host.close(timeout_s=5.0)
    # Owner must inspect shutdown, including when live work raised an exception.
    # Undrained native work still belongs to this process; see below.
```

The sample's host action and evidence collector are application-owned, not new
library permission or execution APIs. Use the existing `fastloop.service.invoke`
for that controller's browser actions, with its real qualification and approval
gates. A shadow result cannot select the live route or waive those gates.

`begin` returns without waiting for inference; its deadline includes time in the
queue. `max_pending` (integer 1..16) bounds running plus queued jobs; saturation
or admission after close produces an unavailable ticket instead of a fallback
prediction. Invalid inputs are still rejected. `timeout_s` is finite and within
`0 < seconds <= 60`. `collect(ticket)` waits only the original remaining deadline,
then returns a stable observation. Native errors, queue expiry, cancellation and
inference timeout carry no selected answer. Feedback on such observations can
retain a real observed outcome while correctness remains unknown. A result that
finished within its deadline can be collected later for that original snapshot;
it is not evidence that the present UI or authority is unchanged.

`cancel(ticket)` prevents future use of that prediction. A queued job can be
cancelled before native inference. Running native work **cannot be killed by
thread cancellation**: its capacity stays occupied until it actually ends, and
its late result is discarded. Do not start a replacement runtime to evade this
backpressure. Cancellation does not revise an already finalized collection.
Tickets are scoped to their originating host; foreign tickets are rejected.

`close(timeout_s=...)` stops admission and discards uncollected results, with a bounded
wait; unlike `begin`, it also permits `timeout_s=0`. Inspect `closed`, `drained`,
and `pending`; `drained` means no queued/running inference remains, not that the
worker thread has exited. A zero-time close can report drained before an idle
worker exits. `drained=false` means the embedding process still owns native work.
A repeated close may wait for actual drainage.
The owner must drain it or use its explicitly authorized process-level cleanup;
there is no promise that a Python process can exit while a hung native thread is
still running. This is not a daemon, global hook, or hard-kill GPU cancellation.
Use `try/finally` around embedding work and always inspect the close result.

The host owns ticket/receipt retention. No raw snapshot or feedback evidence is
written by this coordinator; public receipts still exclude raw state and retain
`training_allowed=false`, `execution_authorized=false`, `cloud_data_used=false`.
There is no cloud teacher, cloud-output benchmark, optimizer, or automatic dataset
promotion. This explicit API does not imply all daily Codex work is shadowed.

## Components and boundaries

| Component | Responsibility | Does not establish |
|---|---|---|
| `jevlocal.schema` | Preserved canonical data | Native-model compatibility by itself |
| `jevlocal.integration.data_bridge` | Explicit reversible soft-target/type/metadata conversion | Permission to train on arbitrary material |
| `jevlocal.integration.dataset` | Preserve four frozen partitions and verify leakage certificates | Human truth of labels |
| `decision_ft` | Original native training candidates | Native weights or measured domain quality |
| `decision_deploy` | Strict loopback inference, runtime/export candidates | Browser permission |
| `fastloop.service.invoke` | One serialized, persistent execution core | Codex host browser integration or final task credit |
| `jevlocal.integration.context_tools` | Explicit authorized-context wrappers | Replacement of built-in Codex compaction |

`noul` is mapped by meaning: old arrays are `[yes, no]`; new keys are `false, true`. The compatibility response always stays `shadow=true` and `execution_authorized=false`. It cannot be used as an implicit action approval.

The `/v1/decide` envelope has exactly `request_id`, `observation_id`, `state`, `questions`. Extra metadata belongs in the host task, not this wire envelope. Inference and responses are bounded and checked; no missing question or candidate is silently accepted. Runtime identity includes the shared `decision_ft.data` source and, for native inference, the actual adapter source.

## Current use

Use the chosen environment's Python with the project `src` on `PYTHONPATH`, or explicitly install this project into an authorized isolated environment. No commands below authorize global installation or automatic browser execution.

```text
python -m jevlocal.integration.cli doctor
python -m jevlocal.integration.cli validate-envelope --input request.json
python -m jevlocal.integration.cli bridge --input samples.jsonl --authorization authorization.json --output records.jsonl
python -m jevlocal.integration.cli prepare-dataset --input records.jsonl --output-dir frozen-dataset
```

`authorization.json` maps each input sample ID to exactly `group_id`, `label_source`, `task`, `training_allowed`. All must be supplied from actual authorized metadata. OOD/unknown samples and missing permission are rejected, not relabeled. `prepare-dataset` preserves existing splits; do not send pre-split canonical data through the older random repartitioning CLI.

The original-package tests use preserved example fixtures. Updated HTTP tests use a valid nonempty request and complete fake response because the integrated boundary now checks these even with a test predictor. Browser process fixtures use hidden Windows hosts and separate disposable profiles.

## Qualification and approval

Qualification is a deterministic evidence/quality gate, not an authorization issuer. Its receipts and model/runtime fingerprint must match the actual calibration, evaluation, dataset and export evidence. Host/user approval remains separate. Private approval keys must not be handed to the acting agent in a real deployment; test-only keys are not a production trust boundary.

The fastloop core has a real CDP adapter for isolated fixtures. A supported Codex host adapter has not been connected; do not configure its standalone CDP bridge as a transparent replacement for the host's browser control or compaction tools.

The integrated registry requires `controller=fastloop` and `control_mode=isolated-cdp`; requesting an unsupported host mode is rejected, not routed around the host. Non-dry-run start/resume/fill/refresh recheck the pinned qualification files inside the target lock before connecting to a browser. Status/cancel remain available without a valid model qualification, so a bad model cannot prevent a safe stop. Task approval and signed fill grants remain separately enforced. Deployment-v1's `run`/CLI and browser-MCP launcher are retired; its low-level fixture functions remain only for regression comparison.

The task's `min_probability` must equal the verified policy's `action_confidence_threshold`; the cutoff cannot be lowered or silently tightened without requalifying that action slice. Policy bytes are rechecked against the receipt hash. HTTP replies reject unknown fields while allowing only numeric timing and positive temperature metadata. MCP preserves the configured store path so the storage layer can reject symlink roots on each use.

The files in `configs` ending in `.example.json`/`.example.toml` are templates, not approvals. Zero ports in the registry must be replaced with the actual explicitly selected fixture endpoints; the example task is unapproved and unqualified. The qualification thresholds are proposed explicit policy values, not measured model capability or a statistical safety guarantee. Freeze the task-appropriate policy before opening held-out tests. No real approval key, qualification receipt or model credential is generated by these templates.

## Explicit object/revision applicability preparation

`jevlocal.integration.evidence_scope` is an optional pure preparation step, not
an alternative model or executor. It does not change existing `rank_evidence`,
active native scoring, passive observation or v1 replay defaults.

The caller supplies `query`, the original rank-schema `candidates`, a `target`
with exactly `object_id` and `revision`, and `bindings` keyed by every candidate
ID. Every binding contains exactly `object_id`, `revision` (each opaque string
or explicit null), and `candidate_sha256`. Compute the latter with
`candidate_digest(candidate)` **before redaction**. It binds ID, source, text and
start_line (default 1), using SHA256 of sorted compact UTF-8 JSON. Changed text,
source or origin line requires a new binding. A matching hash only demonstrates
internal consistency, not authenticity of the caller's object claim.

Target values are required, nonempty strings of at most 256 characters. Identity
comparison is exact and case-sensitive, with no trimming, Unicode normalization,
text/source inference or wildcard matching. Control/format/surrogate characters
and values altered by secret filtering are rejected. The caller, not the model,
is responsible for the target's meaning and consistency with the query.
Target and candidate identifiers must use the same caller-declared namespace;
the API does not discover namespaces or disambiguate identical IDs from different
systems. Supply qualified opaque IDs when integrating multiple sources.

- Any known identity conflict: `mismatch`, even if another component is null.
- Both components known and equal: `match`.
- Otherwise: `unknown`, retained separately, never labeled semantically false.

`scope_evidence(..., max_chars=..., rank_max_chars=12000, known_secrets=())` checks
all candidates before partitioning, sanitizes returned content, retains the full
ordered snapshot and counts, and produces `rank_input` containing only matching
candidates in their original order. Snapshot bindings use fresh digests of the
sanitized candidates, not a retained raw secret-bearing version. Redacted secret
values are not included in that snapshot. Source strings are labels, never files to read.
`max_chars` budgets the entire preparation artifact; insufficient budget returns
`requires_larger_budget` without a partial snapshot. `rank_max_chars` is the
independent downstream ranking budget. No text store is written here.

Matching says nothing about factual truth, semantic relevance, freshness, user
intent or permission. Unknown/mismatched records remain recoverable in the full
snapshot and denominator. If none match, `status=no_matches`, `rank_input=null`;
do not invoke rank with an empty list or silently reinsert unknowns.
This means no declared matching candidates, not no relevant evidence.

Example explicit CLI composition (each output must be new):

```text
python -m jevlocal.integration.cli scope-evidence --input scope-request.json --output scope.json --rank-input-output rank-input.json
python -m jevlocal.integration.cli context rank_evidence --input rank-input.json --output primary.json --shadow-runtime runtime.json --shadow-output shadow.json
python -m jevlocal.integration.cli verify-scoped-evidence --input scope-request.json --prepared scope.json --rank-input rank-input.json --primary primary.json --shadow shadow.json --output verification.json
```

Run the second line only when the first returns `prepared`. `no_matches` emits
only the scope artifact even when `--rank-input-output` was supplied. Each
output is published atomically and exclusively, but the two-file publication
is not a transaction: on a second write error, the command rejects and the
first valid artifact may remain. It is not deleted or reported as full success.

`verify-scoped-evidence` rederives the entire partition from the original input,
validates the exact prepared result and derived rank input, then optionally
chains existing passive `load_replay`. Both `--primary` and `--shadow` are
required together. It reads each explicit artifact once, up to 4 MiB; duplicate
keys, nonfinite floats (including exponent overflow), and excessive nesting are
rejected. No input-specified store path is read, no model/action is replayed and
no semantic labels are fabricated. It proves current artifact consistency, not
authenticated historical execution; `origin_authenticated=false` remains explicit.
Distinct complete manifests can have the same matching projection. A modified
excluded item fails verification against the old manifest, but a newly consistent
manifest can be paired with an existing identical rank input. This v1 verifier
does not prove which complete manifest accompanied the original invocation: the
legacy rank sidecar has no manifest-linked invocation ID. Do not call this a
historical parent-manifest or denominator attestation. Such a claim requires a
separate pre-invocation receipt binding the full manifest, actual invocation, and
consumed projection; no identifier or event time is backfilled here.

Report all scope counts separately from any model metric. Excluded candidates
are not model-correct answers. Semantic accuracy/Brier, when independent labels
exist, is conditional on match AND returned AND independently labeled; retain
the original population and do not compare its filtered score to an unfiltered
old diagnostic as evidence of model improvement.

The optional local MCP server also exposes pure read-only `scope_evidence` with
query/candidates/target/bindings and the two budgets. It does not register a global
MCP or start a server automatically. Its nested `rank_input` is only a preparation
result, not an automatic second tool call. Existing MCP rank storage/routing
configuration is unchanged; the CLI combined replay example uses its exact
derived, storage-free input and must not be silently rewritten for another route.

## Forward-bound scoped passive invocation (v2)

The explicit v2 CLI composes scope preparation, unchanged lexical primary
ranking and one bounded passive observer in a single call. It does not install
or register a host adapter. Existing v1 commands/artifacts remain v1; no old
receipt is promoted to an invocation-bound receipt after the fact.

```text
python -m jevlocal.integration.cli scoped-rank-evidence --input scope-request.json --shadow-runtime runtime.json --manifest-output invocation.json --primary-output primary.json --shadow-output shadow.json --shadow-timeout 60
python -m jevlocal.integration.cli replay-scoped-rank-evidence --input scope-request.json --manifest invocation.json --primary primary.json --shadow shadow.json --output verification.json
```

All output paths must be new, distinct files in existing owned directories and
must not alias the input/runtime. The call generates a fresh UUID and canonical
SHA-256 bindings for the entire sanitized scope manifest and exact derived rank
input. It validates the actual prepared projection before publishing the parent
record exclusively, then starts the child. The child validates that same frozen
manifest/projection and emits invocation-bound receipts. Model-facing
state/questions remain identical to the v1 query/candidate rubric: attribution
is carried by a digest-derived envelope request ID, not random metadata added
to model input.

The primary stays lexical and is published before waiting for the observer.
Unavailable native results, timeout and cleanup failures remain visible; they
do not supply semantic labels or replace primary ranking. No matching candidates
means no observer launch, `rank_input=null`, a `no_matches` primary and no shadow
file; omit `--shadow` when replaying that case. A scope-budget failure publishes
no invocation. Partial multi-file publication is not transactional: a later
failure retains earlier valid artifacts and reports failure, rather than deleting
them or overwriting an existing output. A fresh attempt requires new paths and
receives a new invocation ID.

Pure replay rederives scope from the explicit original input, checks the complete
parent record, actual projection, primary and child receipt binding, and performs
zero model/action calls. It rejects a switched parent or invocation even when
the matching projection is identical. It still proves artifact consistency,
not authenticated historical origin, event time, relevance truth, permission,
model quality or successful Codex integration.

Verification for this slice: 172 integration tests passed, including fake
predictor input invariance, swapped invocation/receipt rejection, invalid actual
projection rejection before publication, primary-first timeout, publication
failure, and legacy v1 coverage. One isolated original-Laya native CLI smoke
returned one observation for one matching candidate, retained the mismatch and
unknown in its three-candidate parent manifest, reaped its child, and passed pure
replay. This is a native binding/lifecycle smoke, not a semantic benchmark or
real Codex host test. The task report preserves commands, source hashes and logs.

## Explicit passive MCP session

The optional stdio server now exposes `passive_start`, `passive_collect`,
`passive_cancel`, and `passive_replay` when both operator-owned environment
variables `JEVLOCAL_CONTEXT_PASSIVE_RUNTIME` and
`JEVLOCAL_CONTEXT_PASSIVE_RECORDS` are set. `JEVLOCAL_CONTEXT_STORE` remains
required. Use absolute, owned local paths. Passive runtime and records must be
configured together; configuring active native and passive native modes together
is rejected. The fixed runtime is not a model default switch.

The underlying server retains its five existing context tools. A host that should
expose only this passive integration must allowlist exactly the four `passive_*`
tools. No runtime path, model selector, command or verifier callback is accepted
as a tool argument. Existing examples do not automatically install configuration.
Task/project registration requires its own explicit permission and supported host
configuration loading; a successful `codex mcp get` is not callable-host evidence.

`passive_start` takes explicit `query`, `candidates`, `target`, `bindings` and
bounded character budgets. It freezes a fresh UUID and the complete sanitized
scope manifest, then publishes the unchanged lexical primary before native work
can block it. A background finalizer alone owns finishing and closing the bounded
observer. There is at most one active native observer per session. Saturation
preserves a new invocation's lexical result with unavailable/capacity evidence;
no-match and scope-budget failure do not start native inference.

`passive_collect`, `passive_cancel`, and `passive_replay` accept the invocation UUID,
not a filesystem path. Collection validates complete persisted bindings and never
promotes mere file existence into success. Cancellation and terminal publication
are linearized under the same lock: an acknowledged cancel cannot subsequently
publish an observed native result; a result published first remains complete.
The primary remains unchanged in either case. Cleanup failure quarantines the
session rather than releasing native capacity. Shutdown signals cancellation and
waits a bounded 1.8 seconds for the finalizer; it is not a general guarantee for
abrupt process crashes or every pathological cleanup delay.

Owned record directories reject reparse/link ancestors. Each artifact is staged,
fsynced and exclusively hard-link-published so readers do not see a half-written
file. Multi-file invocation publication is still not a single transaction;
incomplete records stay interrupted. After a server reopen, explicit saved UUIDs
can be collected or replayed without a rescan, a restarted model, or a repeated
action. Pure replay validates consistency with zero model/action calls; it does
not authenticate historical origin or establish semantic truth.

The persisted manifest includes sanitized caller-supplied text. Sanitization is
best-effort, not permission to submit arbitrary private data. Records have no
automatic TTL and remain until separately authorized cleanup. Model outputs stay
non-authoritative and do not authorize training or tool execution. This is an
explicit, single-task passive entry, not interception of every Codex tool call.

Validation for this slice: 184 integration tests passed. Installed MCP SDK 1.30.0
exercised a real local stdio session with the original Laya checkpoint: normal
observation, explicit cancel, graceful-shutdown cancel, and reopen/read-only replay.
The normal native child exited 0; the two cancelled children exited 1 as expected,
with recorded reaping and temporary cleanup. This is SDK/protocol/native lifecycle
evidence, not a callable Codex-host or model-quality acceptance. Real native
60-second timeout behavior was not exercised by those three stdio cases.

## Evidence levels

Keep these distinct: parsed source, mocked/in-memory unit tests, real isolated browser fixtures with a fake predictor, real native model inference, training results, and end-to-end user tasks. Only the matching run can prove each level. No runtime latency or quality gain is asserted without measurement.

## Later actual Codex-host acceptance (2026-09-23)

The earlier SDK-only and CLI-only evidence above retains its original scope. A
subsequent user restart loaded the four task-allowlisted passive tools in the
actual Codex host. Ten host tool calls exercised no-match start/collect/replay,
normal start/collect/replay, and explicit start/cancel/collect/replay. Normal
observation returned 1/1; the cancelled case returned 0/1 with cancelled status.
Both owned native children were reaped and their temporary artifacts cleaned.
The lexical primary, complete manifest, fresh invocation bindings and persisted
records were verified; all three replays made zero model and action calls.

This closes the earlier missing-callable-host evidence, not semantic quality or
universal automatic following. These records were created after that restart;
read/replay recovery after a further natural host restart remains untested.
The supported active browser adapter is a separate requirement and is not proved
by passive MCP calls. The original Laya runtime and four-tool task-only scope
remain unchanged.

Authoritative report and machine evidence:
`C:/Users/26516/Documents/Codex/2026-09-23/https-chatgpt-com-share-6ab31711-8380/outputs/真实Codex被动接入验收.md`
and `passive-host-evidence.zip` in that same output directory.

## New-mechanism diagnostic and program feedback (2026-09-23)

Eight fresh synthetic SQLite/configuration roots supplied 32 candidate slots per
checkpoint: 28 matched known labels (8 true, 20 false), plus 4 applicability-unknown
slots excluded before inference. Actual SQLite WAL reads and an owned resolver
established the labels; complete event/key inventories were available in each
single-candidate query. All data, metrics, code and checkpoints were frozen before
inference. Twenty-two CPU tests passed; both tokenizers used at most 398 of 1024
tokens without truncation.

The original and existing 20-update checkpoints each ran once and returned 28/28
observations. Both children exited 0 and were reaped. Original TP/TN/FP/FN were
8/3/17/0; trained values were 8/5/15/0. Correct answers improved only from 11/28 to
13/28: configuration improved from 5/14 to 9/14, but SQLite regressed from 6/14 to
4/14. Neither route rejected every candidate in any of the four all-negative
roots. This is development evidence of remaining false acceptance, not a blind
test, calibrated confidence, general improvement, or deployment qualification.

The existing `assess_feedback` interface additionally bound all 56 observations
to re-executed program oracles identical to the prefrozen ones. Proposition labels
were independent; `executed` remained empty and action outcome unknown. All records
retain `training_allowed=false` and `execution_authorized=false`. Three additional
feedback tests passed. No model was rerun for that feedback pass, no new training
or calibration occurred, and the earlier calibration/test partitions remained
unopened. No default runtime or execution permission was changed.

Authoritative report and text evidence:
`C:/Users/26516/Documents/Codex/2026-09-23/https-chatgpt-com-share-6ab31711-8380/outputs/新机制语义诊断.md`
and `new-mechanism-evidence.zip` in that same output directory. Broader quality,
representative real-task feedback and active-browser end-to-end qualification
remain outstanding; successful inference alone does not close those gates.


### Original-model effective-source diagnostic (2026-09-23)

A later frozen development comparison used the original Laya checkpoint only,
with forty identical candidate-state slots per wording and fresh invocation
bindings. Generic relevance gave TP8/TN1/FP31/FN0; explicit effective-source gave
TP8/TN0/FP32/FN0. Both-world counterfactual correctness was 0/8 in both, complete
all-negative-root rejection 0/2. All eighty observations returned and validated;
25 CPU tests passed. This fixed wording did not improve the failure, and neither
the default model nor production question was changed. No training or old held-out
partition reads occurred. These development results confer no semantic or execution
qualification. A supported active host browser adapter remains a separate missing
API/permission-contract dependency; isolated CDP is not its fallback.


## Explicit skill-description passive shadow (2026-09-23)

`context suggest_skills --shadow-runtime RUNTIME.json --shadow-output NEW.json`
now observes only caller-supplied enabled skill descriptions. The lexical primary,
including its order and budget outcome, remains unchanged. Native scores describe
uncalibrated description relevance, not verified skill availability, capability,
permission, or selection for execution. No skill discovery or installation occurs.

The adapter projects the sanitized closed set into the existing scoped observer;
each nonempty invocation receives a fresh binding. The sidecar separates the real
skill-primary hash/ranking/verification from projection-only citation feedback.
Empty sets do not start a model. Valid inputs exceeding projection capacity keep
their lexical primary and produce an unavailable observer. Existing passive MCP
registration and rank-evidence observer behavior remain unchanged.

Independent root verification passed 19 pre-change primary equivalence cases and
194 integration tests (exit 0). One preregistered original-Laya offline CLI smoke
returned two observations, kept primary file/stdout bytes identical, and reaped
and cleaned the child. This is feature wiring evidence, not semantic quality,
latency improvement, automatic all-task capture, or host skill-MCP acceptance.
No training, calibration, default checkpoint change, or configuration change ran.

The user's expanded scope combines twenty source-project feature sets using a
single shared decision/runtime architecture. Authoritative scope and gates:
`C:/Users/26516/Documents/Codex/2026-09-23/https-chatgpt-com-share-6ab31711-8380/outputs/本地Jev功能总表.md`.
After the user flagged inadequate trained-model quality, the next priority is a
capability-to-feature gap analysis and bounded model/data improvement proposal,
not additional wrapper implementation or repeated fixed-weight prompt diagnostics.
The typed batch CLI remains pending after read-only contract discovery; it cannot
substitute for semantic acceptance. New optimizer updates require fresh approval.


## 第二轮实际训练结论（2026-09-23T15:58:28.059750+00:00）

新增训练授权已执行。三类自建程序验真数据1,536项、一轮、192次实际优化器更新；训练子进程212.16秒。新test192项／48根，原始96/192→候选113/192（50%→58.85%）；配置50%未变且仍全判true，完成证据75%，SQLite51.56%并误拒15/32正例。总体正召回由100%降至84.38%；冻结的有限改进与严格质量两门均未过。零更新及训练后保存重载logit差皆0，四相进程已退出，原权重与默认路由未改。

此为训练流程成功而语义未达标，不是二十项功能完成。candidate留在 D:/Projects/JevLocal/models/experiments/laya-capability-sft-02/fit/epoch-1，不晋升到任何消费者。本批rubric是明确断言/要求成立，非泛证据相关性；不可据false自动删日志。GPT/Claude新审到达在训练之后、test之前，已如实记录，未改冻结题目、门槛或重训。反事实迁移、旧能力遗忘及通用相关性均未验。

本轮预算用毕，不自动追加训练、校准、换checkpoint或阈值。下一步仅据现有输出只读分型，形成最小新干预，不再优先扩写外壳。总体目标保持active、未完成；自然重启与主动宿主接口仍各自待验。当前无运行中模型子进程。详见本task outputs/第二轮本地模型训练验收.md 与 work/laya-capability-sft-02/root-acceptance.json。


## F05: explicit completion-claim verification

The CLI now consumes explicit local completion claims and receipt artifacts:

```text
python -m jevlocal.integration.cli verify-completion --input INPUT.json --output NEW-REPORT.json
python -m jevlocal.integration.cli verify-completion --input INPUT.json --native-runtime RUNTIME.json --output NEW-REPORT.json
```

The first form does not load a model. Evidence that passes mechanical checks but
still requires semantic interpretation returns **unknown**, not supported. The
second form uses an explicitly configured local Predictor through a dedicated
three-way Choice question: supported / contradicted / unknown. It does not reuse
relevance scores as completion decisions and has no lexical or cloud fallback.
The existing native-runtime configuration and process-start offline/safe-loading
flags apply. No checkpoint is selected or promoted automatically.

### Input and receipt boundary

The input is an object with exactly `schema_version: 1`, `workspace`, `run_id`, `started_at`, `claims`, `evidence`.
The workspace is an existing absolute local directory. The run ID and UTC window
start identify the caller-selected invocation; they are not inferred from a
filename or the newest log. Each claim has `id`, `text`, `kind`, `evidence_ids`.
Kinds are `tests_passed`, `change_present`, `completion`. Each evidence reference
has `id`, `kind`, `path`, `sha256`; kinds are `test_run`, `tool_result`, `diff`.

Every referenced artifact is a bounded strict UTF-8 JSON receipt, read only from
an explicitly named regular file inside the workspace. Its SHA-256 must match
current bytes. Each receipt carries `schema_version: 1`, `run_id`, `subject_files`,
where subject files are nonempty `{path, sha256}` bindings to the files the
observation concerns. These current bytes are also checked. `evidence_checked_at` records that observation;
`binding_observation_scope: current_at_evidence_validation_only` does not promise
that files remain unchanged during inference or after the report. Symlinks, Windows
reparse points, UNC/device paths, alternate streams and paths outside the
workspace are rejected. No directories are searched and no command from a
receipt is executed.

A test receipt adds `started_at`, `ended_at`, `status`, `command`, `exit_code`, `tests`, `output`.
`command` is argv data, and `tests` contains nonnegative integer counts for
`passed`, `failed`, `errors`, `skipped`. A tool receipt uses the same execution
fields without `tests`. A diff receipt instead adds `observed_at` and `diff`.
Execution status is one of `executed`, `failed`, `skipped`, `not_run`.

These are **caller-supplied receipts, not cryptographic execution attestations**.
The verifier observes current files and validates their consistency; it cannot
prove that a caller did not fabricate a receipt. A bare caller claim, model
self-report or plausible exit code is not independent proof of a real operation.
Collectors must retain the actual command result, invocation, timestamps, and
artifact/subject bytes. An authentic receipt also proves only its recorded scope.

### Guarded three-state output

- Missing references, stale run/window, unrun/skipped evidence or zero executed
  tests cannot become supported, even if a semantic predictor says supported.
- A current required failed command/test contradicts a success claim when no
  missing/stale evidence takes precedence. Failure and skipped counts stay visible.
- `tests_passed` requires test evidence; `change_present` requires a diff;
  `completion` requires both. All listed evidence is required, not optional.
- Eligible claims still need typed semantic support. Invalid model output,
  changed runtime identity, calibration, route failure or invalid evidence is
  rejected, never replaced by a success-shaped fallback.
- The report binds each claim to evidence hashes, exit codes, test counts and reasons;
  `request_sha256` binds the complete canonical input, including claim text. Top-level
  `status: verified` and CLI exit 0 mean that verification ran, **not that every
  claim is supported**. Consumers must inspect each `claims[].status`.
- `execution_authorized` and `action_authorized` are always false. No report
  grants permission to execute, merge, publish, or skip required review.

Evidence is validated before the optional native model is loaded; if no claims
need a semantic decision, that route is not loaded. Output creation is exclusive
and never overwrites a prior report. Inputs remain the recovery source on errors.
Budgets are explicit and oversized input is rejected without silent truncation.

This implementation provides the actual consumer path and deterministic guard
coverage, not F05 model-quality acceptance. Native adapter unit tests use an
explicit fixture Predictor; they are not a native model run. Semantic quality,
representative task evaluation, native latency/resource bounds, and broader
F01-F20 acceptance remain separate gates. The failed external-question-bank SFT
candidate is not promoted to this consumer.


## F04: actual diff risk triage

```text
python -m jevlocal.integration.cli review-diff --input REVIEW.json --output NEW-REPORT.json
python -m jevlocal.integration.cli review-diff --input REVIEW.json --native-runtime RUNTIME.json --output NEW-REPORT.json
```

This is a read-only consumer of **actual before/current file bytes** and explicit
related contract files. It constructs its own structured hunks; a model cannot
supply or rewrite file names, line ranges, or change hashes. It neither edits the
reviewed files nor invokes Git, tests, shell commands, or a second browser core.
The no-native form returns unknown semantic risk while retaining review duties.
The explicit native form uses a separate review rubric, not relevance or
completion-support scores.

### Explicit input

The input has exactly `schema_version: 1`, `workspace`, `request_id`, `changes`, `contracts`.
`workspace` is an existing absolute local directory. A change has
`{path, before, after}`: `path` is its canonical workspace-relative slash path;
`before` and `after` are null or `{path, sha256}` absolute local file bindings.
An after binding must point to the current workspace file itself. Null after is
accepted only when that current path is absent. Before snapshots are supplied
explicitly, including for this non-Git project; null before declares a creation.
Both null and unchanged bytes are rejected. Empty-file creation/deletion retain
structural bindings and review floors but remain unknown without invoking the
semantic judge: an empty hunk alone cannot express their differing effects.
This does **not** attest a Git base
revision, historic file absence, authorship, or permission to delete a file.

Each contract has `id`, `path`, `sha256`, `applies_to`, `category`, `mandatory_review`, `reviewer`.
Its nonempty `applies_to` list names exact changed relative paths, not patterns.
Categories are security, permissions, interface, persistence, general. Reviewers
are main_model or human. IDs use 1..128 ASCII letters/digits or `_.:-`.
Contract bytes and metadata are evidence data, never tool instructions. Their
hashes are recomputed; files are not discovered implicitly. Symlink/reparse,
remote/device/alternate-stream and outside-workspace paths are rejected.

The consumer reads UTF-8 text, preserves line-ending bytes in file hashes, and
builds grouped before/after line hunks with three lines of context. The report
retains file/change hashes and exact hunk line records. Contract citations bind
source paths, hashes, and line ranges. Request hashing binds the complete input.
All bindings are observations at validation time, not promises that files remain
unchanged during inference or after the report.

### Review floors and typed suggestions

Every changed hunk requires main-model review. Missing applicable contracts or
an unavailable optional semantic judge produces unknown, not a low-risk default.
Sensitive contract categories and explicit mandatory-review obligations persist
in the report. A mandatory/sensitive human-review contract forces human review;
model low scores or negative category predictions cannot remove that floor.
Higher model risk or positive sensitive categories may add human escalation.

The dedicated native adapter asks one four-level Score question and four Noul
questions for security, permissions, interface, persistence. Public typed-reply
validation, stable runtime fingerprint, and uncalibrated output are required.
The whole batch is checked before the first inference. Invalid model replies or
route failures are rejected without a lexical/cloud fallback. Native suggestions
remain advisory and semantically unqualified; no checkpoint is auto-selected.
The existing native-runtime configuration and process-start offline flags apply.

`status: reviewed` and exit 0 mean triage ran. They do not approve the code.
`execution_authorized`, `action_authorized`, `merge_authorized`, `review_complete`
are always false. Inspect every `changes[].hunks[]` obligation and risk state;
even a score of zero cannot authorize merging or skip a required review.

Input, source, contract, hunk, native-envelope, and output budgets are bounded;
oversize is rejected without truncating evidence. Output files are exclusive new
artifacts. No native model is loaded when there are no contract-bound hunks to
judge. Failed validation cannot leave a partial report.

Fixture tests establish binding, typed routing and mandatory-review behavior,
not native semantic quality. Real representative review quality and native
resource/performance acceptance remain open. This consumer does not qualify a
failed training candidate or complete the full F04/F01-F20 objective by itself.


### Explicit represented-checkpoint route (F04 and F05)

```text
python -m jevlocal.integration.cli verify-completion --input INPUT.json --state-json-runtime CONFIG.json --output NEW-REPORT.json
python -m jevlocal.integration.cli review-diff --input REVIEW.json --state-json-runtime CONFIG.json --output NEW-REPORT.json
```

`--state-json-runtime` is mutually exclusive with `--native-runtime`. It selects
`StateJsonNativeRuntime` explicitly through the separate
`state_json_uncalibrated_v1` bridge. The generic F04/F05 builders reject a
checkpoint containing `state_representation.json` before loading a model; a
marker never triggers an automatic loader switch. The existing calibrated
SemDecide advisory route and its Score sidecar are not reused. All three types
remain raw: `calibration: null`, temperature 1, calibrated false, and
semantic quality unqualified. No config or checkpoint is auto-selected.

The bounded operator config has exactly `schema_version: 1`, `route`,
`spec`, `expected_identity`, `expected_fingerprint`, `expected_binding`,
`expected_sources`, `expected_checkpoint_files`,
`expected_adapter_sources`, `expected_base_cache`, `calibration`. The exact spec is native Kev,
`purpose: development_evaluation`, an absolute local checkpoint directory,
CPU or CUDA, fp32 weights and amp none. Pin all 14 native source modules, the
complete flat nine-file checkpoint including its representation marker, and
the two bridge/helper modules `jevlocal.integration.represented_native` and
`jevlocal.integration.state_json_advisory`. Extra files, changed hashes,
symlinks/reparse points, marker/binding disagreements, and inconsistent native
identity/fingerprint are rejected. These pins identify caller-selected bytes;
they are not permission or model-quality qualification.

`expected_base_cache` is mandatory even for schema version 1 and has exactly
`base`, `revision`, `hf_home`, `hub_cache`, `files_sha256`. `revision` is an
explicit immutable 40-character lowercase hex commit, never inferred from a
cache ref or automatically selected. The flat snapshot is derived from
`hub_cache/models--<owner>--<repo>/snapshots/<revision>`. It must contain
`config.json`, `tokenizer.json`, `tokenizer_config.json`, optional `merges.txt`
and `vocab.json`, and either `model.safetensors` or an index and all and only
its safe-named `.safetensors` shards. Every listed file must be regular, local,
non-reparse, SHA-256 matched, and no extra entry is allowed. JSON is strict
UTF-8 with no duplicate keys, BOM or nonfinite constants. Existing frozen
operator configs lacking this field fail closed; an operator must freeze a new
config with current adapter-source hashes. Old freezes are not silently edited.

Pure config admission resolves and hashes module origins without executing their
packages or importing torch. It checks identity consistency, snapshot bytes and
inventory, not actual GPU or environment availability. The real explicit builder
requires the existing four process-start offline/safe-loading flags, exact
`HF_HOME` and `HF_HUB_CACHE` environment paths, matching loaded or controlled-
import Hugging Face and Transformers effective cache constants, and no divergent
legacy cache override. Before native construction it safely reads checkpoint
`head.pt` using `torch.load(..., map_location='cpu', weights_only=True)` and
requires its `base` and `base_revision` to match the explicit cache pin. It
then constructs only StateJsonNativeRuntime and compares
its actual full identity, including observed environment, with the config.
It never edits environment flags, repairs config, downloads, substitutes a model
or changes precision to make a mismatch pass.

The complete config digest and returned route metadata bind `expected_base_cache`.
The unchanged native runtime fingerprint algorithm does **not** itself bind that
parent cache. Cache/metadata admission and source pins are not a tokenizer/model
execution test or a semantic-quality claim.

Before each request's forward calls, the bridge prepares the bound state-JSON
representation and preflights every atom with the native encoder: exact question
and candidate order, no state truncation, and 1..512 encoded tokens per atom.
This is separate from the consumers' full-batch public-envelope validation.
An overlarge represented request fails rather than trimming its evidence; a
later request may still reject after an earlier request was inferred, but no
partial consumer report is returned. Owned access is serialized, cache cleanup
is unconditional, and raw typed output plus runtime/config identity is checked.
Reports include both the existing consumer adapter/rubric fingerprint and the
explicit representation route/config/binding/source pins. All deterministic
evidence guards and review/authorization floors remain unchanged.

The route remains lazy: rejected evidence, stale/unrun completion receipts, or
review hunks lacking contracts do not read the operator config or load a model.
Fixtures and pure admission demonstrate wiring and rejection behavior only.
Real native semantic quality, performance and resource acceptance require a
separate frozen run; implementing this bridge does not qualify or promote any
trained checkpoint.

### Native semantic projection and evidence preservation (F04/F05)

The typed native adapters distinguish semantic evidence from mechanically checked
bookkeeping. The core still reads and validates the complete original artifacts;
its full receipt, current-file, change/hunk and contract bindings remain in the
report. No path, hash, timestamp or permission check is moved into a model.

F04 sends the exact relative file path, complete structured hunk and line text,
and every applicable contract's ID, category, review duties, exact text and line
range. It omits only opaque change/hunk hashes and each contract artifact's
absolute path/hash from the model's state. F05 preserves claim ID/text/kind,
evidence ID/kind, exact subject file paths, execution status, argv, exit code,
test counts, raw output and diff where applicable. It omits only receipt
container/schema/run/time and integrity bookkeeping from the semantic view.
Commands, paths, claims, output, diffs and contract text are never shortened or
summarized to fit. A claim that needs omitted metadata has no semantic proof in
that view; the rubric requires unknown rather than inferred support.

The versioned projection and rubric appear in route metadata. Each exact input
item has two deterministic SHA-256 bindings, computed before inference:
`prepared_item_sha256` covers the original prepared item and
`semantic_request_sha256` covers the actual projected state/questions. These
are output provenance, not tokens fed back into the model. They do not attest
caller authorship, grant authority, or replace the core's actual-byte checks.

The encoder limits remain unchanged: the native default state limit is 384
tokens and the represented bridge total limit is 512. Larger semantic evidence
still rejects without truncation. The capacity repair does not establish native
semantic accuracy, broad input coverage or production qualification.

## F03: explicit closed-catalog model routing advice

`advise-model-route` is a local consumer for an explicit model/effort catalog.
It returns advice for primary-model review, not a model switch. The default
metadata-only route validates and screens the catalog but returns `unknown`
with `semantic_unavailable`; it does not import or initialize a native model.

```text
python -m jevlocal.integration.cli advise-model-route --input routing.json --output new-report.json
python -m jevlocal.integration.cli advise-model-route --input routing.json --native-runtime native.json --output new-report.json
python -m jevlocal.integration.cli advise-model-route --input routing.json --state-json-runtime represented.json --output new-report.json
```

The two runtime flags are mutually exclusive. Their existing explicit operator
config formats remain unchanged. Generic routing rejects representation-bound
checkpoints; the explicit StateJson route is uncalibrated, with no fallback.
Both config files are bounded to 64 KiB and read lazily, after complete request
validation and only if an eligible candidate exists. Rejected input and an
existing output file do not initialize the judge. Reports use new atomic files;
errors do not publish partial output or echo private payloads.

The exact version-1 request has `schema_version`, `request_id`, `current_primary`,
`task`, `budget`, `catalog`, and `profiles`. The executable public CPU example is
[`public-cli-input.json`](../runtime/f03-model-routing-01/public-cli-input.json);
its model names, capacity, costs and timestamps are fictitious, not host metadata.
The complete bounded field contract is
[`contract.json`](../runtime/f03-model-routing-01/contract.json).

- `current_primary` is a supported catalog model/effort pair; it need not be
  included in the selectable profiles and is never changed by this consumer.
- `task` states summary, difficulty, risk, required context size and tool IDs.
  `budget` states optional maximum estimated cost/latency and a required
  `cost_unit`. All costs use the same caller-declared per-task unit; they are
  estimates, not verified current provider prices.
- `catalog` supplies a source description, UTC observation timestamp and 1–32
  distinct models, with explicitly supported reasoning efforts. Unknown context
  capacity, tool support, cost and latency are represented by `null`.
- `profiles` contains 1–15 distinct supported model/effort pairs with unique IDs.
  The native Choice adds reserved `__abstain__`, keeping the existing protocol's
  16-candidate limit without batching, truncation or a protocol change.

Known context/tool/cost/latency violations exclude a profile deterministically.
Missing metadata stays explicit in per-profile `unknowns`, not fabricated or
treated as proven fitness. If all profiles are excluded, no judge is loaded.
The native adapter sends one Choice with the full task, budget and surviving
candidate metadata. It validates the native fingerprint, typed probabilities and
uncalibrated reply; out-of-set selection and failures reject without retry.
Prepared-item and actual semantic-request SHA-256 bindings preserve provenance.

Reports retain request/catalog hashes, original primary, budget, screening reasons
and route metadata. Reasons are deterministic status/class labels, not invented
model explanations. `catalog_provenance` is always
`caller_supplied_not_host_authenticated`; `catalog_freshness` is `not_verified`.
All reports require main-model review and set `execution_authorized`,
`model_switch_authorized`, `provider_change_authorized`, and
`current_primary_changed` to false. `semantic_quality` remains `unqualified`.

CPU tests and the standalone metadata-only CLI establish wiring, binding and
rejection behavior only. Actual-host catalog attestation, real native semantic
quality, usable token-capacity coverage and resource/performance acceptance are
not established by this implementation. The existing native encoder/represented
token limits still apply; oversized semantic inputs reject, never truncate.

## F06: explicit-root repository navigation

`navigate-repository` observes real, immediate directory entries within one
explicit local repository root. A supplied local judge may select a directory
for the next bounded hop or select a regular file. Selection does not read the
file's contents, edit it, run it, delete it or authorize another tool. This is
distinct from `rank_evidence`, whose source labels are not filesystem access.

```text
python -m jevlocal.integration.cli navigate-repository --input navigation.json --output new-path-report.json
python -m jevlocal.integration.cli navigate-repository --input navigation.json --native-runtime native.json --output new-path-report.json
python -m jevlocal.integration.cli navigate-repository --input navigation.json --state-json-runtime represented.json --output new-path-report.json
```

The exact request fields are `schema_version: 1`, `request_id`,
`repository_root` (canonical absolute local directory), `current_directory`
(`.` or a canonical root-relative directory), `query`, `candidate_names`
(`null` or 1–15 explicit immediate child names for the first hop), and
`max_hops` (1–4). The bounded field and exclusion policy is recorded in
[`contract.json`](../runtime/f06-repository-navigation-01/contract.json).
The public example directory contains only synthetic files.

Discovery is not recursive: only a judged eligible directory advances the next
hop. UNC/device/ADS paths, traversal components and symlink/reparse ancestors
reject. Link/reparse children, special files and the fixed credential-name
exclusion set are not candidates; only exclusion counts are reported. An
explicitly requested excluded or unavailable name rejects. No file content is
read or hashed. The deny-list does not assert that arbitrary filenames cannot
contain secrets; not reading content is the primary confidentiality boundary.

At most 256 immediate names are inspected per hop; one additional iterator
presence probe detects overflow without inspecting that entry's name or stat.
More than 15 eligible
candidates, or the scan limit, returns `requires_candidate_subset` instead of
calling the judge on a silently truncated list. An explicit subset is recorded
as such and never represented as an exhaustive inventory. Empty directories,
abstention, missing semantic inference and the hop limit remain distinct
outcomes. The same lazy judge is reused across bounded hops without retry or
fallback. The default no-runtime mode records one actual directory snapshot and
returns `unknown`, not a fabricated semantic recommendation.

Candidate IDs and `snapshot_sha256` bind the observed root/current identities,
relative paths, kind, size, inode/device and timestamps. They are metadata-snapshot
identity evidence, **not file-content hashes**. The consumer rechecks path
components and observed identities around inference and before descent; a
change rejects without publishing a partial successful report. Absolute root
paths stay out of the native semantic state. The native Choice contains the
query, current relative directory and exact name/path/kind candidates, plus
`__abstain__`; native identity and uncalibrated typed replies are checked. Route
metadata binds both the prepared item and actual semantic request by SHA-256.

The native adapter itself only consumes supplied prepared metadata; it does not
attest a filesystem path or enforce the consumer's credential inventory policy.
Filesystem callers must use `navigate_repository`, which applies those rules
before creating the judge. Direct native metadata replies grant no access.

Input and lazy runtime configs are bounded to 64 KiB. CLI output is a new atomic
file; existing reports are never overwritten. Generic and StateJson routes stay
explicit and mutually exclusive. All output requires main-model review, has
`semantic_quality: unqualified` and grants no execution, write or deletion
authority. Native encoder token limits remain unchanged.

### Windows handle boundary

Windows validates request paths lexically, then admits one `WindowsDirectoryLease`
for the entire walk. The local fixed/removable drive anchor is opened once;
each component is opened with user-mode `NtOpenFile`, relative to its retained
parent handle. `OBJ_DONT_REPARSE`, `FILE_OPEN_REPARSE_POINT` and inspected
attributes reject reparse traversal. Handle-based directory enumeration uses
bounded 64 KiB buffers with validated record offsets and UTF-16 lengths.
Unsupported API/filesystem, access or sharing failures reject; there is no
Windows pathname fallback.

Root, ancestor and candidate directory handles request `LIST_DIRECTORY` and
exclude `FILE_SHARE_DELETE` **before inference**. A metadata-only directory
probe does not establish that sharing conflict; the directory handle must take
over after an identity comparison. Ordinary files request only
`READ_ATTRIBUTES | SYNCHRONIZE`, never file-body read or delete rights. These
attribute-only file handles do **not** prevent rename/replacement. Each file
name is therefore reopened relative to its retained parent for the existing
identity rechecks. Redundant probes close promptly; original candidate/ancestor
anchors remain until the lease ends. The limits are 256 path components and 512
active handles; exhaustion or any close error prevents a successful report.

The five identity keys are unchanged. On Windows, `device` is volume serial,
`inode` is the 128-bit file ID, `size` is end-of-file, `mtime_ns` is last-write
time and `ctime_ns` is **metadata change time**, converted from FILETIME to Unix
nanoseconds. These are opaque snapshot bindings, not historical `os.stat`
timestamp equivalence.

`path_concurrency_guarantee: windows_handle_relative_no_reparse` identifies this
Windows mechanism, not a permanent path lock or universal security certificate.
The public fixture actually verifies junction rejection, two-hop navigation,
an independent directory rename sharing conflict, successful rename after
closure, and process handle-count restoration. Ordinary-file checks are still
instantaneous; mutation after a check or after return is possible. A later
content/tool consumer needs its own safe admission, not this report as an
access token. Existing privileged handles, filter drivers and other filesystems
are not qualified. Non-Windows retains
`stable_snapshot_not_handle_anchored`; that route was not executed on the
Windows host. See the measured stage
[`root-verdict.json`](../runtime/f06-windows-handle-navigation-01/root-verdict.json).

**Scope limit:**
CPU filesystem fixtures and fake-Predictor interoperability do not establish
real native semantic quality, installed host-consumer integration, performance
acceptance, or full F06 qualification. Missing dynamic references or a model's
selection is never evidence that unselected files are safe to delete.

## F08: local source-bound claim support

`jevlocal.integration.claim_support.verify_claim_support(request, judge_factory=...)`
judges whether explicitly supplied excerpts support or contradict a claim. It
does not substitute relevance ranking for support, retrieve sources, authenticate
their provenance, or treat a verdict as real-world truth. Source references are
labels, never files to open or URLs to fetch.

```python
from jevlocal.integration.claim_support import verify_claim_support

request = {
    "schema_version": 1,
    "request_id": "public-example",
    "claim": "The example release contains twelve modules.",
    "sources": [{
        "id": "release-note",
        "source_ref": "caller-supplied public release note",
        "start_line": 20,
        "text": "The example release contains twelve modules.\n",
    }],
}
report = verify_claim_support(request)
# No judge: status=unknown, reason=semantic_unavailable, not lexical proof.
```

The input is strict JSON with exactly those four top-level fields. It permits
0–15 uniquely identified excerpts; each has exactly `id`, `source_ref`,
`start_line`, and `text`. Claim and excerpt text each permit at most 8192 code
points, source references 2048, and the full canonical UTF-8 request 128 KiB.
No truncation occurs. IDs, valid UTF-8, source line ranges, field types and
unknown fields are checked before creating a lazy judge. Empty sources remain
`unknown/no_sources` without creating a runtime.

A configured local judge evaluates every supplied source once. Per-source
Choice labels are `supports`, `contradicts`, `mixed`, and `insufficient`.
Both support and contradiction, including a mixed source, produce `conflicting`;
all insufficient sources produce `unknown`. Exact source IDs and normalized,
finite, uncalibrated probabilities are validated; malformed/partial output or
an exception rejects the whole call rather than inventing missing evidence.

The consumer, not the model, constructs each citation from the exact original
excerpt, caller's source label, declared starting line, computed ending line
and exact UTF-8 SHA-256. Line counting uses Python `str.splitlines()`. These
citations identify the supplied excerpt; they neither authenticate its source
nor claim to locate a narrower supporting sentence. All source results remain
in input order, including insufficient ones. Full output is bounded to 256 KiB.

`NativeClaimSupportJudge` uses the existing native Predictor and actual typed
envelope/reply validators. The semantic state contains the full claim and
excerpt, not an ID-only proxy. Every source receives the same explicit support
rubric; quoted instructions are evidence data, not new instructions. Route
metadata binds each prepared source and its exact semantic request by SHA-256.
`build_native_claim_support_judge` rejects representation-bound checkpoints;
`build_state_json_claim_support_judge` selects the explicit represented route.
Neither route silently falls back, recalibrates, or bypasses token limits.

Results always retain `semantic_quality: unqualified`, `calibrated: false`,
`provenance: caller_supplied_unverified`, main-model review, and false execution,
write and deletion authorization. Probabilities describe uncalibrated model
outputs, not calibrated truth confidence. Opaque `route` metadata confers no
authority. No source text is stored by this function.

This local consumer is distinct from host tool registration. Its CPU and
fake-Predictor acceptance does not establish real semantic quality, supported
MCP-host interoperability, or the other F08 filtering, ranking, classification
and extraction requirements. Those requirements remain in the full goal.
Measured local evidence is recorded in
[`root-verdict.json`](../runtime/f08-local-claim-support-01/root-verdict.json).
