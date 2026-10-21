# JevLocal Pi extension

This is an opt-in **advisory-only** Pi extension. It registers `jev_ask` for public typed `noul` / `choice` / `score` questions and `jev_choose_tool` for a suggestion among Pi's **currently active** tools. It neither calls the suggested tool nor authorizes execution, shell commands, accounts, browser control, or other actions. A probability is not permission.

## Host and source pin

- Official host: [`@earendil-works/pi-coding-agent` 0.87.1](https://github.com/earendil-works/pi/blob/d6af72e1857cfb10b41d8ff8e69f0d72b4cf6d31/packages/coding-agent/package.json), commit `d6af72e1857cfb10b41d8ff8e69f0d72b4cf6d31`.
- API basis: [official extension guide](https://pi.dev/docs/latest/extensions) and pinned [`ExtensionAPI` types](https://github.com/earendil-works/pi/blob/d6af72e1857cfb10b41d8ff8e69f0d72b4cf6d31/packages/coding-agent/src/core/extensions/types.ts).
- `y0usaf/pi-jev` informed the feature shape only. No cloud Jev backend or code is copied.

## Explicit local configuration

Set both environment variables before starting a trusted Pi session:

```text
JEVLOCAL_PI_PYTHON=<absolute path to the intended local Python executable>
JEVLOCAL_PI_CONFIG=<absolute path to an existing local context-native config>
```

Do not point either variable at a command, relative path, checkpoint without an explicitly admitted configuration, or a still-training candidate. An unqualified experimental checkpoint may only be used within a separately frozen development evaluation, never as a promoted default or evidence of semantic qualification. The extension resolves its shipped `src/jevlocal/integration/pi_worker.py` relative to its own file; it does not accept a worker path from tool parameters. Missing/relative/non-file Python, config or worker paths fail before spawning or loading a model. The child is created lazily on the first explicit tool call with `-I -B`, `shell:false`, `windowsHide:true`, and only a narrow system-path environment allowlist. Configuration and native identity checks remain the Python worker's responsibility; the bridge will not guess or substitute a checkpoint.

Load `pi/extension.ts` using Pi's documented `--extension <absolute extension.ts path>` mechanism in a separately frozen, isolated host test within the user's existing authorization. This does not authorize global installation, account/provider changes, or unbudgeted model execution. The original source slice did not install or run Pi; subsequent isolated SDK and native evidence is recorded below.

`jev_ask` takes `observation_id`, `state`, and `questions`. Only question `type`, `instructions`, and optional `criteria` are public; labels and extra question fields fail. One request has at most 128 KiB of UTF-8 JSON, 1–8 questions and 2–16 `choice`/`score` options. `jev_choose_tool` takes `observation_id`, `state`, `instructions`, and optional `candidate_names`. The optional names form an explicit 2–16-name closed set; unknown, inactive, duplicate, empty or oversized sets fail rather than being truncated. Without it, all currently active tools with current `getAllTools()` metadata are candidates, subject to the same limit. The extension's own two tools are excluded. It checks the whole candidate set again before returning a suggestion.

One owned stdio worker is kept per session. Only one request may be in flight. Session replacement and shutdown cancel any pending caller result, then send stdin EOF so the old worker can finish and discard that response before closing. They wait up to five seconds for exactly `close(0, null)` before permitting another spawn. No pending request is replayed. A 120-second timeout, abort, malformed/stale response, extra stdout or transport failure attempts to terminate the owned child; a forced or otherwise unconfirmed launcher close leaves the bridge terminal, not recovered. Uncertainty remains after the child reference is cleared. An already-aborted request that never spawned a worker does not create such cleanup uncertainty. The bridge checks the worker's reply against the original observation, candidate keys, probabilities, runtime fingerprint and explicit uncalibrated/non-authority metadata. Tool output exposes an advisory result, not host permission.

## Opt-in worker resource policy

The host may additionally set `JEVLOCAL_PI_RESOURCE_POLICY` to an absolute,
regular JSON policy file. The bridge passes it as `--resource-policy`; it does
not add it or other ambient variables to the child environment. Changing the
policy path while reusing a worker poisons and closes that worker. A new session
must wait for the previous child's confirmed close.

The policy is separate from the strict context-native configuration. Its exact
fields are `schema_version` (1), `config_sha256` (SHA256 of the config file bytes),
`device` (`cuda`), `max_cuda_allocated_bytes`, `max_model_loads` (1),
`max_tokenizer_loads` (1), `max_forward_calls` (1–4096), and `max_wall_seconds`
(1–86400, from the first valid request). Integer limits reject booleans and
floating-point values. The allocator ceiling must be below the actual device's
total memory. The policy file is limited to 8192 bytes.

With a policy, the worker must set its process-local PyTorch allocator ceiling
before constructing the native model. It measures actual Kev constructor,
tokenizer and forward calls, recording attempted and completed counts separately,
plus its own PyTorch allocated-memory peak. Resource errors remain terminal and
use the existing sanitized worker error protocol; they do not create a retry or
grant authority. Invalid request frames do not initialize CUDA or a model.

The policy also opts into a new exclusive, bounded JSONL audit file beside the
policy file, with a PID and unique ID in its filename. The audit contains resource
counters, hashes and sanitized status only, never submitted state or questions.
Evidence is written before successful replies because an owned SIGTERM may skip
Python cleanup. A last successful request record does not prove graceful shutdown.

Omitting the optional policy preserves the earlier unmetered worker behavior;
such a run has no resource-policy qualification. The allocator ceiling concerns
PyTorch allocations in that worker, not total board/WDDM memory. The earlier
external-sampler result below remains insufficient. Native evidence for this new
source revision must be recorded separately from pure/fake tests.

## Focused source test

```text
node --test D:/Projects/JevLocal/pi/bridge.test.mjs
```

The test uses child doubles only. It does **not** load a native model, install `typebox` or Pi, verify an actual Pi session, prove semantic quality/calibration, or establish latency/performance.

## Verified official SDK host slice (2026-09-26)

The isolated official `@earendil-works/pi-coding-agent` 0.87.1 SDK was installed under `runtime/pi-host-smoke-01` with lifecycle scripts disabled and separate npm cache, home and agent directories. Its cached host tarball matches the published SHA512 integrity. The recovered root lock omits some integrity fields; the publisher's shrinkwrap and an explicit dependency-verification receipt are retained. No global Pi or provider configuration was changed.

`runtime/pi-host-smoke-01/host-smoke.mjs` passed through the actual public SDK discovery/loader, `ExtensionRunner` and registered-tool wrapper, with the unmodified extension source. Two fresh in-memory host instances registered exactly `jev_ask` and `jev_choose_tool`; TypeBox resolved through the official loader. Missing local-model configuration, an inactive candidate and post-shutdown use were rejected. Instrumented child-process, network and host-action attempts were all zero. This is actual SDK integration evidence, not a hand-written `ExtensionAPI` double.

Evidence: `runtime/pi-host-smoke-01/host-smoke-command.json`, `host-smoke-result.json` and `dependency-verification.json`. This earlier test did **not** run an interactive Pi session, spawn the real Python worker, load a model, demonstrate natural host-process restart recovery, or qualify Noul/Choice/Score semantics.

## Verified native consumer slice (2026-09-26)

`runtime/pi-native-consumer-01` subsequently executed the unchanged shipped
extension through the official SDK, one real owned Python worker, and the strict
context-native predictor. A zero-update inference adaptation of retained CI
epoch-2 preserved all 372 adapter tensors, four head tensors and both original
training-marker files exactly. It is not new context training or a promoted model.

Two fresh public requests exercised Noul, Choice, Score and active-tool advice.
Reloaded worker probabilities matched the export-process reference exactly
(maximum absolute delta 0; frozen tolerance 1e-5). An inactive tool was rejected.
No host action or Node network call was attempted. Shutdown requested SIGTERM
and confirmed the owned worker's close; it was not a graceful EOF shutdown or
natural host-process restart test. The complete stage took 33.504 seconds,
including two process-local model constructions; this is not an inference-latency
benchmark. Internal worker load/forward counts are source-derived, not measured.

The export process directly measured 3,110,770,688 bytes (about 2.90 GiB) of peak
PyTorch CUDA allocation under a 10 GiB allocator cap. The separate CUDA-driver
sampler stayed below that number and did not track the real allocation. It is
**not accepted as evidence of the worker's memory ceiling**. Worker peak memory
and OS-level Python egress isolation remain unverified; offline flags and Node
network denial are narrower controls.

Evidence: `runtime/pi-native-consumer-01/root-verdict.json`, `outer-result.json`,
`export-result.json`, `host-result.json` and `adaptation-lineage.json`.
This terminal stage must not be rerun. Semantic qualification, interactive
recovery and applicable performance/resource evidence remain outstanding.
F19 and the full F01–F20 goal are incomplete.

## Verified worker resources and normal EOF (2026-09-26)

`runtime/pi-worker-resources-03` exercised the repaired production bridge through
the same official SDK in two strictly sequential cold host processes. Each worker
measured one model load, one tokenizer load and two real forwards. Both matched
the saved reference and each other with maximum probability delta 0. Each
process-local CUDA allocator peaked at 3,110,770,688 bytes under its 10 GiB cap.

Both idle shutdowns sent stdin EOF without a kill, produced terminal resource
audit records and closed with exit code 0 and no signal. Separate read-only
Win32 probes then confirmed the actual interpreter PIDs (not merely their
Windows launcher PIDs) were no longer running. The independent saved-evidence
postcheck passed. The previous W02 termination failure remains a terminal
failure; this new evidence does not retroactively reclassify it.

Evidence: `runtime/pi-worker-resources-03/root-native-verdict.json`,
`native-postcheck.json`, `native-outer-command.json` and `native-freeze.json`.
These prove the measured worker-resource and normal idle-close slice. They do
not prove pending-task recovery, interactive host recovery, OS-level egress
isolation, semantic qualification, default promotion or end-to-end speed.

## Pending-session drain repair (2026-09-26, CPU evidence only)

`runtime/pi-pending-drain-01` fixes the distinction between a closed Windows
launcher and a confirmed normal worker shutdown. Pending replacement/shutdown
now use EOF, discard the old response, and wait for normal close. Forced or
spontaneous closure cannot re-enable the same bridge. A pre-spawn cancellation
remains recoverable because no worker exists to leave behind.

The root reran 21 fake-child tests and Node syntax validation successfully.
These are not native pending-session evidence. W03 remains historical evidence
for its pinned idle-close source. The separately admitted W04 test below supplies
the later native pending-session evidence for this repair.

Evidence: `runtime/pi-pending-drain-01/root-verdict.json`.

## Verified native pending lifecycle (2026-09-26)

`runtime/pi-worker-resources-04` used the pinned official SDK and repaired
production bridge in two sequential hosts. A real `jev_choose_tool` invocation
was still pending after its complete stdin write when the SDK emitted
`session_start/new` or `session_shutdown/quit`. The caller received
`session_changed` or `closed`; the old worker finished its request through EOF,
but its late answer was discarded rather than returned as advice.

Host A then started a replacement worker only after a separate read-only Win32
probe confirmed the old actual Python interpreter had stopped. The probe ended
31 ms before the replacement spawn. All three workers closed normally with
terminal audits and independent actual-PID stop evidence. Measured totals were
three model loads, three tokenizer loads and five forwards (2 + 1 + 2), with
zero updates or exports. Normal delivered answers had maximum reference and
cold-host probability delta 0. Each worker peaked at 3,110,770,688 bytes of own
PyTorch CUDA allocation under a 10 GiB cap. The complete isolated stage took
74.224 seconds, including cold loads; this is not an end-to-end speed claim.

The independent saved-evidence postcheck passed. Neither a suggested tool nor
a host action was executed. Probabilities remained uncalibrated and semantics
unqualified. This proves the scoped pending advisory lifecycle, not arbitrary
task replay, interactive host recovery, forced process-tree termination,
OS-level Python egress isolation, semantic qualification or default promotion.

Evidence: `runtime/pi-worker-resources-04/root-native-verdict.json`,
`native-postcheck.json`, `native-outer-command.json` and `native-freeze.json`.
The one-shot stage is terminal and must not be rerun. F19 and the full F01–F20
goal remain incomplete.
