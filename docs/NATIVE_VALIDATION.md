# Native validation boundary

## Status (2026-09-23)

| Route | Evidence | Limits |
|---|---|---|
| Laya multilingual | Real CUDA FP32 inference for Choice/Noul/Score; finite, normalized raw probabilities; semantic candidate keys preserved | One tiny synthetic literal fixture, not representative quality |
| Laya gradient/reload | Two atomic questions; nonzero head/backbone gradients; max probability difference 0, tolerance 1e-5 | Zero optimizer updates; no training improvement claim |
| Laya ASGI integration | Actual native predictor through authenticated in-process API: valid 200, unauthenticated 401, browser Origin 403, unexpected envelope field 400; reply bindings validated | ASGI test transport, no HTTP listener, real browser or account |
| Kev-0.8B | Real CPU FP32 inference for all three types; nonzero head/LoRA gradients; save/reload max probability difference 0, tolerance 1e-5 | Same tiny synthetic scope and zero optimizer updates; reference PyTorch kernels, not a speed result |

Neither route has domain calibration/qualification or permission to operate a browser. The action gate remains unchanged. The existing unapproved examples are not deployment receipts.

## Pins and local evidence

- Laya source: `c7527708f9f5220c669d8aa385077cd28d04708a`; model: `convaiinnovations/laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, multilingual subfolder.
- Kev source: `557598fced1dada75dfbf36ed144dce309ac6ceb`; adapter/head: `jaredpalmer/kev-0.8b` revision `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`.
- Qwen base: `Qwen/Qwen3.5-0.8B-Base` revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Do not replace the full revision with an arbitrary local path or unrelated model.
- Sources and exact file hashes: `runtime/native-source-audit/source-manifest.json`.
- All 15 model files (2,493,540,969 bytes): `runtime/native-downloads/range-20260923T025652856Z.json`.
- Laya install: `runtime/laya-environment/installation.json`. Verified source built in a separate staging directory and installed offline without changing dependencies.
- Laya runs: `runtime/native-smoke/laya/{inference,doctor,service}.json`; corresponding `*-preflight.json`, `*-process.json` and logs bind environment, source, commands and exit status.
- Kev runs: `runtime/native-smoke/kev/{inference,doctor}.json` plus corresponding preflight/process receipts and logs; installed-source verification in `runtime/kev-package-install/installation.json`.
- Experiment contract: `runtime/native-smoke/experiment-card.json`. All labels are fabricated, explicit and program-verified; no private training data.
- Model cards/licenses are retained under `vendor/model-cards`. Do not infer that these licenses cover all integration code.

The first Laya weight transfer ended unexpectedly. Recovery reused the existing prefix and required exact HTTP range responses followed by full-file hash verification; it did not silently restart the entire download.

## Environment and safety

Laya uses the project `.venv` (Python 3.11.9, Torch 2.11.0+cu128, Transformers 5.5.1). It inherits base packages read-only; the reported unrelated `chromadb`/`tokenizers` conflict remains visible. Kev uses `.venvs/kev` (Python 3.12.14, Torch 2.8.0+cpu, Transformers 5.17.0, PEFT 0.21.0). Its 68 hash-pinned official wheels total 377,649,636 retained bytes; offline install and `pip check` passed, followed by offline installation and source-hash verification of audited Kev 0.1.0. PyPI's unrelated `kev` package was not installed.

Kev logs explicitly report reference PyTorch implementations because `causal_conv1d` and `flash-linear-attention` are not installed. PEFT also reports that its remote base-config existence probe is unavailable offline and assumes unchanged vocabulary; pinned base metadata, disabled special embeddings and strict adapter-key coverage remain intact. These warnings were not suppressed. No accelerated-kernel claim or cross-route latency comparison is made.

Native tests set offline/telemetry-disable flags before imports and `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`. The actual Kev head was loaded with explicit `weights_only=True` after its pinned SHA-256 was verified. No unsafe deserialization fallback is permitted.

Inference and gradient/reload processes deny all socket connections. The ASGI-only test additionally permits Python's own same-thread loopback socketpair used by Windows asyncio; all other connections remain denied. Its first attempt failed because the test guard blocked this runtime self-pipe; the original failed logs are preserved separately. This was a harness issue, not a product inference error, and no external-network exception was added.

The Laya adapter matches upstream's optional eager-mode setting: if `encoder.config.reference_compile` exists, set it false. This pinned multilingual encoder does not expose that field (recorded as null); fixture tests cover both cases. Strict weight loading and the frozen auxiliary action head are unchanged.

## Remaining acceptance

Evaluate representative, authorized frozen data, fit/verify calibration without test leakage, and bind actual artifacts into qualification. Training requires its own authorized data and optimizer-run scope. Provider-specific ONNX/TensorRT parity/performance and a supported Codex host adapter are separate gates. Do not infer any of these from the completed smoke results.
