# Publishing boundary

Date: **2026-10-21**. The initial source-only upload to [XLGXB/jev-local](https://github.com/XLGXB/jev-local) completed at [b31492c](https://github.com/XLGXB/jev-local/commit/b31492c131315f73daad3f8f06a6bb430c1d93e2). The rights holder subsequently confirmed the right to license project code, including the three generated ZIP inputs, under **Apache-2.0**. See [LICENSE](../LICENSE), [NOTICE](../NOTICE), and [LICENSING.md](LICENSING.md). This records the owner's decision, not independent legal clearance. It does not authorize publication of private data, weights, logs or separately obtained third-party material.

## Source boundary

The reviewed source-only payload includes the four `src/` packages, focused tests, scripts/tools, opt-in `pi/` extension, docs and configuration examples. The current authorized additions include the new public synthetic demo under `demos/evidence_ranking/`, its bounded result JSON, de-identified historical summaries under `docs/experiments/`, contribution guidance, issue forms and dependency-light CI. Future changes still require inspection for rights, secrets and private content; directory names are not approval.

Keep these local and outside repository payload:

- Actual `data/` and all `datasets/`: corpora, labels, partitions and sealed evaluation material.
- `models/`: weights, checkpoints, adapters, exports and caches.
- `vendor/`, `third_party/`: retained upstream sources/model cards and imported snapshots pending rights review.
- `runtime/`: experiment runners/inputs, artifacts, receipts, logs, screenshots and environment captures.
- `.venv/`, `.venvs/` and other environments/caches/build outputs.
- Credentials, API keys/tokens, cookies, signing/private keys, account state, local environment files and private configs, wherever located. `configs/passive-runtime.original-laya.json` is a machine-local checkpoint config, not a portable example.

### Narrow data-fixture exception

Only `data/examples/modernbert.train.jsonl`, `data/examples/qwen_kev.train.jsonl` (three records each), and `data/schema/decision_sample.schema.json` are included source fixtures. `tests/test_core.py::test_example_datasets_validate` reads the two JSONL files. They are tiny format-test examples, **not actual training data**; the ModernBERT examples explicitly mark their provenance notes as synthetic examples rather than real trajectories. These files were included in the source review and the owner's publication authorization. This exception does not make other data public.

Preserve excluded originals, archives and evidence in place. Exclusion is not deletion or a guarantee that source can reproduce native runs without separately authorized assets. `.gitignore` is only a guardrail: it does not prove absence of secrets or remove already tracked content. Review the actual prospective file list and content before staging.

Docs may name local evidence paths, but published source will not contain those receipts. Do not bundle runtime trees to satisfy pointers. The currently authorized historical publication is limited to reviewed configuration, metrics and summary hashes; exclude original private examples/labels, weights and raw logs. The 24-question public synthetic demo is newly authored public material, not a repackaging of the private training corpus.

## Rights and licensing limits

[PROVENANCE.md](PROVENANCE.md) records three original archives and their hashes. No `LICENSE` or `NOTICE` was found in those archives. Do not assume MIT, public domain, or an inherited blanket license. Model licenses/cards under `vendor/` do not cover all integration code, data, dependencies or derived weights.

The rights holder's explicit Apache-2.0 project-code decision now covers downstream use, modification, distribution and sublicensing under that license, including the integrated generated code. It does not change the historical absence of license files in the archives or relicense external libraries, models, data or trademarks. [LICENSING.md](LICENSING.md) records the reviewed evidence and limitations; no exhaustive legal clearance or non-infringement warranty is asserted.

## Publication checks

1. Verify the exact target and its visibility; inspect its current branches before writing. The initial check found the public target empty. Preserve subsequent remote changes; do not force-push over them.
2. Review the exact source payload, including tests/fixtures/configs, for private material, secrets, account state, local paths and rights restrictions. Exclude weights, actual datasets, logs and environments.
3. Bind relevant verification to the source snapshot. The organization pass recorded 26 core, 23 service and 65 JavaScript checks passing, plus 32 ignore-boundary cases. These are scoped checks, not a full-suite, model or deployment qualification.
4. Inspect the staged path list and content before committing. After pushing, compare the remote commit and tree with the reviewed local snapshot. A local commit alone is not evidence of upload.

The local candidate manifest, scoped test report and publication receipt remain under ignored `runtime/repository-organization-20261021/`; they are not part of the public payload. Cleanup Hub's bounded scan returned `skipped: preflight_entry_limit`, not a full-tree clean result. Git upload does not install dependencies, train, download models, start daemons, register MCP, delete archives or establish complete F01–F20 acceptance.

## Readiness reporting

Keep the new CPU demo separate from historical SFT results and from hosted CI. The local 20/24 versus lexical 14/24 result retains all four lexical-distractor failures; it is not calibrated, representative, trained on these cases, or proof of time savings. The separately authorized manual cloud-CPU reproduction remains unexecuted. CI configuration, repository publication, experiment commits and issue forms do not establish independent community use. Separate commits for historical experiments must retain their real experiment dates rather than imply continuous recent maintenance.
