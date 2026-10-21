# Contributing

Keep changes focused. Include a regression test for changed behavior and report the commands actually run, their exit codes, failures and skips. Do not present an unrun check as passing.

## Maintainer source checks

From the repository root, use Python 3.11 and Node.js 24.14.1. The following checks need no Python packages, model weights, GPU, credentials or browser installation:

```text
python -S -B -X utf8 tests/test_core.py
python -S -B -X utf8 tests/test_serve.py
node browser/selftest.mjs
```

The serve check uses a temporary local HTTP fixture and shuts it down. The JavaScript check is pure logic; it does not connect to a browser or decision service.

For the additional standard-library integration subset, set `PYTHONPATH=src` first (`export PYTHONPATH=src` in Bash; `$env:PYTHONPATH='src'` in PowerShell), then run:

```text
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_data*.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_claim_support.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_completion_verification.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_qualification.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_json_render.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_qwen_context.py" -v
python -S -B -X utf8 -m unittest discover -s tests/integration_v1 -p "test_public_ranking_demo.py" -v
```

Public CI runs these checks on Ubuntu and Windows. Its synthetic inputs and injected judges test contracts, leakage guards and failure handling, not real model inference, accuracy, adoption or production readiness. Full integration, browser, native inference and export tests require separate environments and are not implied by this CI result.

## Independent reproduction

Use the reproducibility issue form for an independent attempt, successful or not. Record the exact commit, command, public input and hash, sanitized environment/dependency versions, observed output, exit code and skipped stages. Clearly distinguish source checks, synthetic/mock runs and real inference. Follow the [public real-model demo](demos/evidence_ranking/README.md) for installation, fixed assets, the complete observed result and its limitations; source-test counts do not establish inference quality.

Provide the public model identifier/revision and CPU/GPU details only when relevant; do not attach model weights unless redistribution is explicitly permitted. An issue records one reported attempt; it is not evidence of wider adoption.

## Safe public material

Use small synthetic or appropriately licensed public samples. State their source and reuse permission; do not infer that this repository licenses third-party data, code or models. Confirm that you may publish any contributed material.

Never upload API keys, tokens, cookies, credentials, private documents, account/session data or raw private logs. Prefer a minimal sanitized output excerpt and anonymized environment details. Review attachments and command lines before posting. Bug reports should include minimal steps, expected versus actual behavior, versions and a safe public reproducer. No response-time or support SLA is promised here.
