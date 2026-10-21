# JevLocal

Independent local typed-decision integration for Codex and explicitly configured consumers: data preparation, native inference, evidence-bound advice, and bounded browser execution. Separate from the original novel-world project.

**Status, 2026-10-21:** source snapshot for [XLGXB/jev-local](https://github.com/XLGXB/jev-local). The full **F01–F20 goal remains incomplete**. Implemented interfaces and isolated native/consumer tests do not establish semantic qualification, production readiness, or end-to-end speed gains. See [status](docs/PROJECT_STATUS.md) and [publishing](docs/PUBLISHING.md).

## Project map

| Location | Purpose |
|---|---|
| [src/jevlocal](src/jevlocal/) | Canonical data/schema, extraction/export utilities, integration CLI, context and advisory consumers |
| [src/decision_ft](src/decision_ft/) | Native backends, training/evaluation, calibration and represented-checkpoint adapters |
| [src/decision_deploy](src/decision_deploy/) | Typed inference protocol, runtime identity, qualification and deployment interfaces |
| [src/fastloop](src/fastloop/) | Active bounded browser core: `fastloop.service.invoke`, target lock, durable state and uncertain-effect handling |
| [pi](pi/README.md) | Opt-in advisory extension using the official Pi extension API; not an action executor |
| [tests](tests/) | Integration, imported-package and focused regression tests; fixture scope matters |
| [configs](configs/) | Examples and explicitly selected local configuration; templates are not approval or deployment receipts |
| [browser](browser/) | Historical JavaScript design/regression reference, **not a second active controller** |
| [docs](docs/README.md) | Interface reference, evidence limits, provenance and historical design index |

Actual data/datasets, models, vendor/imported snapshots, runtime evidence and virtual environments are local-only. A narrowly reviewed exception retains two tiny format-test JSONL examples and the sample schema; see [publishing](docs/PUBLISHING.md). Preserve original archives and evidence in place. Source alone does not include a runnable model/environment.

## Environment and entrypoints

Python **3.11+** is required. Use an explicitly authorized isolated environment; never install dependencies into system Python as a command side effect. The core package has no mandatory dependencies. [pyproject.toml](pyproject.toml) defines optional extras:

- `runtime`, `browser`: service/browser integration dependencies.
- `native`, `train`: native inference/training dependencies; neither grants download or training permission.
- `mcp`: optional context MCP transport.
- `export`, `export-trt`, `serve`, `dev`: export/service/development tooling; hardware/provider/environment acceptance is separate.

After installation is separately authorized, select only needed extras, e.g. `python -m pip install -e ".[dev]"` from this directory. This is an example, not an action performed by this organization pass. Generic extras do not replace pinned route-specific environments.

For an existing source environment, put the absolute project `src` directory on process-local `PYTHONPATH`:

```text
python -m jevlocal.integration.cli doctor
python -m jevlocal.integration.cli validate-envelope --input request.json
python -m jevlocal.integration.cli bridge --input samples.jsonl --authorization authorization.json --output records.jsonl
python -m jevlocal.integration.cli prepare-dataset --input records.jsonl --output-dir frozen-dataset
python -m jevlocal.integration.cli --help
```

On Windows, the source launcher accepts an explicit interpreter:

```text
pwsh.exe -NoLogo -NoProfile -File scripts/jevlocal.ps1 -Python <isolated-python.exe> doctor
```

It never installs packages or falls back to system Python. Omit `-Python` only after an authorized project-local `.venv` exists.

Installed console scripts are `jevlocal`, `jevlocal-extract`, `jevlocal-decisions`, `jevlocal-browser`, `jevlocal-context-mcp`, and `jevlocal-semdecide`. Inference needs explicitly selected local assets. Context MCP requires optional dependencies, an absolute `JEVLOCAL_CONTEXT_STORE`, and separate host registration; the example TOML is not automatically installed. See [integration](docs/INTEGRATION.md), [SemDecide](docs/semdecide.md), and [Pi](pi/README.md) for distinct contracts.

## Verification

Dependency-light format, synthetic and loopback checks (no model/GPU):

```text
python -B -X utf8 tests/test_core.py
python -B -X utf8 tests/test_serve.py
node browser/selftest.mjs
```

Broader integration entrypoints, with their required optional dependencies:

```text
python -m unittest discover -s tests/integration_v1 -p "test_*.py" -v
python -m pytest --import-mode=importlib tests
node browser/sm_selftest.mjs
```

These are entrypoints, not results from the 2026-10-21 documentation pass. Browser fixtures require an explicitly supplied isolated `TEST_CHROMIUM` and disposable profiles. Missing dependencies/skipped fixtures do not prove browser acceptance. Historical JavaScript tests use fake IO. Evidence must bind interpreter, source, command and exit status; see [status](docs/PROJECT_STATUS.md).

## Safety and rights

- Default to dry-run. Advice, confidence, qualification, signed grants and actual user approval are distinct. Advice cannot authorize execution, merge, completion, deletion or training.
- Preserve public-question/label separation, semantic candidate keys, provenance, explicit training permission and frozen train/dev/calibration/test partitions.
- Use only `fastloop.service.invoke` as the active browser core. An unavailable supported host adapter must not trigger a second CDP controller or fallback provider.
- Reject stale/malformed evidence; keep status/cancel and uncertain-effect quarantine. Credentials/signing keys stay outside model-visible context. Secret screening is best-effort, not permission to send private material externally.
- No service, global MCP registration, model download or account operation is installed automatically.

The project owner confirmed authorization to publish this source snapshot on 2026-10-21. The three original imported archives contain no located `LICENSE` or `NOTICE`; that confirmation is not an independent license audit or a blanket license grant to downstream users. No project license has been selected. See [provenance](docs/PROVENANCE.md), [publishing boundary](docs/PUBLISHING.md), and [AGENTS.md](AGENTS.md).
