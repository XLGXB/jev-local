# JevLocal project contract

This is an independent project. The original novel-project directory is read-only reference material, not this project's workspace or runtime.

- Keep the current user-selected Codex model as the decision owner. Local typed decisions are suggestions, never authorization.
- Use one active browser core: `fastloop.service.invoke`, including its per-target cross-process lock. The old JS and deployment-v1 execution paths are regression references, not alternate active controllers.
- Preserve public question/label separation, semantic candidate keys, explicit training permission, source provenance and frozen dataset partitions.
- Never equate statistical confidence, qualification, a grant-shaped JSON object, or model-reported success with user approval. Uncertain effects remain quarantined.
- Default to dry-run. Real host integration must use supported host interfaces; an unavailable host adapter must not silently fall back to a second CDP controller.
- Core verification: `python -m unittest discover -s tests/integration_v1 -p "test_*.py" -v`. Full subproject verification: `python -m pytest --import-mode=importlib tests`. Browser fixtures require an explicitly supplied isolated `TEST_CHROMIUM` and temporary profiles. These do not prove native model quality or production readiness.
- Do not install dependencies, edit global MCP/provider/proxy settings, download models, or operate real accounts without the corresponding user authorization. Record commands, versions, exit codes, source hashes and unverified stages.
- Preserve original archives and provenance. Do not claim an inherited license covers all imported/generated code.
