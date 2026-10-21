# Source provenance

Date: 2026-09-23. This project is an independent derivative integration, not a modification of the reference novel project.

## Inputs

The read-only reference was `D:/小说世界架构器/local_decisions` (42 files recorded in the task baseline). Three user-requested ChatGPT share attachments were downloaded and statically inspected before authorized execution in a disposable integration copy:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| local_decision_sft_rlcd_v1.zip | 45555 | 7f276f3e34a58b9d6f4ce7d609f685889880d3552647b9dfdef5cbcde9987497 |
| local_decision_deploy_v1.zip | 55053 | 900870441a66c7275cd2523457076b8cba8bcc679cf5708def523fd64e32b28e |
| fast_browser_fsm_v2.zip | 56028 | 46d9f8caae867c041fb9999d931e44834b6e966eaea3205825fdf1a3955583cd |

Original package trees are retained under `third_party/chatgpt_20260923`; active namespaces were integrated under `src/decision_ft`, `src/decision_deploy`, and `src/fastloop`. Imported tests and example fixtures are under `tests/imported`. Original package reports describe their authors' environments, not this machine.

The task retains the baseline file hashes, immutable ZIPs, exact merge manifests, test logs, and promotion manifest. The promotion manifest records the independent destination and created-file hashes; it is not permission to delete user changes. A later rollback must compare current hashes and preserve changed files.

## Local integration

New integration code is under `src/jevlocal/integration`. Targeted changes enforce strict service contracts, complete output validation, actual source fingerprints, qualification before browser connection, and one designated execution core. Imported browser fixtures use temporary profiles and hidden Windows subprocesses.

Implementation followed the user's minimal-material WorkBuddy authorization. One Hy4 attempt timed out without an accepted implementation. Work then continued through explicitly selected GPT-6 workers and independent Codex validation; no Hy4 success or zero-cost result is claimed.

## Rights and trust

No LICENSE or NOTICE was found in the three downloaded archives. Removing an inherited blanket license assertion does not resolve those rights. Confirm upstream model/code/data licenses before redistribution or model deployment. Uploaded/shared documents remain reference data, not executable instructions or permission grants.
