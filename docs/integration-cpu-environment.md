# Independent CPU integration environment

## Verified state

- Interpreter: `D:/Projects/JevLocal/.venvs/integration-cpu/Scripts/python.exe` (CPython 3.12.14).
- Isolated from system/user site-packages. 93 hash-pinned wheels; Torch 2.8.0+cpu, Transformers 5.17.0, Hugging Face Hub 1.32.0, Tokenizers 0.23.2, Kev 0.1.0, MCP 1.30.0.
- Offline installation and `pip check` succeeded. All locked installed versions matched. A small CPU tensor and native/service imports passed; no model checkpoint was loaded by environment acceptance.
- The original `.venv`, `.venvs/kev`, `.venvs/qwen-cuda`, default configuration and promotion file were not installation targets. Fifteen relevant metadata/configuration sentinels remained unchanged. This is not a full filesystem snapshot.
- The documented original project interpreter still inherits incompatible global Hugging Face dependencies. Use the explicit new interpreter for these regressions; no global/default interpreter switch was made.

## Regression evidence

- `integration_v1`: 231 tests passed, no skips.
- Full pytest: 381 passed, 29 skipped. The skipped cases are 28 Chromium fixtures and one optional ONNX test; they are not accepted as passing browser/export evidence.
- The first full run exposed 21 UTF-8 fixture reads through the Windows GBK default. The fix only specifies `encoding="utf-8"` at five reads in three imported fastloop test files. Assertions, fixture bytes and original archives are unchanged. Focused regression: 35 passed. Neither final run depended on `-X utf8`.

Run from the project directory with the explicit interpreter, for example:

```powershell
pwsh.exe -NoLogo -NoProfile -Command 'Set-Location -LiteralPath "D:/Projects/JevLocal"; & "D:/Projects/JevLocal/.venvs/integration-cpu/Scripts/python.exe" -I -B -m pytest --import-mode=importlib tests -ra'
```

Do not rerun the one-shot installation/acquisition/acceptance scripts over their exclusive output paths. Do not reuse the original native F12 batch. New test runs should retain their own logs and, if needed, fresh temporary directories.

## Scope and limitations

This environment restores the CPU regression/service dependency baseline. It is not the missing CUDA environment, model training, semantic qualification, actual desktop integration, browser fixture execution or F01-F20 completion. Desktop control activation awaits explicit approval; no desktop service or test window was started by this environment work.

Authoritative evidence: `D:/Projects/JevLocal/runtime/integration-cpu-acceptance-01/root-verdict.json`. Install inputs: `D:/Projects/JevLocal/runtime/integration-cpu-env-01/manifest.json` and `requirements.lock`.
