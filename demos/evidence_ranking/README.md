# Public CPU evidence-ranking demo

This is a small, real-model reproduction of JevLocal's advisory context-ranking
path, not a qualified general decision system. It preserves the lexical primary
and attaches Qwen relevance suggestions; it never executes a suggested action.

## Recorded result (2026-10-21)

The [complete local report](results/local-cpu-20261021.json) contains every
prediction, both yes/no logits, all rankings, labels, timings and identities.
It is the unedited JSON emitted by one successful real CPU run, not a mock or
a selection of successful cases. SHA-256:
`3154e2031eeceeff38a9f77bca1d425c8b96e1d8c63cf513f91119b2954e45a4`.

| Synthetic category | Cases | Qwen top-1 | Lexical top-1 |
|---|---:|---:|---:|
| Direct relevance | 12 | 12/12 | 10/12 |
| Paraphrase | 8 | 8/8 | 4/8 |
| Lexical distractors | 4 | **0/4** | **0/4** |
| Total | 24 | **20/24 (83.33%)** | **14/24 (58.33%)** |

MRR (mean reciprocal rank of the expected candidate) is **0.888889** for Qwen
and **0.756944** for lexical. All four hard cases (`lexical-01` through
`lexical-04`) put the expected candidate **third**, for both methods. The model
fails when a superficially matching passage competes with a useful but less
lexically similar passage. These failures remain in the published inputs and
report; no prompt, model, labels or samples were tuned after seeing this run.

Actual host: Windows 11 AMD64, Intel Core i5-12600KF, 64 GiB installed RAM,
CPython 3.12.14, Torch 2.8.0+cpu, four Torch threads, FP32, SDPA, seed 42.
The six direct package versions are in [requirements-cpu.txt](requirements-cpu.txt).
One model load served 24 forwards / 72 query-candidate judgments; no optimizer
updates or exports occurred. Start/end: `2026-10-21T10:38:29.622405Z` /
`2026-10-21T10:40:32.244759Z`; process exit code: 0.

| Measured interval | Seconds |
|---|---:|
| Entire reported evaluation | 122.617 |
| Model load | 18.721 |
| Native inference, total | 102.817 |
| Native forwards within inference | 102.718 |
| Validation plus lexical preparation, total | 0.004042 |

The last row times the entire `qc.prepare` call, **not** an isolated lexical
algorithm benchmark. Native inference was much slower in this bounded run;
there is no speedup claim. These are one-run observations, not latency
percentiles, peak-memory measurements or cross-hardware performance guarantees.

## Inputs and evaluation boundary

- [cases.json](cases.json): 24 newly authored, AI-assisted synthetic English
  cases, three candidates each; no private corpus, logs or historic training
  samples were used. Each answer is determined from the supplied text.
- [labels.json](labels.json): separate expected choices, categories and
  rationales. Labels were reviewed by a second AI agent **before** inference;
  this is not human annotation or an independent public-user evaluation.
- Expected positions are balanced (eight each for a/b/c), using seed
  `20261021`, Mulberry32 and descending Fisher-Yates. This reduces a trivial
  position cue; it does not make the sample representative or statistically
  independent. No held-out real-world benchmark claim is made.
- The inference entrypoint does not read the label file until **all** model
  predictions are complete. The optional validation-only command reads labels
  but does not load or call a model. Labels and category are not prompt inputs.
- Inputs and scoring source were hashed before the local run and checked
  unchanged afterward. Input SHA-256 values are:
  - cases: `e72eb5e119f6b9c3e32b26b58b5c659eab93fe2e4162f1110eb04927ad4ea730`
  - labels: `281b38df85e54974c3c15316e404a167078ef3e432cf5b2675d118f2fa13a5d4`
- Scores are the softmax of the model's yes/no next-token logits, **not calibrated
  probabilities of truth**. Top-1/MRR evaluate ranking, not factual verification.
  The report retains `calibrated`, `execution_authorized`, `training_allowed`
  and `primary_changed` as false.

The code and new synthetic examples use the project's [Apache-2.0 license](../../LICENSE).
The model remains subject to its own
[pinned model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/e61197ed45024b0ed8a2d74b80b4d909f1255473/README.md)
(Apache-2.0 declaration). Dependencies keep their own licenses. This repository
does not redistribute model weights or historical private training data.

## Reproduce from a fresh checkout

Use a local disk path without symlink/reparse-point ancestors. You need Python
3.12, network access for the explicit installation/download steps, roughly
1.2 GB for the model files plus environment/cache space, and several GB of free
RAM for FP32 inference. Peak memory was not measured; 64 GiB is the observed
host, not a minimum requirement. No GPU, Hugging Face account/token or private
dataset is required. Review these downloads before running them.

```text
git clone https://github.com/XLGXB/jev-local.git
cd jev-local
```

Create an isolated environment (do not use system Python for pip installs):

```bash
# Linux shell.
python3.12 -m venv .venvs/public-demo
source .venvs/public-demo/bin/activate
```

```powershell
# Windows PowerShell session; Python 3.12 must already be installed.
py -3.12 -m venv .venvs/public-demo
.\.venvs\public-demo\Scripts\Activate.ps1
```

If activation is unavailable, replace `python` below with the full path to that
environment's interpreter; do not change machine execution policies. The local
record is Windows CPU; the manual workflow below targets Ubuntu x64 CPU.
The pinned Torch CPU index commands below target these Windows/Linux platforms.

Run from the repository root with the isolated interpreter active:

```text
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demos/evidence_ranking/requirements-cpu.txt
python -B -X utf8 demos/evidence_ranking/run.py --validate-only
```

These pin the six direct runtime packages, **not every transitive dependency**.
Record `python -m pip freeze` locally when diagnosing a different environment;
review it for private URLs before sharing. No editable project install is needed.

Download only the 14 manifest files from the immutable revision. This explicit
network step uses `token=False` and does not need login. It writes the weights
under the ignored `models/` directory, not into the source payload:

```text
python -c "import json; from huggingface_hub import snapshot_download; m=json.load(open('demos/evidence_ranking/model-manifest.json',encoding='utf-8')); snapshot_download(repo_id=m['model_id'],revision=m['revision'],allow_patterns=[f['path'] for f in m['files']],local_dir='models/public-demo-qwen',token=False)"
```

`run.py` requires an **absolute** model directory and a **new** output file in
an existing directory. Print the model path, then substitute that printed path
for `ABSOLUTE_MODEL_DIRECTORY` below (keep quotes around it):

```text
python -c "from pathlib import Path; print(Path('models/public-demo-qwen').resolve()); Path('runtime/public-demo').mkdir(parents=True,exist_ok=True)"
python -B -X utf8 demos/evidence_ranking/run.py --model-dir "ABSOLUTE_MODEL_DIRECTORY" --output runtime/public-demo/result-01.json
```

The evaluator checks every model file's size and SHA-256 against
[model-manifest.json](model-manifest.json), then uses `local_files_only=True`,
`trust_remote_code=False`, safetensors, offline environment flags and a Python
socket guard. This is application-level offline behavior, not a claimed OS
network sandbox. It does not download, install, train, retry failed inference,
overwrite an existing report, or execute ranked content. Exit 0 means evaluation
completed, **not** that every answer was right. Use a different output filename
for another run. Default errors show stage/type; `--debug` exposes local traceback
details and is for local diagnosis, not unreviewed public log upload.

Compare `metrics`, all 24 `cases[*].evaluation` entries, asset hashes and runtime
versions. Invocation IDs and timings change; floating-point scores may vary
across CPUs/libraries. Exact score equality is not promised.

## Public maintenance and reproduction evidence

[Source checks](../../.github/workflows/ci.yml) exercise dependency-light
contracts on Windows and Ubuntu with injected judges; they are not model runs.
The separate [manual CPU workflow](../../.github/workflows/model-reproduction.yml)
creates a fresh Ubuntu environment, downloads the pinned public model and emits
one public report artifact (seven-day retention). It has no scheduled/push model
run and no accuracy threshold that would hide the four observed failures.

### Verified fresh hosted run

On 2026-10-21, the owner manually triggered the authorized CPU reproduction
**once**. [Run 39593252859, attempt 1](https://github.com/XLGXB/jev-local/actions/runs/39593252859)
completed successfully against commit
`18d153e6bf4ec8a493a4b86becae25b29f95ea97` in a fresh Ubuntu 24.04 runner,
CPython 3.12.15, Torch 2.8.0+cpu, FP32/SDPA/four threads. The complete
[cloud report](results/github-cpu-39593252859.json) is preserved unchanged so it
remains available after the seven-day Actions artifact expires. Report SHA-256:
`b52892fe51f05d2ea04eabaffbf99e4f973bcc12bc9bca51a5bac72046b1c434`.
The [reproduction receipt](results/reproduction-20261021.json) binds the run,
commit, artifact archive digest, report digest, source CI and comparison.

The cloud run again measured **20/24** native versus **14/24** lexical, with the
same **0/4** hard-case failures. Dataset/scoring-source/model-asset identities,
all aggregate metrics and all 24 full native rankings match the local result.
The maximum absolute native score difference was approximately `1.91e-6`;
bitwise floating-point equality is not claimed. Cloud evaluation took 83.294 s
(load 4.335 s; inference 78.097 s; forwards 77.967 s; validation plus lexical
preparation 0.005738 s). This is another single-run observation, not a controlled
hardware speed comparison or a promise about other machines.

[Source CI at the same commit](https://github.com/XLGXB/jev-local/actions/runs/39592849752)
passed both Ubuntu 24.04 and Windows 2022 jobs. The first workflow definition
had an [Actions context-validation failure](https://github.com/XLGXB/jev-local/actions/runs/39591967708):
`runner.temp` was unavailable in job-level `env`. It created **zero jobs** and
performed no model evaluation. Commit `18d153e` moved path initialization into
a runner step before the single successful manual model run. That failure is
retained in public history rather than presented as an executed model attempt.

For an independent attempt, use the repository's reproducibility issue form
and follow [CONTRIBUTING.md](../../CONTRIBUTING.md). Report successes **and**
failures with the commit, command, sanitized environment, hashes and result.
Maintainer CI/reproduction is public operational evidence, not third-party
adoption, a security audit, or completion of the wider F01-F20 project.
