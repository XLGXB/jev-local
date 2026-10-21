# Licensing and source rights

Decision date: **2026-10-21**. Project code is licensed under the
[Apache License, Version 2.0](../LICENSE). See the project [NOTICE](../NOTICE)
and [source provenance](PROVENANCE.md).

## Rights-holder decision

The rights holder explicitly confirmed the right to license the project code,
including the three requested ChatGPT-generated attachments, under Apache-2.0.
The confirmation covers downstream use, modification, distribution and
sublicensing under that license, not merely permission to publish a GitHub copy.
This decision includes the integrated code in `src/decision_ft`,
`src/decision_deploy` and `src/fastloop`, as well as the project's own code and
integration contributions. Separately identified third-party materials retain
their own terms; this decision does not purport to grant rights the licensor
does not control.

## Source review and retained evidence

The review distinguishes existing declarations from the new licensing decision:

| Source | Observed licensing evidence | Relationship to current source |
|---|---|---|
| Original JevLocal reference, version 0.1.0 | Its `pyproject.toml` declared `license = { text = "Apache-2.0" }`; no standalone license file was found at the top level or in the reviewed source, documentation and browser paths. | All 20 original Python source files remain represented: 18 byte-identical, with `schema.py` and `__init__.py` modified. This declaration alone does not license later imports. |
| `local_decision_sft_rlcd_v1.zip` | No LICENSE or NOTICE; no license or authors field in its package metadata. The rights holder has now confirmed the right to license this generated material. | 22 archived files; 10 original package source files correspond to current `decision_ft`: 9 byte-identical and `backends.py` modified. |
| `local_decision_deploy_v1.zip` | No LICENSE or NOTICE; no license or authors field in its package metadata. The rights holder has now confirmed the right to license this generated material. | 34 archived files; 11 original package source files correspond to current `decision_deploy`: 4 byte-identical and 7 modified. |
| `fast_browser_fsm_v2.zip` | No LICENSE or NOTICE; no license or authors field in its package metadata. The rights holder has now confirmed the right to license this generated material. | 31 archived files; 9 original package source files correspond to current `fastloop`: 7 byte-identical, with `model_client.py` and `service.py` modified. |

On 2026-10-21 the three retained original ZIPs matched the byte sizes and
SHA-256 values in [PROVENANCE.md](PROVENANCE.md). Comparing all **87 archived
files** against their retained trees under `third_party/chatgpt_20260923` found
no missing files and no byte differences. The source comparisons above describe
that review snapshot, not a promise that later revisions remain byte-identical.
Original archives, download records and local evidence remain outside the public
source payload; they were not deleted or relicensed in place.

The imported training package describes itself as independently written adaptation
and training code and cites Laya/Kev interfaces and method references. That is a
source statement, not an independent guarantee against copied material. The
review found no additional author, copyright or SPDX license declarations in the
reviewed imported source, README and package metadata files. Missing declarations
were not treated as a public-domain dedication or an inherited blanket license.

## Third-party boundary

External libraries, separately obtained upstream source, models, weights,
tokenizers, adapters, training/evaluation data and services retain their own
licenses and usage conditions. Project Apache-2.0 metadata does not relicense
them or authorize their download, publication, training or deployment.
Actual datasets, model assets and retained upstream trees are excluded from the
source-only publication described in [PUBLISHING.md](PUBLISHING.md).

The locally retained Laya and Kev source versions have Apache-2.0 license files:

- [Laya `c7527708f9f5220c669d8aa385077cd28d04708a` LICENSE](https://raw.githubusercontent.com/NandhaKishorM/laya/c7527708f9f5220c669d8aa385077cd28d04708a/LICENSE).
- [Kev `557598fced1dada75dfbf36ed144dce309ac6ceb` LICENSE](https://raw.githubusercontent.com/jaredpalmer/kev/557598fced1dada75dfbf36ed144dce309ac6ceb/LICENSE).

Those licenses apply to their respective upstream material, not automatically
to every project component or dataset. Preserve applicable copyright,
attribution and license notices when distributing third-party material; preserve
relevant upstream NOTICE content when supplied. The project's NOTICE does not
invent missing third-party notices or inventory unbundled dependencies.
Trademarks remain with their respective holders; Apache-2.0 does not grant a
general trademark license.

## Official references and review limits

- [Apache License 2.0, January 2004, official text](https://www.apache.org/licenses/LICENSE-2.0.txt): the root LICENSE contains the full terms and appendix, without project-specific changes to the license text. Its content was checked against the official text on 2026-10-21, including all nine sections and the appendix.
- [OpenAI Terms of Use, effective January 1, 2026](https://openai.com/policies/terms-of-use/): the output-ownership clause is limited to rights between the user and OpenAI, to the extent permitted by law, and does not extend to other users' output or Third Party Output. Account type, region and applicable service terms still matter.
- [GitHub: Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository): public availability and an express open-source license are distinct.

This record documents the owner's decision, observed source evidence and
licensing boundaries. It is not a legal opinion, an exhaustive copyright or
patent search, or a guarantee that no unknown third-party claim exists. The
Apache-2.0 warranty and liability terms govern; no additional non-infringement
warranty is made here.
