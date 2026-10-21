"""Synthetic qualification evidence tests; these do not qualify a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from decision_ft.data import file_sha, leakage_profile, write_jsonl
from jevlocal.integration.qualification import build_receipt, inspect_qualification, verify_receipt


CHECKPOINT = "c" * 64
TYPES = ("choice", "score", "noul")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8")


def row(split: str, kind: str, number: int) -> dict:
    question = {"type": kind, "instructions": "decide"}
    if kind == "choice":
        question.update(criteria={"act": "act", "needs_host": "stop"}, label="act")
    elif kind == "score":
        question.update(criteria=["low", "high"], label=1)
    else:
        question.update(label=True)
    uid = f"{split}-{kind}-{number}"
    return {"state": {"uid": uid}, "questions": {"q": question},
            "_meta": {"id": uid, "group_id": uid, "source": "synthetic-test",
                      "task": "test", "label_source": "human_verified",
                      "training_allowed": True, "split": split}}


def policy() -> dict:
    return {"schema_version": 1, "required_types": list(TYPES),
            "min_calibration_per_type": 2, "min_evaluation_per_type": 2,
            "min_evaluation_total": 6, "min_action_decisions": 2,
            "min_accepted_actions": 1, "min_accuracy": 0.7,
            "max_nll_target": 1.0, "max_brier_target": 0.5,
            "max_ece_verified": 0.2, "max_rps_target": 0.5,
            "max_score_mean_mae_target": 0.5,
            "action_confidence_threshold": 0.95,
            "max_action_error_rate": 0.1, "min_action_coverage": 0.5,
            "max_parity_logit_abs": 0.001,
            "max_parity_probability_abs": 0.0001,
            "min_parity_top1": 1.0, "min_parity_decisions": 1}


def metric(n: int, *, score: bool = False) -> dict:
    out = {"n": n, "n_verified": n, "accuracy_against_supplied_label": 0.9,
           "nll_target": 0.4, "brier_target": 0.2, "ece_verified_15_bins": 0.1}
    if score:
        out.update(rps_target=0.2, score_mean_mae_target=0.2)
    return out


class EvidenceFixture:
    def __init__(self, root: Path, mode: str = "native"):
        self.root = root
        self.data = root / "data"
        self.data.mkdir()
        entries = {}
        for split in ("train", "dev", "calibration", "test"):
            records = [row(split, kind, i) for kind in TYPES for i in range(2)]
            path = self.data / f"{split}.jsonl"
            write_jsonl(path, records)
            entries[split] = {"sha256": file_sha(path), "records": len(records),
                              "questions": len(records), "leakage_profile": leakage_profile(records)}
        save(self.data / "manifest.json", {"schema_version": 1, "seed": 1,
                                            "input_sha256": "i" * 64, "files": entries})
        self.identity = {"spec": {"mode": mode},
                         "artifact": {"checkpoint_sha256": CHECKPOINT} if mode == "native" else
                         {"bundle_manifest_sha256": ""}}
        self.calibration_path = root / "calibration.json"
        self.evaluation_path = root / "evaluation.json"
        self.policy_path = root / "policy.json"
        self.parity_path = root / "parity.json"
        self.golden_path = root / "golden.json"
        self.bundle_path = root / "bundle-manifest.json"
        if mode == "ort":
            save(self.bundle_path, {"checkpoint_sha256": CHECKPOINT})
            self.identity["artifact"]["bundle_manifest_sha256"] = file_sha(self.bundle_path)
        self.fingerprint = digest(self.identity)
        self.calibration = {"stage": "calibrate", "runtime_fingerprint": self.fingerprint,
                            "runtime_identity": self.identity,
                            "dataset_manifest_sha256": file_sha(self.data / "manifest.json"),
                            "calibration_file_sha256": file_sha(self.data / "calibration.jsonl"),
                            "fits": {str(i): {"status": "fitted", "n": 2, "temperature": 1.2}
                                     for i in range(3)},
                            "temperatures": {str(i): 1.2 for i in range(3)}}
        self.evaluation = {"stage": "evaluate", "partition": "test",
                           "runtime_fingerprint": self.fingerprint, "runtime_identity": self.identity,
                           "dataset_manifest_sha256": self.calibration["dataset_manifest_sha256"],
                           "raw": {"all": metric(6)},
                           "calibrated": {"all": metric(6), **{
                               f"type:{kind}": metric(2, score=kind == "score") for kind in TYPES}},
                           "action_coverage_excluding_needs_host": {"0.95": {
                               "decisions": 2, "accepted_actions": 2, "coverage": 1.0,
                               "action_error_rate": 0.0}}}
        self.policy = policy()
        if mode == "ort":
            golden_identity = {"spec": {"mode": "native"},
                               "artifact": {"checkpoint_sha256": CHECKPOINT}}
            save(self.golden_path, {"stage": "golden", "runtime_fingerprint": digest(golden_identity),
                "runtime_identity": golden_identity,
                "samples": [{"request": {"questions": {"q": {}}}, "logits": {"q": [1, 2]}}]})
            self.parity = {"stage": "compare", "status": "PASS", "runtime_fingerprint": self.fingerprint,
                           "runtime_identity": self.identity, "top1_agreement": 1.0,
                           "golden_sha256": file_sha(self.golden_path),
                           "decisions": [{"qid": "q", "max_logit_abs": 0.0001,
                                          "max_probability_abs": 0.00001, "same_top1": True}]}
            save(self.parity_path, self.parity)
        save(self.calibration_path, self.calibration)
        self.evaluation["calibration_sha256"] = file_sha(self.calibration_path)
        save(self.evaluation_path, self.evaluation)
        save(self.policy_path, self.policy)

    def args(self) -> dict:
        out = {"runtime_fingerprint": self.fingerprint, "calibration_path": self.calibration_path,
               "evaluation_path": self.evaluation_path, "policy_path": self.policy_path,
               "dataset_dir": self.data}
        if self.identity["spec"]["mode"] == "ort":
            out.update(parity_path=self.parity_path, golden_path=self.golden_path,
                       bundle_manifest_path=self.bundle_path)
        return out


class QualificationTests(unittest.TestCase):
    def test_native_three_types_and_no_authorization(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            receipt = build_receipt(**f.args())
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(set(receipt["artifact_sha256"]),
                             {"calibration", "evaluation", "dataset_manifest"})
            self.assertFalse(receipt["action_authorized"])
            self.assertTrue(verify_receipt(receipt, **f.args()))
            self.assertFalse(any(p.name == "grant.json" for p in Path(d).iterdir()))

    def test_plain_dicts_cannot_claim_trusted_evidence(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            result = inspect_qualification(runtime_fingerprint=f.fingerprint,
                calibration=f.calibration, evaluation=f.evaluation, policy=f.policy)
            self.assertEqual(result["status"], "FAIL")

    def test_missing_small_nonfinite_fit_and_binding_fail(self):
        mutations = [
            lambda f: f.evaluation.pop("calibrated"),
            lambda f: f.evaluation["calibrated"].pop("type:noul"),
            lambda f: f.evaluation["calibrated"]["type:score"].update(n_verified=1),
            lambda f: f.evaluation["calibrated"]["type:choice"].update(nll_target=float("nan")),
            lambda f: f.evaluation["raw"]["all"].update(nll_target=float("inf")),
            lambda f: f.calibration["fits"]["0"].update(status="bound_hit"),
            lambda f: f.calibration["fits"]["1"].update(n=1),
            lambda f: f.evaluation.update(runtime_fingerprint="e" * 64),
            lambda f: f.evaluation.update(calibration_sha256="0" * 64),
            lambda f: f.evaluation.update(partition="dev"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as d:
                f = EvidenceFixture(Path(d))
                mutate(f)
                save(f.calibration_path, f.calibration)
                # Preserve a real byte binding unless this mutation targets the binding.
                if f.evaluation.get("calibration_sha256") != "0" * 64:
                    f.evaluation["calibration_sha256"] = file_sha(f.calibration_path)
                f.evaluation_path.write_text(json.dumps(f.evaluation), encoding="utf-8")
                self.assertEqual(build_receipt(**f.args())["status"], "FAIL")

    def test_modified_evidence_and_calibration_replacement_invalidate_receipt(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            receipt = build_receipt(**f.args())
            f.evaluation_path.write_text(f.evaluation_path.read_text() + " ", encoding="utf-8")
            self.assertFalse(verify_receipt(receipt, **f.args()))
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            receipt = build_receipt(**f.args())
            f.calibration["temperatures"]["0"] = 1.5
            save(f.calibration_path, f.calibration)
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")
            self.assertFalse(verify_receipt(receipt, **f.args()))

    def test_ort_needs_same_checkpoint_and_numeric_parity(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d), "ort")
            self.assertEqual(build_receipt(**f.args())["status"], "PASS")
            missing = f.args(); missing.pop("parity_path")
            self.assertEqual(build_receipt(**missing)["status"], "FAIL")
            f.parity["decisions"][0]["max_logit_abs"] = 1.0
            save(f.parity_path, f.parity)
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")
            f.parity["decisions"][0]["max_logit_abs"] = 0.0001
            save(f.parity_path, f.parity)
            save(f.bundle_path, {"checkpoint_sha256": "d" * 64})
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")

    def test_runtime_fingerprint_binds_identity_bytes(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            self.assertEqual(build_receipt(**f.args())["status"], "PASS")
            changed = {**f.identity, "untracked_semantics": "changed"}
            f.calibration["runtime_identity"] = changed
            f.evaluation["runtime_identity"] = changed
            save(f.calibration_path, f.calibration)
            f.evaluation["calibration_sha256"] = file_sha(f.calibration_path)
            save(f.evaluation_path, f.evaluation)
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")

    def test_ort_golden_file_and_decision_count_are_bound(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d), "ort")
            self.assertEqual(build_receipt(**f.args())["status"], "PASS")
            golden = json.loads(f.golden_path.read_text(encoding="utf-8"))
            golden["samples"][0]["request"]["questions"]["q"]["instructions"] = "replaced"
            save(f.golden_path, golden)
            # Checkpoint is unchanged; the exact golden file must still be bound.
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")
            golden["samples"][0]["request"]["questions"]["q2"] = {}
            save(f.golden_path, golden)
            f.parity["golden_sha256"] = file_sha(f.golden_path)
            save(f.parity_path, f.parity)
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")

    def test_action_pool_is_not_every_choice_question(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            test_path = f.data / "test.jsonl"
            extra = row("test", "choice", 99)
            extra["questions"]["q"].update(criteria={"a": "A", "b": "B"}, label="a")
            rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
            rows.append(extra)
            write_jsonl(test_path, rows)
            manifest_path = f.data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["test"].update(sha256=file_sha(test_path), records=len(rows),
                questions=len(rows), leakage_profile=leakage_profile(rows))
            save(manifest_path, manifest)
            f.calibration["dataset_manifest_sha256"] = file_sha(manifest_path)
            f.evaluation["dataset_manifest_sha256"] = file_sha(manifest_path)
            f.evaluation["calibrated"]["all"].update(n=7, n_verified=7)
            f.evaluation["calibrated"]["type:choice"].update(n=3, n_verified=3)
            save(f.calibration_path, f.calibration)
            f.evaluation["calibration_sha256"] = file_sha(f.calibration_path)
            save(f.evaluation_path, f.evaluation)
            self.assertEqual(build_receipt(**f.args())["status"], "PASS")

    def test_unverified_source_fails(self):
        with TemporaryDirectory() as d:
            f = EvidenceFixture(Path(d))
            test_path = f.data / "test.jsonl"
            rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["_meta"]["label_source"] = "unverified_teacher"
            write_jsonl(test_path, rows)
            manifest_path = f.data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["test"].update(sha256=file_sha(test_path),
                leakage_profile=leakage_profile(rows))
            save(manifest_path, manifest)
            f.calibration["dataset_manifest_sha256"] = file_sha(manifest_path)
            f.evaluation["dataset_manifest_sha256"] = file_sha(manifest_path)
            save(f.calibration_path, f.calibration)
            f.evaluation["calibration_sha256"] = file_sha(f.calibration_path)
            save(f.evaluation_path, f.evaluation)
            self.assertEqual(build_receipt(**f.args())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
