"""Mechanical public-demo tests only; synthetic logits do not test semantics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def demo_module():
    path = ROOT / "demos/evidence_ranking/run.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("public_ranking_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicRankingDemoTests(unittest.TestCase):
    def setUp(self):
        self.demo = demo_module()

    def need_demo(self):
        self.assertIsNotNone(self.demo, "CPU public ranking implementation is required")
        return self.demo

    def cases(self):
        return {"schema_version": 1, "dataset_id": "test-public", "provenance": {
            "kind": "new_synthetic_ai_assisted_examples", "license": "Apache-2.0",
            "created": "2026-10-21", "private_data_used": False}, "cases": [
            {"id": qid, "query": "find useful evidence", "candidates": [
                {"id": "a", "source": "public-example:a", "text": "useful evidence"},
                {"id": "b", "source": "public-example:b", "text": "other text"}],
             "max_chars": 12000} for qid in ("q1", "q2")]}

    def labels(self):
        return {"schema_version": 1, "dataset_id": "test-public", "labels": [
            {"id": qid, "expected_id": "a", "category": "direct", "rationale": "LABEL_ONLY_SENTINEL"}
            for qid in ("q1", "q2")]}

    def test_implementation_exists(self):
        self.need_demo()

    def test_case_and_label_boundaries(self):
        d = self.need_demo()
        prepared = d.validate_cases(self.cases())
        self.assertEqual(len(prepared), 2)
        d.validate_labels(self.labels(), self.cases())
        for mutation in ([], self.cases()["cases"] * 2):
            bad = self.cases()
            bad["cases"] = mutation
            with self.assertRaises(ValueError):
                d.validate_cases(bad)
        for labels in ([], self.labels()["labels"][:1], self.labels()["labels"] * 2, self.labels()["labels"] + [
                {"id": "extra", "expected_id": "a", "category": "direct", "rationale": "x"}]):
            bad = self.labels()
            bad["labels"] = labels
            with self.assertRaises(ValueError):
                d.validate_labels(bad, self.cases())
        bad = self.cases()
        bad["cases"][0]["candidates"][1]["id"] = "a"
        with self.assertRaises(ValueError):
            d.validate_cases(bad)

    def test_assets_streamed_and_tampering_rejected(self):
        d = self.need_demo()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"schema_version": 1, "model_id": d.MODEL_ID, "revision": d.REVISION,
                        "model_card": d.MODEL_CARD, "files": []}
            for name in d.ASSET_NAMES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture asset")
                manifest["files"].append({"path": name, "bytes": 13,
                    "sha256": hashlib.sha256(b"fixture asset").hexdigest()})
            result = d.verify_assets(root, manifest)
            self.assertTrue(result["files_verified"])
            (root / "config.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                d.verify_assets(root, manifest)

    def test_predictions_precede_labels_and_model_load_once(self):
        d = self.need_demo()
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, labels, output = root / "cases.json", root / "labels.json", root / "report.json"
            cases.write_text(json.dumps(self.cases()), encoding="utf-8")
            labels.write_text(json.dumps(self.labels()), encoding="utf-8")
            real_read = d.read_json

            def read(path):
                if Path(path) == labels:
                    self.assertEqual(events, ["q1", "q2"])
                return real_read(path)

            def infer(query, safe, native):
                events.append("q" + str(len(events) + 1))
                prompts = d.qc.prompts(query, safe)
                self.assertNotIn("LABEL_ONLY_SENTINEL", json.dumps(prompts))
                return {"scores": [{"id": c["id"], "source_sha256": hashlib.sha256(c["source"].encode()).hexdigest(),
                    "no_logit": 0.0, "yes_logit": 2.0 if c["id"] == "a" else -2.0} for c in safe],
                    "usage": {"judgments": len(safe), "forwards": 1, "returned": len(safe),
                              "token_lengths": {c["id"]: 20 for c in safe}}}, 0.01, 0.005

            with patch.object(d, "read_json", side_effect=read), \
                 patch.object(d, "verify_assets", return_value={"model_id": d.MODEL_ID, "revision": d.REVISION,
                    "files_verified": True, "assets_sha256": "a" * 64}), \
                 patch.object(d, "load_native", return_value=object()) as load, \
                 patch.object(d, "infer_native", side_effect=infer), \
                 patch.object(d, "runtime_metadata", return_value={"test_only": "no native dependencies"}):
                report = d.evaluate(cases, labels, root / "model", output)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(report["metrics"]["native"]["top1_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["native"]["mrr"], 1.0)
            direct = report["metrics"]["native"]["per_category"]["direct"]
            self.assertEqual(direct, {"count": 2, "top1_accuracy": 1.0, "mrr": 1.0, "errors": []})
            self.assertEqual(report["metrics"]["native"]["per_category"]["paraphrase"]["count"], 0)
            self.assertIsNone(report["metrics"]["native"]["per_category"]["paraphrase"]["mrr"])
            self.assertEqual(report["total_validation_and_lexical_seconds"], sum(
                row["timing"]["validation_and_lexical_seconds"] for row in report["cases"]))
            self.assertGreaterEqual(report["total_validation_and_lexical_seconds"], 0.0)
            self.assertTrue(output.is_file())
            self.assertFalse(report["execution_authorized"])
            self.assertNotIn(str(root), output.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                d.evaluate(cases, labels, root / "model", output)

    def test_nonfinite_logits_fail_without_report(self):
        d = self.need_demo()
        with self.assertRaises(ValueError):
            d.validate_logits([[float("nan"), 1.0]], 1)
        with self.assertRaises(ValueError):
            d.validate_logits([[1.0, 2.0], [3.0, 4.0]], 1)
        with self.assertRaises(ValueError):
            d.validate_logits([[1.0]], 1)

        # Independent rank arithmetic, with one synthetic example per category.
        rows, labels = [], {}
        for index, category in enumerate(d.CATEGORIES):
            qid = "metric" + str(index)
            native_order = ("b", "a") if index < 2 else ("a", "b")
            lexical_order = ("a", "b") if index < 2 else ("b", "a")
            rows.append({"id": qid, "recommendation": {"ranking": [{"id": x} for x in native_order]},
                         "primary": {"results": [{"id": x} for x in lexical_order]}})
            labels[qid] = {"expected_id": "a", "category": category}
        metrics = d.add_metrics(rows, labels)
        self.assertAlmostEqual(metrics["native"]["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["native"]["mrr"], 2 / 3)
        self.assertAlmostEqual(metrics["lexical"]["mrr"], 5 / 6)
        self.assertEqual(metrics["native"]["per_category"]["direct"],
                         {"count": 1, "top1_accuracy": 0.0, "mrr": 0.5, "errors": ["metric0"]})

    def test_prediction_failure_writes_no_report(self):
        d = self.need_demo()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, output = root / "cases.json", root / "report.json"
            cases.write_text(json.dumps(self.cases()), encoding="utf-8")
            # Missing labels are intentional: prediction failure must precede any label access.
            with patch.object(d, "verify_assets", return_value={}), \
                 patch.object(d, "load_native", return_value=object()), \
                 patch.object(d, "infer_native", side_effect=ValueError("fixture invalid logits")):
                with self.assertRaises(ValueError):
                    d.evaluate(cases, root / "absent-labels.json", root / "model", output)
            self.assertFalse(output.exists())

    def test_help_and_validation_do_not_import_native_dependencies(self):
        d = self.need_demo()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, labels = root / "cases.json", root / "labels.json"
            cases.write_text(json.dumps(self.cases()), encoding="utf-8")
            labels.write_text(json.dumps(self.labels()), encoding="utf-8")
            code = (
                "import runpy,sys; "
                "sys.argv=[sys.argv[1],*sys.argv[2:]]; "
                "original=__import__; "
                "exec(\"def guarded(name,*args,**kwargs):\\n"
                " if name.split('.')[0] in ('torch','transformers','safetensors','numpy'):\\n"
                "  raise AssertionError('heavy dependency imported')\\n"
                " return original(name,*args,**kwargs)\"); "
                "import builtins; builtins.__import__=guarded; "
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            )
            for args in (("--help",), ("--validate-only", "--cases", str(cases), "--labels", str(labels))):
                result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(Path(d.__file__)), *args],
                    capture_output=True, text=True, timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(result.returncode, 0, result.stderr)
            for debug in (False, True):
                args = ["--validate-only", "--cases", str(root / "missing.json"), "--labels", str(labels)]
                if debug:
                    args.append("--debug")
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(Path(d.__file__)), *args],
                    capture_output=True, text=True, timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(result.returncode, 2)
                diagnostic = json.loads(result.stderr.splitlines()[0])
                self.assertEqual(diagnostic["stage"], "cases_validation")
                self.assertEqual(diagnostic["reason"], "stage_failed")
                self.assertEqual("Traceback" in result.stderr, debug)
                if not debug:
                    self.assertNotIn(str(root), result.stderr)


if __name__ == "__main__":
    unittest.main()
