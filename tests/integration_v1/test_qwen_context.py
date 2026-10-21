"""Mechanical Qwen consumer tests; no model, CUDA, or network."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration import context_tools, qwen_context as qc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tools"))
import qwen_rank


class QwenContextTests(unittest.TestCase):
    def setUp(self):
        self.arguments = {"query": "find expiry validation", "candidates": [
            {"id": "b", "source": "auth/session.py", "text": "checks expiry token=SECRET"},
            {"id": "a", "source": "ui/banner.ts", "text": "shows banner"}],
            "max_chars": 6000, "known_secrets": ["SECRET"]}

    def test_snapshot_redaction_prompt_rubric_and_primary(self):
        original = json.loads(json.dumps(self.arguments))
        query, safe, primary, snapshot = qc.prepare(self.arguments)
        self.assertEqual(self.arguments, original)
        self.assertEqual(primary, context_tools.rank_evidence(**self.arguments))
        self.assertEqual({x["id"] for x in primary["results"]}, {"a", "b"})
        self.assertNotIn("SECRET", json.dumps(safe))
        rendered = qc.prompts(query, safe)
        self.assertEqual(len(rendered), 2)
        self.assertIn('"source":"auth/session.py"', rendered[0]["text"])
        self.assertIn(qc.INSTRUCTION, rendered[0]["text"])
        self.assertIn(qc.CRITERIA["true"], rendered[0]["text"])
        self.assertIn(qc.CRITERIA["false"], rendered[0]["text"])
        self.assertEqual(snapshot, qc.digest({"query": query, "candidates": safe}))

    def test_score_tie_order_and_all_ids(self):
        _, safe, _, snapshot = qc.prepare(self.arguments)
        identity = {"backend": qc.BACKEND}
        scores = [{"id": c["id"], "source_sha256": hashlib.sha256(c["source"].encode()).hexdigest(),
                   "no_logit": 0.0, "yes_logit": 0.0} for c in safe]
        usage = {"judgments": 2, "forwards": 1, "returned": 2, "token_lengths": {"a": 10, "b": 11}}
        result = qc.compose(snapshot, safe, identity, "call-1",
                            {"snapshot_sha256": snapshot, "identity": identity,
                             "invocation_id": "call-1", "scores": scores, "usage": usage})
        self.assertEqual([r["id"] for r in result["ranking"]], ["a", "b"])
        self.assertTrue(all(r["score"] == 0.5 for r in result["ranking"]))
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["usage"], usage)

    def test_bad_child_binding_nonfinite_duplicates_and_unknown(self):
        _, safe, _, snapshot = qc.prepare(self.arguments)
        identity = {"backend": qc.BACKEND}
        scores = [{"id": c["id"], "source_sha256": hashlib.sha256(c["source"].encode()).hexdigest(),
                   "no_logit": 0.0, "yes_logit": 1.0} for c in safe]
        child = {"snapshot_sha256": snapshot, "identity": identity, "invocation_id": "call-1",
                 "scores": scores, "usage": {"judgments": 2, "forwards": 1,
                 "returned": 2, "token_lengths": {"a": 10, "b": 11}}}
        for changed in (
            {**child, "snapshot_sha256": "0"*64},
            {**child, "invocation_id": "older-call"},
            {**child, "usage": {**child["usage"], "forwards": 0}},
            {**child, "scores": [scores[0], scores[0]]},
            {**child, "scores": [{**scores[0], "yes_logit": float("nan")}, scores[1]]},
            {**child, "scores": [{**scores[0], "source_sha256": "0"*64}, scores[1]]},
        ):
            with self.assertRaises(ValueError):
                qc.compose(snapshot, safe, identity, "call-1", changed)
        self.assertEqual(qc.unavailable(snapshot, identity, "timeout", "call-1")["ranking"], [])

    def test_json_rejects_duplicate_nonfinite(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.assertRaises(ValueError):
                qc.strict_json(raw)

    def test_manifest_escape_link_and_missing_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"checkpoint"
            root.mkdir()
            for name in qc.CRITICAL:
                (root/name).write_text(name, encoding="utf-8")
            assets = [{"path": name, "bytes": len(name),
                       "sha256": hashlib.sha256(name.encode()).hexdigest()} for name in sorted(qc.CRITICAL)]
            config = {"schema_version": 1, "backend": qc.BACKEND, "revision": "a"*40,
                      "checkpoint": str(root), "assets": assets, "device": "cuda", "dtype": "bfloat16",
                      "max_tokens": 1024, "prompt_version": qc.PROMPT_VERSION, "calibrated": False}
            self.assertEqual(qc.validate_config(config)["backend"], qc.BACKEND)
            for path in ("../escape", "/absolute", "a/../../bad", "C:drive-relative"):
                with self.assertRaises(ValueError):
                    qc.validate_config({**config, "assets": [{**assets[0], "path": path}, *assets[1:]]}, verify_assets=False)
            (root/"config.json").unlink()
            with self.assertRaises(ValueError):
                qc.validate_config(config)
            (root/"config.json").write_text("config.json", encoding="utf-8")
            original_is_symlink = Path.is_symlink
            with patch.object(Path, "is_symlink", autospec=True,
                              side_effect=lambda p: p == root/"config.json" or original_is_symlink(p)):
                with self.assertRaises(ValueError):
                    qc.validate_config(config)

    def test_token_limit_pure_boundary(self):
        self.assertEqual(qc.check_prompt_lengths([{"text": "x"}], lambda *_a, **_kw: [1]*1024), [1024])
        with self.assertRaisesRegex(ValueError, "no truncation"):
            qc.check_prompt_lengths([{"text": "x"}], lambda *_a, **_kw: [1]*1025)
        self.assertEqual(qc.score_pair(-1000, 1000), 1.0)
        with self.assertRaises(ValueError):
            qc.score_pair(float("inf"), 0)

    def test_worker_requires_offline_flags_before_model_import(self):
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}):
            with self.assertRaisesRegex(ValueError, "offline flags"):
                qwen_rank._worker("unused", "unused")

    def test_timeout_reaps_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            input_path, config_path = directory/"input.json", directory/"config.json"
            input_path.write_text(json.dumps(self.arguments), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            class Timed:
                killed = False
                waits = 0
                def wait(self, timeout):
                    self.waits += 1
                    if self.waits == 1:
                        raise subprocess.TimeoutExpired("fake", timeout)
                    return -9
                def kill(self):
                    self.killed = True
            process = Timed()
            identity = {"backend": qc.BACKEND}
            captured = {}
            def fake_popen(command, **kwargs):
                captured.update(command=command, kwargs=kwargs)
                return process
            with patch.object(qwen_rank, "PYTHON", input_path), \
                 patch.object(qc, "validate_config", return_value=identity):
                result = qwen_rank.invoke(input_path, config_path, timeout=1,
                         popen=fake_popen)
            self.assertTrue(process.killed)
            self.assertIn("-I", captured["command"])
            self.assertNotIn("PYTHONPATH", captured["kwargs"]["env"])
            self.assertEqual(captured["kwargs"]["env"]["PYTHONNOUSERSITE"], "1")
            self.assertTrue(result["lifecycle"]["reaped"])
            self.assertEqual(result["recommendation"]["status"], "unavailable")
            self.assertEqual(result["primary"], context_tools.rank_evidence(**self.arguments))
            self.assertGreaterEqual(result["lifecycle"]["elapsed_seconds"], 0)

    def test_output_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"same.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                qwen_rank.invoke(path, path, output_path=path)

    def test_primary_budget_stops_before_qwen_and_keeps_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            input_path, config_path = directory/"input.json", directory/"config.json"
            arguments = {**self.arguments, "max_chars": 35}
            input_path.write_text(json.dumps(arguments), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            with patch.object(qwen_rank, "PYTHON", input_path), \
                 patch.object(qc, "validate_config", side_effect=AssertionError("model config reached")):
                outcome = qwen_rank.invoke(input_path, config_path,
                          popen=lambda *_a, **_kw: self.fail("child started"))
            self.assertEqual(outcome["primary"], context_tools.rank_evidence(**arguments))
            self.assertEqual(outcome["primary"]["status"], "requires_larger_budget")
            self.assertEqual(outcome["recommendation"]["reason"], "primary_budget")
            self.assertFalse(outcome["lifecycle"]["child_started"])

    def test_whole_json_budget_returns_unavailable_without_dropping_primary(self):
        primary = {"status": "ok", "results": [{"id": "a", "quote": "q"}]}
        outcome = {"primary": primary, "recommendation": {
                   "status": "observed", "identity": {"backend": qc.BACKEND},
                   "ranking": [{"id": "a", "score": 1.0, "extra": "x"*500}]}}
        with patch.object(qwen_rank, "MAX_JSON", 400):
            bounded = qwen_rank._bounded(outcome, "s"*64, "call-1")
        self.assertEqual(bounded["primary"], primary)
        self.assertEqual(bounded["recommendation"]["reason"], "output_budget")
        self.assertEqual(bounded["recommendation"]["ranking"], [])

    def test_cleanup_error_keeps_primary_and_invalidates_recommendation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            input_path, config_path = directory/"input.json", directory/"config.json"
            input_path.write_text(json.dumps(self.arguments), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            created = []
            real_mkdtemp = tempfile.mkdtemp
            def create(*args, **kwargs):
                path = real_mkdtemp(*args, **kwargs)
                created.append(Path(path))
                return path
            class Finished:
                pid = 123
                def wait(self, timeout):
                    return 1
                def kill(self):
                    self.fail("already reaped")
            with patch.object(qwen_rank, "PYTHON", input_path), \
                 patch.object(qc, "validate_config", return_value={"backend": qc.BACKEND}), \
                 patch.object(qwen_rank.tempfile, "mkdtemp", side_effect=create), \
                 patch.object(qwen_rank.shutil, "rmtree", side_effect=OSError("fixture cleanup")):
                outcome = qwen_rank.invoke(input_path, config_path,
                          popen=lambda *_a, **_kw: Finished())
            self.assertEqual(outcome["primary"], context_tools.rank_evidence(**self.arguments))
            self.assertEqual(outcome["recommendation"]["reason"], "cleanup_failed")
            self.assertTrue(outcome["lifecycle"]["reaped"])
            for path in created:
                self.assertEqual(path.resolve().parent, Path(tempfile.gettempdir()).resolve())
                shutil.rmtree(path)

    def test_unreaped_child_retains_only_owned_workdir(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            input_path, config_path = directory/"input.json", directory/"config.json"
            input_path.write_text(json.dumps(self.arguments), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            class Unreaped:
                pid = 456
                def wait(self, timeout):
                    raise OSError("fixture wait")
                def kill(self):
                    raise OSError("fixture kill")
            with patch.object(qwen_rank, "PYTHON", input_path), \
                 patch.object(qc, "validate_config", return_value={"backend": qc.BACKEND}), \
                 patch.object(qwen_rank.shutil, "rmtree") as removal:
                outcome = qwen_rank.invoke(input_path, config_path,
                          popen=lambda *_a, **_kw: Unreaped())
                removal.assert_not_called()
            self.assertEqual(outcome["recommendation"]["reason"], "cleanup_failed")
            self.assertFalse(outcome["lifecycle"]["reaped"])
            retained = Path(outcome["lifecycle"]["retained_workdir"])
            self.assertTrue(retained.is_dir())
            self.assertEqual(retained.resolve().parent, Path(tempfile.gettempdir()).resolve())
            shutil.rmtree(retained)

    def test_cleanup_wait_timeout_does_not_reset_five_second_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            input_path, config_path = directory/"input.json", directory/"config.json"
            input_path.write_text(json.dumps(self.arguments), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            class TwiceTimed:
                pid = 789
                def __init__(self):
                    self.waits = []
                    self.kills = 0
                def wait(self, timeout):
                    self.waits.append(timeout)
                    raise subprocess.TimeoutExpired("fake", timeout)
                def kill(self):
                    self.kills += 1
            process = TwiceTimed()
            with patch.object(qwen_rank, "PYTHON", input_path), \
                 patch.object(qc, "validate_config", return_value={"backend": qc.BACKEND}), \
                 patch.object(qwen_rank.shutil, "rmtree") as removal:
                outcome = qwen_rank.invoke(input_path, config_path, timeout=1,
                          popen=lambda *_a, **_kw: process)
                removal.assert_not_called()
            self.assertEqual(len(process.waits), 2)
            self.assertEqual(process.kills, 1)
            self.assertLessEqual(process.waits[1], 5.0)
            self.assertEqual(outcome["recommendation"]["reason"], "cleanup_failed")
            self.assertFalse(outcome["lifecycle"]["reaped"])
            retained = Path(outcome["lifecycle"]["retained_workdir"])
            self.assertTrue(retained.is_dir())
            shutil.rmtree(retained)


if __name__ == "__main__":
    unittest.main()
