import asyncio
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("cryptography", "playwright", "filelock"))


@unittest.skipUnless(AVAILABLE, "isolated browser dependencies are required")
class ExecutionBoundaryTests(unittest.TestCase):
    def test_host_mode_never_falls_back_to_cdp(self):
        from fastloop.contracts import Guard
        from fastloop.service import config
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"registry.json"
            cfg = {"schema_version": 2, "project_id": "test", "controller": "fastloop",
                   "control_mode": "codex-host", "cdp": "http://127.0.0.1:9",
                   "decision_service": "http://127.0.0.1:8"}
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(Guard, "no_cdp_fallback"):
                config(path)
            cfg["control_mode"] = "isolated-cdp"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(config(path)["controller"], "fastloop")

    def test_execution_requires_verified_quality_files(self):
        from fastloop.contracts import Guard
        from fastloop.service import require_qualification
        with self.assertRaisesRegex(Guard, "qualification_evidence_required"):
            require_qualification({}, SimpleNamespace(expected_runtime_fingerprint="a" * 64))
        cfg = {"qualification": {key: "not-absolute" for key in (
            "receipt", "calibration_path", "evaluation_path", "policy_path", "dataset_dir")}}
        with self.assertRaisesRegex(Guard, "absolute"):
            require_qualification(cfg, SimpleNamespace(expected_runtime_fingerprint="a" * 64))

    def test_busy_target_exits_before_any_browser_or_model_connection(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from filelock import FileLock
        from fastloop.contracts import digest
        from fastloop.service import invoke
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = root/"state"; state.mkdir()
            public = root/"approval-public.pem"
            public.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
            cfg = {"schema_version": 2, "controller": "fastloop", "control_mode": "isolated-cdp",
                   "project_id": "test", "cdp": "http://127.0.0.1:9", "decision_service": "http://127.0.0.1:8",
                   "state_dir": str(state), "approval_public_key": str(public),
                   "tasks": {"t": {"goal": "fixture", "allowed_origins": ["http://127.0.0.1:9"],
                                    "verifier": {"selector": "#done"}}}}
            path = root/"registry.json"; path.write_text(json.dumps(cfg), encoding="utf-8")
            lock = FileLock(str(state/(digest([cfg["cdp"], "target"]) + ".lock")))
            with lock, patch("fastloop.service.async_playwright", side_effect=AssertionError("must not connect")), \
                    patch("fastloop.service.HTTPDecision", side_effect=AssertionError("must not connect")):
                result = asyncio.run(invoke(path, "start", "r", task_id="t", target_id="target", dry_run=True))
            self.assertEqual(result, {"status": "busy", "run_id": "r"})

    def test_action_cutoff_is_bound_to_the_verified_policy_bytes(self):
        from fastloop.contracts import Guard
        from fastloop.service import require_qualification
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policy = root/"policy.json"
            policy.write_text('{"action_confidence_threshold": 0.99}', encoding="utf-8")
            receipt = root/"receipt.json"
            receipt.write_text(json.dumps({"policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}), encoding="utf-8")
            cfg = {"qualification": {"receipt": str(receipt), "policy_path": str(policy),
                "calibration_path": str(root/"cal.json"), "evaluation_path": str(root/"eval.json"),
                "dataset_dir": str(root/"data")}}
            task = SimpleNamespace(expected_runtime_fingerprint="a"*64, min_probability=0.1)
            # Qualification internals have separate file-backed tests. This test
            # isolates how the browser boundary consumes an accepted receipt.
            with patch("jevlocal.integration.qualification.verify_receipt", return_value=True):
                with self.assertRaisesRegex(Guard, "task_threshold_outside"):
                    require_qualification(cfg, task)
                task.min_probability = 0.99
                require_qualification(cfg, task)
                policy.write_text('{"action_confidence_threshold": 0.1}', encoding="utf-8")
                task.min_probability = 0.1
                with self.assertRaisesRegex(Guard, "task_threshold_outside"):
                    require_qualification(cfg, task)

    def test_legacy_entry_cannot_start_another_controller(self):
        from decision_deploy.cdp import Stop, run
        with self.assertRaisesRegex(Stop, "retired"):
            asyncio.run(run("unused", "t", "target", "r", execute=True))


if __name__ == "__main__":
    unittest.main()
