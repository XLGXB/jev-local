"""CPU-only Pi resource tests. No installed Torch or Kev is imported."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock

from jevlocal.integration.pi_resources import (
    AUDIT_MAX_RECORDS, ResourceFault, ResourceGuard, load_policy,
)


CONFIG = b'{"schema_version":1}'
CONFIG_SHA = hashlib.sha256(CONFIG).hexdigest()


def policy(**changes):
    value = {"schema_version": 1, "config_sha256": CONFIG_SHA,
             "device": "cuda", "max_cuda_allocated_bytes": 400,
             "max_model_loads": 1, "max_tokenizer_loads": 1,
             "max_forward_calls": 2, "max_wall_seconds": 30}
    value.update(changes)
    return value


class FakeCuda:
    def __init__(self, events):
        self.events = events
        self.peak = self.allocated = 0

    def is_available(self):
        return True

    def get_device_properties(self, device):
        self.events.append(("total", device))
        return types.SimpleNamespace(total_memory=1000)

    def set_per_process_memory_fraction(self, fraction, device):
        self.events.append(("cap", fraction, device))

    def max_memory_allocated(self):
        return self.peak

    def memory_allocated(self):
        return self.allocated


def fake_native(events, *, fail_tokenizer=False, fail_forward=False):
    class Model:
        def __init__(self, *args, **kwargs):
            events.append(("model",))

        def forward_batch(self, rows):
            events.append(("forward",))
            if fail_forward:
                raise RuntimeError("SECRET forward failure")
            return rows

    def tokenizer(*args, **kwargs):
        events.append(("tokenizer",))
        if fail_tokenizer:
            raise RuntimeError("SECRET tokenizer failure")
        return object()

    model = types.SimpleNamespace(DecisionModel=Model, load_tokenizer=tokenizer)
    checkpoint = types.SimpleNamespace(DecisionModel=Model, load_tokenizer=tokenizer)
    return model, checkpoint


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "policy.json"
        self.write(policy())

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_strict_policy_and_binding(self):
        path, admitted, digest = load_policy(str(self.path), CONFIG_SHA)
        self.assertEqual(path, self.path)
        self.assertEqual(admitted, policy())
        self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())
        variants = [policy(config_sha256="0" * 64), policy(max_model_loads=True),
                    policy(max_cuda_allocated_bytes=0), policy(max_forward_calls=4097),
                    policy(max_wall_seconds=86401), policy(extra="forbidden"),
                    policy(device="cpu")]
        for value in variants:
            with self.subTest(value=value):
                self.write(value)
                with self.assertRaises(ValueError):
                    load_policy(str(self.path), CONFIG_SHA)
        self.path.write_bytes(b'{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            load_policy(str(self.path), CONFIG_SHA)
        self.path.write_bytes(b"x" * 8193)
        with self.assertRaises(ValueError):
            load_policy(str(self.path), CONFIG_SHA)
        with self.assertRaises(ValueError):
            load_policy("relative.json", CONFIG_SHA)

    def test_symlink_rejected(self):
        alias = self.path.with_name("alias.json")
        try:
            alias.symlink_to(self.path)
        except OSError:
            self.skipTest("local symlinks unavailable")
        with self.assertRaises(ValueError):
            load_policy(str(alias), CONFIG_SHA)

    def test_reparse_and_symlink_metadata_rejected_without_link_privilege(self):
        original = Path.lstat
        def symlink_at_policy(path):
            if path == self.path:
                return types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o777,
                                             st_file_attributes=0)
            return original(path)
        with mock.patch.object(Path, "lstat", new=symlink_at_policy):
            with self.assertRaises(ValueError):
                load_policy(str(self.path), CONFIG_SHA)
        def reparse_at_policy(path):
            if path == self.path:
                return types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600,
                                             st_file_attributes=0x400)
            return original(path)
        with mock.patch.object(Path, "lstat", new=reparse_at_policy):
            with self.assertRaises(ValueError):
                load_policy(str(self.path), CONFIG_SHA)

    def test_cap_before_factory_actual_calls_and_restore(self):
        events = []
        cuda = FakeCuda(events)
        model, checkpoint = fake_native(events)
        original_init = model.DecisionModel.__init__
        original_forward = model.DecisionModel.forward_batch
        original_tokenizer = model.load_tokenizer
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}},
                                        torch_module=types.SimpleNamespace(cuda=cuda),
                                        kev_model=model, kev_checkpoint=checkpoint)
        guard.before_factory()
        checkpoint.load_tokenizer("fake")
        instance = model.DecisionModel()
        guard.after_factory()
        guard.before_predict()
        self.assertEqual(instance.forward_batch([1]), [1])
        guard.after_predict()
        guard.record("request")
        self.assertEqual(events[0], ("total", 0))
        self.assertEqual(events[1][0], "cap")
        self.assertEqual(events[1][2], 0)
        self.assertTrue(0 < events[1][1] < 0.4)
        self.assertEqual(guard.attempted, {"model": 1, "tokenizer": 1, "forward": 1})
        self.assertEqual(guard.completed, guard.attempted)
        audit = guard.audit_path
        guard.close()
        self.assertIs(model.DecisionModel.__init__, original_init)
        self.assertIs(model.DecisionModel.forward_batch, original_forward)
        self.assertIs(model.load_tokenizer, original_tokenizer)
        self.assertIs(checkpoint.load_tokenizer, original_tokenizer)
        records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["phase"] for row in records],
                         ["admitted", "started", "request", "terminal"])
        self.assertEqual(records[2]["attempted"], guard.attempted)
        self.assertEqual(records[2]["own_peak_allocated_bytes"], 0)
        self.assertNotIn("SECRET", audit.read_text(encoding="utf-8"))
        self.assertEqual(set(records[2]), {"schema_version", "pid", "policy_sha256",
                                           "parent_pid",
                                           "config_sha256", "phase", "error_class",
                                           "attempted", "completed", "own_peak_allocated_bytes",
                                           "own_allocated_bytes", "elapsed_ms"})

    def test_parent_pid_is_captured_once_and_stable_in_all_audit_rows(self):
        events = []
        model, checkpoint = fake_native(events)
        with mock.patch("jevlocal.integration.pi_resources.os.getppid",
                        side_effect=[2516, 9999, 8888, 7777, 6666]) as getppid:
            guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                            {"spec": {"device": "cuda"}},
                                            torch_module=types.SimpleNamespace(cuda=FakeCuda(events)),
                                            kev_model=model, kev_checkpoint=checkpoint)
            guard.start()
            guard.record("request")
            guard.close()
        self.assertEqual(getppid.call_count, 1)
        records = [json.loads(line) for line in guard.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["phase"] for row in records],
                         ["admitted", "started", "request", "terminal"])
        self.assertEqual([row["parent_pid"] for row in records], [2516] * 4)

    def test_invalid_parent_pid_fails_before_audit_creation(self):
        for invalid in (0, -1, True, False, "2516", 2516.0, None):
            with self.subTest(invalid=invalid):
                before = set(self.path.parent.iterdir())
                with mock.patch("jevlocal.integration.pi_resources.os.getppid", return_value=invalid):
                    with self.assertRaises(ResourceFault):
                        ResourceGuard(self.path, policy(), "a" * 64)
                self.assertEqual(set(self.path.parent.iterdir()), before)

    def test_overflow_and_attempted_not_completed_on_failure(self):
        events = []
        model, checkpoint = fake_native(events, fail_forward=True)
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}},
                                        torch_module=types.SimpleNamespace(cuda=FakeCuda(events)),
                                        kev_model=model, kev_checkpoint=checkpoint)
        guard.start()
        instance = model.DecisionModel()
        with self.assertRaisesRegex(RuntimeError, "SECRET"):
            instance.forward_batch([])
        self.assertEqual(guard.attempted["forward"], 1)
        self.assertEqual(guard.completed["forward"], 0)
        with self.assertRaises(ResourceFault):
            model.DecisionModel()
        self.assertEqual(guard.attempted["model"], 2)
        self.assertEqual(guard.completed["model"], 1)
        self.assertEqual(events.count(("model",)), 1)
        guard.fault("request", "resource_limit")
        guard.close()

    def test_failed_tokenizer_is_attempted_not_completed_and_audit_is_private(self):
        events = []
        model, checkpoint = fake_native(events, fail_tokenizer=True)
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}},
                                        torch_module=types.SimpleNamespace(cuda=FakeCuda(events)),
                                        kev_model=model, kev_checkpoint=checkpoint)
        guard.start()
        with self.assertRaisesRegex(RuntimeError, "SECRET"):
            checkpoint.load_tokenizer("PRIVATE PROMPT")
        self.assertEqual(guard.attempted["tokenizer"], 1)
        self.assertEqual(guard.completed["tokenizer"], 0)
        guard.fault("request", "initialization_failed")
        path = guard.audit_path
        guard.close()
        contents = path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET", contents)
        self.assertNotIn("PRIVATE PROMPT", contents)

    def test_peak_wall_cap_and_audit_budgets_fail_closed(self):
        events = []
        cuda = FakeCuda(events)
        model, checkpoint = fake_native(events)
        now = [0.0]
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}},
                                        torch_module=types.SimpleNamespace(cuda=cuda),
                                        kev_model=model, kev_checkpoint=checkpoint,
                                        clock=lambda: now[0])
        guard.start()
        cuda.peak = 401
        with self.assertRaises(ResourceFault):
            guard.before_predict()
        cuda.peak = 0
        now[0] = 31.0
        with self.assertRaises(ResourceFault):
            guard.before_predict()
        now[0] = 0.0
        guard._records = AUDIT_MAX_RECORDS
        with self.assertRaises(ResourceFault):
            guard.record("request")
        with self.assertRaises(ResourceFault):
            guard.close()

    def test_postcall_peak_fault_still_counts_completed_call(self):
        events = []
        cuda = FakeCuda(events)
        model, checkpoint = fake_native(events)
        def costly(self, rows):
            cuda.peak = 401
            return rows
        model.DecisionModel.forward_batch = costly
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}},
                                        torch_module=types.SimpleNamespace(cuda=cuda),
                                        kev_model=model, kev_checkpoint=checkpoint)
        guard.start()
        instance = model.DecisionModel()
        with self.assertRaises(ResourceFault):
            instance.forward_batch([1])
        self.assertEqual(guard.attempted["forward"], 1)
        self.assertEqual(guard.completed["forward"], 1)
        guard.fault("request", "resource_limit")
        self.assertNotIn("SECRET", guard.audit_path.read_text(encoding="utf-8"))
        cuda.peak = 0
        guard.close()

    def test_invalid_total_and_no_audit_overwrite(self):
        events = []
        model, checkpoint = fake_native(events)
        self.write(policy(max_cuda_allocated_bytes=1000))
        kwargs = {"torch_module": types.SimpleNamespace(cuda=FakeCuda(events)),
                  "kev_model": model, "kev_checkpoint": checkpoint}
        guard = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                        {"spec": {"device": "cuda"}}, **kwargs)
        with self.assertRaises(ResourceFault):
            guard.start()
        self.assertEqual(events, [("total", 0)])
        guard.close()
        with mock.patch("jevlocal.integration.pi_resources.uuid.uuid4",
                        return_value=types.SimpleNamespace(hex="a" * 32)):
            first = ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                            {"spec": {"device": "cuda"}}, **kwargs)
            try:
                with self.assertRaises(FileExistsError):
                    ResourceGuard.from_file(str(self.path), CONFIG_SHA,
                                            {"spec": {"device": "cuda"}}, **kwargs)
            finally:
                first.close()


if __name__ == "__main__":
    unittest.main()
