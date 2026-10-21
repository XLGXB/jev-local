"""Pure fake checks for the opt-in native consumer; no model weights are loaded."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_deploy.common import sha  # noqa: E402
from decision_ft.data import json_bytes  # noqa: E402
from decision_ft.state_json_binding import binding_manifest, write_checkpoint_marker  # noqa: E402
import decision_ft.state_json_native as consumer  # noqa: E402


def request(state=None):
    return {"state": {"empty": state}, "questions": {"q": {
        "type": "choice", "instructions": "Select", "criteria": {"left": "L", "right": "R"},
    }}}


class FakeBackend:
    def __init__(self, outputs=None, failure=None):
        self.cache = {}
        self.model = mock.Mock()
        self.outputs = outputs
        self.failure = failure
        self.forward_calls = 0

    def forward(self, atoms):
        self.forward_calls += 1
        self.cache[atoms[0].fingerprint] = object()
        if self.failure:
            raise self.failure
        return self.outputs


class StateJsonNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = self.root / "checkpoint"
        self.ckpt.mkdir()
        self.binding = binding_manifest()
        write_checkpoint_marker(self.ckpt, expected_binding=self.binding,
                                training_freeze_sha256="a" * 64, epoch=1)
        (self.ckpt / "model.pt").write_bytes(b"fake checkpoint, never loaded")
        self.inventory = {p.relative_to(self.ckpt).as_posix(): sha(p)
                          for p in self.ckpt.rglob("*") if p.is_file()}
        self.source_paths = {}
        for name in consumer.SOURCE_MODULES:
            if name.startswith("kev."):
                path = self.root / (name.replace(".", "_") + ".py")
                path.write_text("# fake source\n", encoding="utf-8")
            else:
                path = Path(importlib.util.find_spec(name).origin)
            self.source_paths[name] = path
        self.sources = {name: sha(path) for name, path in self.source_paths.items()}
        self.spec = {"backend": "kev", "mode": "native", "purpose": "development_evaluation",
                     "checkpoint": str(self.ckpt), "device": "cpu", "weights_dtype": "fp32", "amp": "none"}

    def construct(self, backend, *, spec=None, binding=None, inventory=None, sources=None):
        with (mock.patch.object(consumer, "_source_paths", return_value=self.source_paths),
              mock.patch("decision_ft.backends.load_backend", return_value=backend) as load,
              mock.patch("decision_ft.train.validate_amp"),
              mock.patch.object(consumer, "versions", return_value={"python": "fake"})):
            self.last_load = load
            runtime = consumer.StateJsonNativeRuntime(
                self.spec if spec is None else spec,
                expected_binding=self.binding if binding is None else binding,
                expected_checkpoint_files=self.inventory if inventory is None else inventory,
                expected_sources=self.sources if sources is None else sources,
            )
        return runtime, load

    def test_preload_guards_reject_bad_spec_binding_source_and_tree(self):
        backend = FakeBackend()
        cases = []
        for key, value in [("backend", "toy"), ("mode", "ort"),
                           ("purpose", "production"), ("weights_dtype", "bf16"),
                           ("amp", "bf16"), ("extra", True)]:
            changed = dict(self.spec)
            changed[key] = value
            cases.append((changed, self.binding, self.inventory, self.sources))
        missing = dict(self.spec)
        del missing["purpose"]
        cases.append((missing, self.binding, self.inventory, self.sources))
        bad_binding = copy.deepcopy(self.binding)
        bad_binding["schema_version"] = True
        cases.append((self.spec, bad_binding, self.inventory, self.sources))
        bad_sources = dict(self.sources)
        bad_sources["decision_ft.state_json_native"] = "0" * 64
        cases.append((self.spec, self.binding, self.inventory, bad_sources))
        missing_source = dict(self.sources)
        del missing_source["kev.api"]
        cases.append((self.spec, self.binding, self.inventory, missing_source))
        missing_file = dict(self.inventory)
        del missing_file["model.pt"]
        cases.append((self.spec, self.binding, missing_file, self.sources))
        traversal = dict(self.inventory)
        traversal["../outside"] = "0" * 64
        cases.append((self.spec, self.binding, traversal, self.sources))
        with (mock.patch.object(consumer, "_source_paths", return_value=self.source_paths),
              mock.patch("decision_ft.backends.load_backend", return_value=backend) as load):
            for spec, binding, inventory, sources in cases:
                with self.subTest(spec=spec, inventory=inventory):
                    with self.assertRaises((ValueError, TypeError)):
                        consumer.StateJsonNativeRuntime(
                            spec, expected_binding=binding,
                            expected_checkpoint_files=inventory, expected_sources=sources)
            load.assert_not_called()

    def test_source_bytes_marker_and_extra_tree_fail_before_load(self):
        backend = FakeBackend()
        native_source = self.source_paths["kev.api"]
        native_source.write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.construct(backend)
        self.last_load.assert_not_called()
        self.sources["kev.api"] = sha(native_source)

        marker_path = self.ckpt / "state_representation.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["status"] = "old_native"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        self.inventory["state_representation.json"] = sha(marker_path)
        with self.assertRaisesRegex(ValueError, "marker"):
            self.construct(backend)
        self.last_load.assert_not_called()

        marker["status"] = "experimental_requires_explicit_consumer"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        self.inventory["state_representation.json"] = sha(marker_path)
        (self.ckpt / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.construct(backend)
        self.last_load.assert_not_called()
        self.assertEqual(backend.forward_calls, 0)

    def test_identity_shared_atoms_prediction_and_cache_cleanup(self):
        import torch
        backend = FakeBackend(outputs=[torch.tensor([0.0, 2.0])])
        runtime, load = self.construct(backend)
        load.assert_called_once_with("kev", str(self.ckpt), "cpu", "fp32")
        self.assertEqual(runtime.identity["representation"]["binding"], self.binding)
        self.assertEqual(runtime.identity["checkpoint_files_sha256"], self.inventory)
        pinned_binding = copy.deepcopy(runtime.identity["representation"]["binding"])
        pinned_fingerprint = runtime.fingerprint
        self.binding["policy"]["id"] = "caller mutated after construction"
        self.assertEqual(runtime.expected_binding, pinned_binding)
        self.assertEqual(runtime.identity["representation"]["binding"], pinned_binding)
        self.assertEqual(runtime.fingerprint, pinned_fingerprint)
        qid, atom, logits = runtime.logits(request(None))[0]
        self.assertEqual(qid, "q")
        self.assertEqual(atom.state, '{"empty":null}')
        self.assertEqual(atom.representation_digest,
                         hashlib.sha256(json_bytes(pinned_binding, sort_keys=True)).hexdigest())
        self.assertEqual(logits.tolist(), [0.0, 2.0])
        self.assertEqual(backend.cache, {})
        answer = runtime.predict(request(None))["answers"]["q"]
        self.assertEqual(answer["selected_key"], "right")
        self.assertEqual(set(answer), {"selected_key", "probabilities", "p_selected", "calibrated", "temperature"})
        with self.assertRaisesRegex(ValueError, "calibration"):
            runtime.predict(request(None), calibration={"runtime_fingerprint": "old"})
        self.assertEqual(backend.forward_calls, 2)
        self.assertEqual(backend.cache, {})

    def test_forward_shape_finite_count_failure_and_preprocessing_cleanup(self):
        import torch
        for outputs, message in [([], "count"), ([torch.tensor([[1.0, 2.0]])], "shape"),
                                 ([torch.tensor([1.0])], "shape"),
                                 ([torch.tensor([float("nan"), 0.0])], "nonfinite")]:
            backend = FakeBackend(outputs=outputs)
            runtime, _ = self.construct(backend)
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                runtime.logits(request(None))
            self.assertEqual(backend.cache, {})
        backend = FakeBackend(failure=RuntimeError("forward failed"))
        runtime, _ = self.construct(backend)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            runtime.logits(request(None))
        self.assertEqual(backend.cache, {})
        backend.cache["stale"] = object()
        with self.assertRaises(ValueError):
            runtime.logits({"state": None, "questions": {}})
        self.assertEqual(backend.cache, {})

    def test_mocked_reparse_point_is_rejected_before_load(self):
        fake_path = mock.Mock()
        fake_path.lstat.return_value.st_file_attributes = 0x400
        fake_path.is_symlink.return_value = False
        self.assertTrue(consumer._is_link_or_reparse(fake_path))
        with (mock.patch.object(consumer, "_source_paths", return_value=self.source_paths),
              mock.patch.object(consumer, "_is_link_or_reparse", side_effect=lambda p: p == self.ckpt),
              mock.patch("decision_ft.backends.load_backend") as load):
            with self.assertRaisesRegex(ValueError, "regular directory"):
                consumer.StateJsonNativeRuntime(
                    self.spec, expected_binding=self.binding,
                    expected_checkpoint_files=self.inventory, expected_sources=self.sources)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
