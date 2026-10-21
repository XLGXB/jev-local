"""Synthetic context runtime checks; no tokenizer or model is loaded."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from decision_deploy.common import digest, sha
from decision_ft import context_backend, context_native
from decision_ft.state_json_binding import binding_manifest


CONTEXT = {"schema_version": 1, "max_state": 768, "max_branch": 1408,
           "max_packed": 2432, "max_encoding_tokens": 1024}


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.values)


class ContextNativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.binding = binding_manifest()
        self.sources = {name: "a" * 64 for name in context_native._checked_sources.__globals__["SOURCE_MODULES"]}
        self.context_sources = {
            name: sha(Path(sys.modules[name].__file__))
            for name in context_native.CONTEXT_SOURCE_MODULES
        }
        self.encoder_sources = {name: self.sources[name] for name in context_backend.ENCODER_SOURCES}
        self.representation_marker = {
            "schema_version": 1, "status": "experimental_requires_explicit_consumer",
            "binding": copy.deepcopy(self.binding),
            "training_freeze_sha256": "b" * 64, "epoch": 1,
        }
        self.context_marker = {
            "schema_version": 1, "status": "experimental_context_requires_explicit_consumer",
            "context": copy.deepcopy(CONTEXT), "context_sha256": context_backend.context_digest(CONTEXT),
            "context_backend_sha256": self.context_sources["decision_ft.context_backend"],
            "encoder_sources_sha256": copy.deepcopy(self.encoder_sources),
        }
        for name in context_native.CHECKPOINT_FILES:
            path = self.directory / name
            if name == "state_representation.json":
                path.write_text(json.dumps(self.representation_marker), encoding="utf-8")
            elif name == "context_policy.json":
                path.write_text(json.dumps(self.context_marker), encoding="utf-8")
            else:
                path.write_bytes(name.encode("ascii"))
        self.files = {name: sha(self.directory / name) for name in context_native.CHECKPOINT_FILES}
        self.spec = {
            "backend": "kev", "mode": "native", "purpose": "development_evaluation",
            "checkpoint": str(self.directory), "device": "cpu", "weights_dtype": "fp32",
            "amp": "none", "context_sha256": context_backend.context_digest(CONTEXT),
        }
        self.load = mock.Mock()
        self.model = types.SimpleNamespace(eval=mock.Mock())
        self.loaded = types.SimpleNamespace(model=self.model)
        self.load.return_value = self.loaded
        self.wrapped = []

        def wrap(native, context, *, expected_encoder_sources):
            self.assertIs(native, self.loaded)
            self.assertEqual(context, CONTEXT)
            self.assertEqual(expected_encoder_sources, self.encoder_sources)
            output = types.SimpleNamespace(model=self.model, cache={"old": 1},
                                           forward=mock.Mock(side_effect=lambda atoms: [
                                               FakeTensor([0.25] * len(atom.keys)) for atom in atoms
                                           ]))
            self.wrapped.append(output)
            return output

        self.patches = [
            mock.patch.object(context_native, "_checked_sources", return_value=self.sources),
            mock.patch.object(context_backend, "_checked_encoder_sources", side_effect=lambda x: x),
            mock.patch.object(context_backend, "ContextKevBackend", side_effect=wrap),
            mock.patch.dict(sys.modules, {
                "torch": types.SimpleNamespace(inference_mode=lambda: nullcontext(),
                                               cuda=types.SimpleNamespace(is_available=lambda: False)),
                "decision_ft.backends": types.SimpleNamespace(load_backend=self.load),
                "decision_ft.train": types.SimpleNamespace(validate_amp=lambda *args: None),
            }),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make(self, **changes):
        return context_native.ContextStateJsonNativeRuntime(
            changes.get("spec", self.spec), expected_binding=changes.get("binding", self.binding),
            expected_checkpoint_files=changes.get("files", self.files),
            expected_sources=changes.get("sources", self.sources),
            context=changes.get("context", CONTEXT),
            expected_context_sources=changes.get("context_sources", self.context_sources),
        )

    def refresh(self, name):
        self.files[name] = sha(self.directory / name)

    def test_valid_identity_wrapper_and_inherited_cache_clear(self):
        runtime = self.make()
        self.load.assert_called_once_with("kev", str(self.directory), "cpu", "fp32")
        self.assertEqual(len(self.wrapped), 1)
        self.model.eval.assert_called_once()
        self.assertEqual(runtime.identity["context"], {
            "policy": CONTEXT, "marker": self.context_marker,
        })
        self.assertEqual(runtime.identity["context_sources_sha256"], self.context_sources)
        self.assertEqual(runtime.identity["consumer_source_sha256"],
                         self.context_sources["decision_ft.context_native"])
        self.assertNotIn("checkpoint", runtime.identity["spec"])
        self.assertEqual(runtime.fingerprint, digest(runtime.identity))
        request = {"state": {"x": 1}, "questions": {
            "q": {"type": "noul", "instructions": "Continue?"}}}
        rows = runtime.logits(request)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(self.wrapped[0].forward.call_args.args[0]), 1)
        self.assertEqual(self.wrapped[0].cache, {})
        self.wrapped[0].cache["again"] = 1
        with self.assertRaises(ValueError):
            runtime.logits({"bad": "request"})
        self.assertEqual(self.wrapped[0].cache, {})

    def test_invalid_spec_and_context_fail_before_backend(self):
        variants = [
            {"spec": {**self.spec, "context_sha256": "0" * 64}},
            {"spec": {**self.spec, "amp": "bf16"}},
            {"spec": {**self.spec, "extra": 1}},
            {"context": {**CONTEXT, "max_state": 769}},
            {"binding": {**self.binding, "schema_version": 2}},
        ]
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.make(**changed)
        self.load.assert_not_called()

    def test_missing_and_tampered_context_marker_fail_before_backend(self):
        path = self.directory / "context_policy.json"
        path.unlink()
        with self.assertRaises(ValueError):
            self.make()
        path.write_text(json.dumps({**self.context_marker, "status": "wrong"}), encoding="utf-8")
        self.refresh(path.name)
        with self.assertRaisesRegex(ValueError, "status"):
            self.make()
        self.load.assert_not_called()

    def test_changed_context_hash_and_old_marker_fail_before_backend(self):
        path = self.directory / "context_policy.json"
        path.write_text(json.dumps({**self.context_marker, "context_sha256": "0" * 64}), encoding="utf-8")
        self.refresh(path.name)
        with self.assertRaisesRegex(ValueError, "hash"):
            self.make()
        path.write_text(json.dumps(self.context_marker), encoding="utf-8")
        self.refresh(path.name)
        old_path = self.directory / "state_representation.json"
        old_path.write_text(json.dumps({**self.representation_marker, "status": "wrong"}), encoding="utf-8")
        self.refresh(old_path.name)
        with self.assertRaisesRegex(ValueError, "status"):
            self.make()
        self.load.assert_not_called()

    def test_source_and_full_inventory_fail_before_backend(self):
        with mock.patch.object(context_native, "_checked_sources",
                               side_effect=ValueError("native consumer source hash mismatch")):
            with self.assertRaisesRegex(ValueError, "source"):
                self.make()
        bad = {**self.context_sources, "decision_ft.context_native": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source"):
            self.make(context_sources=bad)
        bad_files = dict(self.files)
        bad_files.pop("head.pt")
        with self.assertRaises(ValueError):
            self.make(files=bad_files)
        (self.directory / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.make()
        self.load.assert_not_called()

    def test_backend_constructor_failure_has_no_fallback(self):
        self.load.side_effect = RuntimeError("synthetic load failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
            self.make()
        self.load.assert_called_once()
        self.assertEqual(self.wrapped, [])


if __name__ == "__main__":
    unittest.main()
