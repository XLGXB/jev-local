"""CPU-only contract tests for the explicit Kev context adapter (no model load)."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision_ft import context_backend as cb

_REAL_SOURCE_RESOLVER = cb._encoder_source_hashes


POLICY = {"schema_version": 1, "max_state": 768, "max_branch": 1408,
          "max_packed": 2432, "max_encoding_tokens": 1024}
PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / "vendor" / "kev-557598fced1dada75dfbf36ed144dce309ac6ceb" / "kev"
SOURCE_PATHS = {"kev.model": VENDOR / "model.py", "kev.data": VENDOR / "data.py",
                "decision_ft.backends": PROJECT / "src" / "decision_ft" / "backends.py"}


def actual_source_hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in SOURCE_PATHS.items()}


class FakeAtom:
    def __init__(self, state="short"):
        self.state = state
        self.question = {"type": "choice", "instructions": "Which?",
                         "criteria": {"a": "first", "b": "second"}}
        self.keys = ["a", "b"]
        self.fingerprint = hashlib.sha256(state.encode()).hexdigest()


class FakeData:
    @staticmethod
    def materialize(request):
        q = request["questions"]["q"]
        return {"state": request["state"], "questions": [{"keys": list(q["criteria"]),
                "label": q["label"], "src": q["src"]}]}


class FakeFloat:
    def __init__(self, value):
        self.value = value
        self.converted = False

    def float(self):
        self.converted = True
        return self


class FakeModel:
    default_state = 384

    def __init__(self):
        self.calls = []
        self.force_len = None
        self.truncated = False

    def encode(self, tok, record, *, strict, max_state=384, max_branch=1024):
        self.calls.append((tok, copy.deepcopy(record), strict, max_state, max_branch))
        length = self.force_len if self.force_len is not None else len(record["state"]) + 10
        if strict and len(record["state"]) + 1 > max_state:
            raise ValueError("state overflow")
        if length > max_branch:
            raise ValueError("row overflow")
        return {"ids": [1] * length, "labels": [0], "state_truncated": self.truncated}

    def forward_batch(self, encodings):
        return [[FakeFloat(len(enc["ids"]))] for enc in encodings]


class FakeNative:
    def __init__(self):
        self.model = FakeModel()
        self.device = "cpu"
        self.tok = object()
        self.meta = object()
        self.data = FakeData()
        self.provenance = {"local": {"source": "fake-cpu-only"}}
        self.source_files = [str(SOURCE_PATHS["kev.model"])]
        self.cache = {}
        self.encode_calls = 0
        self.exports = 0

    def encode(self, atom):
        self.encode_calls += 1
        q = copy.deepcopy(atom.question)
        q["label"], q["src"] = atom.keys[0], "local"
        enc = self.model.encode(self.tok, self.data.materialize(
            {"state": atom.state, "questions": {"q": q}}), strict=True)
        enc.pop("labels")
        return enc

    def train_groups(self, backbone_lr, head_lr):
        return [(backbone_lr, head_lr)]

    def source_manifest(self):
        return {str(SOURCE_PATHS["kev.model"]): actual_source_hashes()["kev.model"]}

    def export(self, directory):
        self.exports += 1
        directory.mkdir(exist_ok=False)
        (directory / "native.txt").write_text("fake", encoding="utf-8")


class ContextBackendTests(unittest.TestCase):
    def setUp(self):
        self.source_hashes = actual_source_hashes()
        self.resolver = patch.object(cb, "_encoder_source_hashes", return_value=self.source_hashes)
        self.resolver.start()
        self.addCleanup(self.resolver.stop)

    def backend(self, native=None, context=None):
        return cb.ContextKevBackend(native or FakeNative(), context or POLICY,
                                    expected_encoder_sources=self.source_hashes)

    def test_loaded_encoder_origin_must_match_hashed_import_origin(self):
        specs = lambda name: SimpleNamespace(origin=str(SOURCE_PATHS[name]))
        with patch.object(cb.importlib.util, "find_spec", side_effect=specs):
            with patch.dict(sys.modules, {"kev.model": SimpleNamespace(__file__=str(SOURCE_PATHS["kev.model"]))}):
                self.assertEqual(_REAL_SOURCE_RESOLVER(), self.source_hashes)
            for loaded in (SimpleNamespace(),
                           SimpleNamespace(__file__=str(SOURCE_PATHS["kev.data"]))):
                with self.subTest(loaded=vars(loaded)), patch.dict(sys.modules, {"kev.model": loaded}):
                    with self.assertRaisesRegex(ValueError, "loaded encoder source origin differs"):
                        _REAL_SOURCE_RESOLVER()

    def test_policy_exact_types_formula_and_detachment(self):
        self.assertEqual(cb.validate_context(POLICY), POLICY)
        for changed in ({**POLICY, "schema_version": True},
                        {**POLICY, "max_state": 383},
                        {**POLICY, "max_state": 7553},
                        {**POLICY, "max_branch": 1409},
                        {**POLICY, "max_encoding_tokens": 768},
                        {**POLICY, "max_encoding_tokens": 1409},
                        {**POLICY, "extra": 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cb.validate_context(changed)
        original = copy.deepcopy(POLICY)
        backend = self.backend(context=original)
        original["max_state"] = 384
        exposed = backend.context
        exposed["max_state"] = 384
        self.assertEqual(backend.context, POLICY)
        self.assertEqual(backend.context_sha256, cb.context_digest(POLICY))
        with self.assertRaises(TypeError):
            cb.ContextKevBackend(FakeNative(), POLICY)

    def test_distinct_context_cache_direct_kwargs_and_legacy_short_parity(self):
        native = FakeNative()
        short = FakeAtom()
        legacy = native.encode(short)
        legacy_calls = len(native.model.calls)
        backend = self.backend(native)
        self.assertEqual(backend.encode(short), legacy)
        self.assertEqual(len(native.model.calls), legacy_calls + 1)
        self.assertIs(backend.encode(short), backend.encode(short))
        self.assertEqual(len(native.model.calls), legacy_calls + 1)
        self.assertEqual(native.encode_calls, 1)  # only our explicit legacy parity call
        self.assertEqual(native.cache, {})
        self.assertEqual(backend.model.calls[-1][2:], (True, 768, 1408))
        self.assertNotIn("labels", backend.encode(short))
        self.assertEqual(short.question, {"type": "choice", "instructions": "Which?",
                                          "criteria": {"a": "first", "b": "second"}})
        self.assertEqual(backend.train_groups(1.0, 2.0), [(1.0, 2.0)])
        self.assertIn(str(Path(cb.__file__)), backend.source_manifest())
        other_policy = {**POLICY, "max_state": 769, "max_branch": 1409, "max_packed": 2433}
        other = self.backend(native, other_policy)
        self.assertNotEqual(other.context_sha256, backend.context_sha256)
        self.assertNotEqual(next(iter(other.encode(short)["ids"]), None), None)
        self.assertEqual(len(other.cache), 1)
        self.assertEqual(len(backend.cache), 1)
        self.assertNotEqual(next(iter(other.cache)), next(iter(backend.cache)))
        long = FakeAtom("x" * 500)
        self.assertEqual(len(backend.encode(long)["ids"]), 510)
        self.assertEqual(backend.model.calls[-1][2:], (True, 768, 1408))
        self.assertEqual(FakeModel.default_state, 384)
        logits = backend.forward([short, long])
        self.assertEqual([x.value for x in logits], [15, 510])
        self.assertTrue(all(x.converted for x in logits))

    def test_reject_truncation_packed_and_explicit_budget(self):
        native = FakeNative()
        backend = self.backend(native)
        native.model.truncated = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            backend.encode(FakeAtom())
        native.model.truncated = False
        native.model.force_len = 1025
        with self.assertRaisesRegex(ValueError, "explicit token budget"):
            backend.encode(FakeAtom())
        native.model.force_len = 1409
        with self.assertRaisesRegex(ValueError, "row overflow"):
            backend.encode(FakeAtom())
        self.assertEqual(backend.cache, {})

    def test_export_and_strict_marker_read(self):
        backend = self.backend()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "checkpoint"
            backend.export(output)
            self.assertTrue((output / "native.txt").is_file())
            backend_sha = hashlib.sha256(Path(cb.__file__).read_bytes()).hexdigest()
            expected = dict(expected_context=POLICY, expected_backend_sha256=backend_sha,
                            expected_encoder_sources=self.source_hashes)
            marker = cb.read_context_marker(output, **expected)
            self.assertEqual(set(marker), cb._MARKER_FIELDS)
            self.assertEqual(marker["status"], "experimental_context_requires_explicit_consumer")
            self.assertEqual(marker["encoder_sources_sha256"], self.source_hashes)
            self.assertNotIn("epoch", marker)
            with self.assertRaisesRegex(ValueError, "backend source hash mismatch"):
                cb.read_context_marker(output, **{**expected, "expected_backend_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "encoder source hash mismatch"):
                cb.read_context_marker(output, **{**expected, "expected_encoder_sources":
                    {**self.source_hashes, "kev.data": "0" * 64}})
            with self.assertRaises(FileExistsError):
                backend.export(output)
            for field, value in (("status", "ready"), ("context_sha256", "0" * 64),
                                 ("context_backend_sha256", "0" * 64),
                                 ("encoder_sources_sha256", {**self.source_hashes, "kev.model": "0" * 64}),
                                 ("schema_version", True), ("context", {**POLICY, "max_state": True})):
                with self.subTest(field=field):
                    (output / cb.CONTEXT_MARKER).write_text(json.dumps({**marker, field: value}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        cb.read_context_marker(output, **expected)
            (output / cb.CONTEXT_MARKER).write_text(json.dumps({**marker, "extra": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                cb.read_context_marker(output, **expected)
            (output / cb.CONTEXT_MARKER).write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cb.read_context_marker(output, **expected)
            (output / cb.CONTEXT_MARKER).unlink()
            with self.assertRaises(FileNotFoundError):
                cb.read_context_marker(output, **expected)


if __name__ == "__main__":
    unittest.main()
