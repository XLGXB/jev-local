import contextlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from decision_deploy.common import digest
from decision_deploy.identity import decision_source_hashes
from decision_deploy.runtime import Predictor, Runtime


REQUEST = {"state": "visible only", "questions": {"q": {"type": "choice", "instructions": "select",
           "criteria": {"a": "first", "b": "second", "c": "third"}}}}


class RuntimeContractTests(unittest.TestCase):
    def test_shared_candidate_code_changes_identity(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "data.py"
            source.write_text("KEYS = ['false', 'true']\n", encoding="utf-8")
            with patch("decision_deploy.identity.inspect.getfile", return_value=str(source)):
                first = decision_source_hashes()
                source.write_text("KEYS = ['true', 'false']\n", encoding="utf-8")
                second = decision_source_hashes()
            self.assertNotEqual(first, second)
            self.assertNotEqual(digest({"decision_source_sha256": first}), digest({"decision_source_sha256": second}))

    def test_native_identity_includes_actual_adapter(self):
        sources = decision_source_hashes(native_backend=True)
        self.assertEqual(set(sources), {"decision_ft_data", "decision_ft_backends"})
        self.assertTrue(all(len(value) == 64 for value in sources.values()))

    def make_ort(self, output):
        runtime = Runtime.__new__(Runtime)
        runtime.mode = "ort"
        runtime.spec = {}
        runtime.buckets = [8]
        runtime.manifest = {"candidates": 4}
        runtime.frontend = SimpleNamespace(tensors=lambda *args: (8, {}))
        runtime.sessions = {8: SimpleNamespace(get_inputs=lambda: [], run=lambda *args: [np.array(output)])}
        return runtime

    def test_ort_padded_output_preserves_all_three_candidates(self):
        runtime = self.make_ort([[1., 2., 3., -100.]])
        self.assertEqual(runtime.logits(REQUEST)[0][2].tolist(), [1., 2., 3.])

    def test_ort_wrong_shapes_and_nonfinite_fail_closed(self):
        for output in [[[1., 2.]], [1., 2., 3., 4.], [[1., 2., 3., 4.], [1., 2., 3., 4.]], [[1., float("nan"), 3., 4.]]]:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.make_ort(output).logits(REQUEST)

    def test_native_short_batch_is_not_silently_zipped(self):
        runtime = Runtime.__new__(Runtime)
        runtime.mode = "native"; runtime.spec = {}
        runtime.native = SimpleNamespace(forward=lambda atoms: [], cache={"private_fixture": 1})
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict("sys.modules", {"torch": fake_torch}), self.assertRaisesRegex(RuntimeError, "count"):
            runtime.logits(REQUEST)
        self.assertEqual(runtime.native.cache, {})

    def test_predictor_rejects_candidate_count_drift(self):
        atom = SimpleNamespace(kind=0, keys=["a", "b", "c"])
        fake = SimpleNamespace(fingerprint="test", logits=lambda request: [("q", atom, np.array([1., 2.]))])
        with self.assertRaisesRegex(RuntimeError, "shape"):
            Predictor(fake).predict(REQUEST)


if __name__ == "__main__":
    unittest.main()
