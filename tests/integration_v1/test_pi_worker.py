"""Mechanical Pi worker wire tests; all predictors are fakes, never models."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from decision_ft.data import option_keys
from jevlocal.integration import context_native, pi_worker


def request(questions=None, *, request_id="request-1", observation_id="observation-1"):
    return {"request_id": request_id, "observation_id": observation_id,
            "state": {"public": "bounded"}, "questions": questions or {
                "q": {"type": "noul", "instructions": "Choose whether to continue."}}}


def frame(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


class FakePredictor:
    def __init__(self, *, fingerprint="fake-runtime", fail=False, stale=False,
                 invalid=False, contradictory_calibration=False):
        self.fingerprint = fingerprint
        self.fail = fail
        self.stale = stale
        self.invalid = invalid
        self.contradictory_calibration = contradictory_calibration
        self.calls = []

    @property
    def metadata(self):
        return {"schema_version": 1, "kind": "fake-context-native",
                "runtime_fingerprint": self.fingerprint, "calibration": None,
                "calibrated": False, "advisory_only": True,
                "execution_authorized": False, "semantic_quality": "unqualified"}

    def predict(self, public_request):
        self.calls.append(public_request)
        if self.fail:
            print("synthetic native stdout")
            raise RuntimeError("synthetic prediction failure: secret state")
        answers = {}
        for qid, question in public_request["questions"].items():
            keys = option_keys(question)
            probabilities = {key: float(index == 0) for index, key in enumerate(keys)}
            if self.invalid:
                probabilities[keys[-1]] = float("nan")
            answers[qid] = {"selected_key": keys[0], "probabilities": probabilities,
                            "p_selected": 1.0,
                            "calibrated": self.contradictory_calibration}
        result = {"runtime_fingerprint": self.fingerprint, "answers": answers,
                  "inference_total_ms": 0.5}
        if self.stale:
            result["observation_id"] = "old-observation"
        return result


def packets(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


class PiWorkerTests(unittest.TestCase):
    def test_resource_guard_is_lazy_and_audits_before_reply(self):
        predictor = FakePredictor()
        factory = mock.Mock(return_value=predictor)
        guard = mock.Mock()
        events = []
        guard.before_factory.side_effect = lambda: events.append("before_factory")
        guard.after_factory.side_effect = lambda: events.append("after_factory")
        guard.before_predict.side_effect = lambda: events.append("before_predict")
        guard.after_predict.side_effect = lambda: events.append("after_predict")
        guard.record.side_effect = lambda *args: events.append(("audit", *args))
        bad = request()
        bad["questions"]["q"]["label"] = True
        worker = pi_worker.Worker({}, factory, guard)
        first, fatal = worker.handle(frame(bad))
        self.assertFalse(fatal)
        self.assertEqual(first["error"]["code"], "invalid_request")
        guard.before_factory.assert_not_called()
        factory.assert_not_called()
        second, fatal = worker.handle(frame(request()))
        self.assertFalse(fatal)
        self.assertTrue(second["ok"])
        self.assertEqual(events, [("audit", "request", "invalid_request"),
                                  "before_factory", "after_factory", "before_predict",
                                  "after_predict", ("audit", "request")])
        output = io.BytesIO()
        self.assertEqual(pi_worker.run_stream(io.BytesIO(), output, worker), 0)
        guard.close.assert_called_once()
        self.assertEqual(output.getvalue(), b"")

    def test_resource_fault_poison_is_sanitized_and_audit_fault_fails_closed(self):
        for method, expected in (("before_factory", "initialization_failed"),
                                 ("after_predict", "prediction_failed"),
                                 ("record", "prediction_failed")):
            with self.subTest(method=method):
                guard = mock.Mock()
                getattr(guard, method).side_effect = pi_worker.ResourceFault("SECRET cap")
                worker = pi_worker.Worker({}, lambda _: FakePredictor(), guard)
                packet, fatal = worker.handle(frame(request()))
                self.assertTrue(fatal)
                self.assertTrue(worker.poisoned)
                self.assertEqual(packet["error"]["code"], expected)
                self.assertNotIn("SECRET", json.dumps(packet))

    def test_three_types_two_requests_construct_once_and_bind_metadata(self):
        predictor = FakePredictor()
        factory = mock.Mock(return_value=predictor)
        questions = {
            "n": {"type": "noul", "instructions": "Proceed?"},
            "c": {"type": "choice", "instructions": "Choose current allowed tool.",
                  "criteria": {"tool.alpha": "First active tool", "tool.beta": "Second active tool"}},
            "s": {"type": "score", "instructions": "Rate public evidence.",
                  "criteria": ["weak", "strong"]},
        }
        input = io.BytesIO(frame(request(questions)) +
                           frame(request(request_id="request-2", observation_id="observation-2")))
        output = io.BytesIO()
        self.assertEqual(pi_worker.run_stream(input, output, pi_worker.Worker({"pin": 1}, factory)), 0)
        factory.assert_called_once_with({"pin": 1})
        self.assertEqual(len(predictor.calls), 2)
        first, second = packets(output)
        self.assertEqual(set(first), {"schema_version", "request_id", "observation_id",
                                      "ok", "reply", "metadata"})
        self.assertTrue(first["ok"])
        self.assertEqual(set(first["reply"]["answers"]), {"n", "c", "s"})
        self.assertEqual(set(first["reply"]["answers"]["c"]["probabilities"]),
                         {"tool.alpha", "tool.beta"})
        self.assertEqual(first["reply"]["runtime_fingerprint"],
                         first["metadata"]["runtime_fingerprint"])
        self.assertTrue(first["metadata"]["advisory_only"])
        self.assertFalse(first["metadata"]["execution_authorized"])
        self.assertEqual(second["reply"]["observation_id"], "observation-2")

    def test_bad_request_never_constructs_model_and_next_valid_one_can(self):
        predictor = FakePredictor()
        factory = mock.Mock(return_value=predictor)
        bad = request()
        bad["questions"]["q"]["label"] = True
        output = io.BytesIO()
        self.assertEqual(pi_worker.run_stream(io.BytesIO(frame(bad) + frame(request())),
                                              output, pi_worker.Worker({}, factory)), 0)
        first, second = packets(output)
        self.assertEqual(first["error"]["code"], "invalid_request")
        self.assertEqual(first["request_id"], "request-1")
        self.assertTrue(second["ok"])
        factory.assert_called_once()
        self.assertEqual(len(predictor.calls), 1)

    def test_oversize_utf8_and_duplicate_keys_never_construct(self):
        factory = mock.Mock(return_value=FakePredictor())
        invalid = [
            b"x" * (pi_worker.REQUEST_MAX_BYTES + 1) + b"\n",
            b"\xff\n",
            b'{"request_id":"r","request_id":"other","observation_id":"o"}\n',
        ]
        for value in invalid:
            with self.subTest(kind=value[:16]):
                output = io.BytesIO()
                pi_worker.run_stream(io.BytesIO(value), output, pi_worker.Worker({}, factory))
                self.assertEqual(packets(output)[0]["error"]["code"], "invalid_request")
        factory.assert_not_called()

    def test_deep_invalid_json_still_returns_one_packet(self):
        factory = mock.Mock(return_value=FakePredictor())
        deep = (b'{"request_id":"r","observation_id":"o","state":' +
                b"[" * 2000 + b"0" + b"]" * 2000 + b',"questions":{}}\n')
        output = io.BytesIO()
        self.assertEqual(pi_worker.run_stream(io.BytesIO(deep), output,
                                              pi_worker.Worker({}, factory)), 0)
        self.assertEqual(len(packets(output)), 1)
        self.assertEqual(packets(output)[0]["error"]["code"], "invalid_request")
        factory.assert_not_called()

    def test_stale_invalid_reply_and_metadata_are_terminal_without_retry(self):
        variants = (FakePredictor(stale=True), FakePredictor(invalid=True),
                    FakePredictor(contradictory_calibration=True))
        for predictor in variants:
            with self.subTest(predictor=predictor.__dict__):
                factory = mock.Mock(return_value=predictor)
                output = io.BytesIO()
                result = pi_worker.run_stream(io.BytesIO(frame(request()) + frame(request(
                    request_id="request-2", observation_id="observation-2"))),
                    output, pi_worker.Worker({}, factory))
                self.assertEqual(result, 1)
                self.assertEqual(len(packets(output)), 1)
                self.assertEqual(packets(output)[0]["error"]["code"], "prediction_failed")
                self.assertEqual(len(predictor.calls), 1)
                factory.assert_called_once()

        predictor = FakePredictor()
        with mock.patch.object(FakePredictor, "metadata", new_callable=mock.PropertyMock,
                               return_value={"runtime_fingerprint": "wrong"}):
            output = io.BytesIO()
            self.assertEqual(pi_worker.run_stream(io.BytesIO(frame(request())), output,
                                                  pi_worker.Worker({}, lambda _: predictor)), 1)
            self.assertEqual(packets(output)[0]["error"]["code"], "prediction_failed")

    def test_initialization_and_prediction_failure_are_terminal(self):
        for factory in (mock.Mock(side_effect=RuntimeError("secret config")),
                        mock.Mock(return_value=FakePredictor(fail=True))):
            with self.subTest(factory=factory):
                output = io.BytesIO()
                input = io.BytesIO(frame(request()) + frame(request(request_id="request-2")))
                self.assertEqual(pi_worker.run_stream(input, output, pi_worker.Worker({}, factory)), 1)
                self.assertEqual(len(packets(output)), 1)
                self.assertIn(packets(output)[0]["error"]["code"],
                              ("initialization_failed", "prediction_failed"))
                self.assertNotIn("secret", output.getvalue().decode("utf-8"))
                factory.assert_called_once()

    def test_direct_handle_cannot_reuse_poisoned_worker(self):
        for factory in (mock.Mock(side_effect=RuntimeError("failed initialization")),
                        mock.Mock(return_value=FakePredictor(fail=True))):
            with self.subTest(factory=factory):
                worker = pi_worker.Worker({}, factory)
                packet, terminal = worker.handle(frame(request()))
                self.assertTrue(terminal)
                self.assertFalse(packet["ok"])
                with self.assertRaisesRegex(RuntimeError, "terminal"):
                    worker.handle(frame(request(request_id="request-2")))
                factory.assert_called_once()
                if worker.predictor is not None:
                    self.assertEqual(len(worker.predictor.calls), 1)

    def test_main_admits_explicit_config_and_wires_existing_builder_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text('{"schema_version":1}', encoding="utf-8")
            admitted = {"expected_base_cache": {"hf_home": directory,
                                                  "hub_cache": directory}}
            with mock.patch.object(context_native, "validate_config", return_value=admitted) as validate, \
                 mock.patch.object(context_native, "build_predictor") as builder, \
                 mock.patch.object(pi_worker, "run_stream", return_value=0) as run, \
                 mock.patch.object(pi_worker.os, "dup", return_value=100), \
                 mock.patch.object(pi_worker.os, "dup2"), \
                 mock.patch.object(pi_worker.os, "fdopen", return_value=io.BytesIO()), \
                 mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(pi_worker.main(["--config", str(config_path)]), 0)
                validate.assert_called_once_with({"schema_version": 1})
                worker = run.call_args.args[2]
                self.assertIs(worker.factory, builder)
                self.assertIsNone(worker.predictor)
                builder.assert_not_called()
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_main_passes_optional_policy_without_native_load(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_bytes(b'{"schema_version":1}')
            expected_sha = __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()
            admitted = {"expected_base_cache": {"hf_home": directory,
                                                  "hub_cache": directory}}
            with mock.patch.object(context_native, "validate_config", return_value=admitted), \
                 mock.patch.object(context_native, "build_predictor") as builder, \
                 mock.patch.object(pi_worker.ResourceGuard, "from_file") as admit, \
                 mock.patch.object(pi_worker, "run_stream", return_value=0) as run, \
                 mock.patch.object(pi_worker.os, "dup", return_value=100), \
                 mock.patch.object(pi_worker.os, "dup2"), \
                 mock.patch.object(pi_worker.os, "fdopen", return_value=io.BytesIO()):
                self.assertEqual(pi_worker.main(["--config", str(config_path),
                                                 "--resource-policy", str(Path(directory) / "policy.json")]), 0)
                admit.assert_called_once_with(str(Path(directory) / "policy.json"),
                                              expected_sha, admitted)
                self.assertIs(run.call_args.args[2].resource, admit.return_value)
                builder.assert_not_called()

    def test_invalid_policy_admission_fails_before_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_bytes(b'{"schema_version":1}')
            admitted = {"expected_base_cache": {"hf_home": directory,
                                                  "hub_cache": directory},
                        "spec": {"device": "cuda"}}
            with mock.patch.object(context_native, "validate_config", return_value=admitted), \
                 mock.patch.object(context_native, "build_predictor") as builder:
                self.assertEqual(pi_worker.main(["--config", str(config_path),
                                                 "--resource-policy", "relative.json"]), 2)
                builder.assert_not_called()


    def test_nonabsolute_or_invalid_config_fails_without_process_or_model(self):
        with mock.patch.object(context_native, "build_predictor") as builder:
            self.assertEqual(pi_worker.main(["--config", "relative.json"]), 2)
            builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
