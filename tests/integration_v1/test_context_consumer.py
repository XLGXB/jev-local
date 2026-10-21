"""Context route tests with scripted CPU doubles, not model quality."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import test_represented_native as fixture
import test_review_diff as review_fixture
import test_completion_verification as completion_fixture
from decision_ft.context_backend import context_digest
from jevlocal.integration import context_native as bridge
from jevlocal.integration import represented_native as legacy
from jevlocal.integration import state_json_advisory as pure
from jevlocal.integration.native_review import NativeReviewJudge
from jevlocal.integration.native_completion import NativeCompletionJudge
from jevlocal.integration.review_diff import review_diff
from jevlocal.integration.completion_verification import verify_completion

POLICY = {'schema_version': 1, 'max_state': 768, 'max_branch': 1408,
          'max_packed': 2432, 'max_encoding_tokens': 1024}


class ScriptedRuntime(fixture.FakeRuntime):
    def __init__(self, config):
        super().__init__(config)
        self.config = copy.deepcopy(config)
        self.native.context = copy.deepcopy(config['context'])
        self.native.length = 600
        self.result = None

    def predict(self, req, calibration=None):
        self.calls.append((copy.deepcopy(req), calibration))
        if self.mutate_after:
            self.fingerprint = '0' * 64
        return copy.deepcopy(self.result) if self.result is not None else fixture.raw_prediction(self.config, req)


@contextmanager
def offline_runtime(config, constructor):
    old = sys.modules.get('decision_ft.state_json_native')
    if old is None:
        old = types.ModuleType('decision_ft.state_json_native')
        old.__file__ = str(pure._module_file('decision_ft.state_json_native'))
    fake = types.ModuleType('decision_ft.context_native')
    fake.__file__ = str(pure._module_file('decision_ft.context_native'))
    fake.ContextStateJsonNativeRuntime = constructor
    with fixture.offline_mock_runtime(old, config), mock.patch.dict(
            sys.modules, {'decision_ft.context_native': fake}):
        yield


class ContextConsumerTests(unittest.TestCase):
    def setUp(self):
        base = fixture.RepresentedNativeTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        self.config = copy.deepcopy(base.config)
        self.snapshot = base.snapshot
        c = self.config
        c['route'] = bridge.ROUTE
        c['context'] = copy.deepcopy(POLICY)
        c['spec']['context_sha256'] = context_digest(POLICY)
        c['expected_context_sources'] = {
            name: pure._sha(pure._module_file(name)) for name in bridge._CONTEXT_SOURCES}
        c['expected_adapter_sources'] = {
            name: pure._sha(pure._module_file(name)) for name in bridge._ADAPTER_SOURCES}
        marker = {
            'schema_version': 1, 'status': 'experimental_context_requires_explicit_consumer',
            'context': copy.deepcopy(POLICY), 'context_sha256': context_digest(POLICY),
            'context_backend_sha256': c['expected_context_sources']['decision_ft.context_backend'],
            'encoder_sources_sha256': {name: c['expected_sources'][name]
                                      for name in bridge.ENCODER_SOURCES}}
        self.checkpoint = Path(c['spec']['checkpoint'])
        (self.checkpoint / bridge.CONTEXT_MARKER).write_text(json.dumps(marker), encoding='utf-8')
        c['expected_checkpoint_files'] = pure._checkpoint_files(self.checkpoint)
        c['expected_identity'].update(
            context={'policy': copy.deepcopy(POLICY), 'marker': marker},
            context_sources_sha256=copy.deepcopy(c['expected_context_sources']),
            consumer_source_sha256=c['expected_context_sources']['decision_ft.context_native'],
            checkpoint_files_sha256=copy.deepcopy(c['expected_checkpoint_files']),
            spec={k: v for k, v in c['spec'].items() if k != 'checkpoint'})
        self.refresh_identity(c)

    @staticmethod
    def refresh_identity(c):
        c['expected_fingerprint'] = pure._digest(c['expected_identity'])

    def predictor(self):
        runtime = ScriptedRuntime(self.config)
        return bridge.ContextNativePredictor(runtime, self.config), runtime

    def test_pure_admission_defensive_copy_and_legacy_rejection(self):
        before = 'torch' in sys.modules
        checked = bridge.validate_config(self.config)
        self.assertEqual(checked, self.config)
        self.assertIsNot(checked, self.config)
        checked['context']['max_state'] = 384
        self.assertEqual(self.config['context'], POLICY)
        self.assertEqual('torch' in sys.modules, before)
        with self.assertRaises(ValueError):
            legacy.validate_config(self.config)

    def test_policy_spec_source_identity_rejections_before_constructor(self):
        mutations = [
            lambda c: c['context'].update(max_state=True),
            lambda c: c['context'].update(max_branch=1024),
            lambda c: c['context'].update(max_encoding_tokens=1409),
            lambda c: c['spec'].update(context_sha256='0' * 64),
            lambda c: c['expected_context_sources'].update({'decision_ft.context_backend': '0' * 64}),
            lambda c: c['expected_adapter_sources'].update({'jevlocal.integration.context_native': '0' * 64}),
            lambda c: c['expected_sources'].update({'kev.model': '0' * 64}),
            lambda c: c['expected_identity']['context']['policy'].update(max_encoding_tokens=1000),
            lambda c: c.update(calibration={}),
        ]
        constructor = mock.Mock()
        with offline_runtime(self.config, constructor):
            for mutate in mutations:
                c = copy.deepcopy(self.config)
                mutate(c)
                self.refresh_identity(c)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    bridge.build_predictor(c)
        constructor.assert_not_called()

    def test_marker_tamper_recomputed_inventory_and_identity_still_fails(self):
        c = copy.deepcopy(self.config)
        path = self.checkpoint / bridge.CONTEXT_MARKER
        marker = json.loads(path.read_text(encoding='utf-8'))
        marker['status'] = 'qualified'
        path.write_text(json.dumps(marker), encoding='utf-8')
        c['expected_checkpoint_files'] = pure._checkpoint_files(self.checkpoint)
        c['expected_identity']['checkpoint_files_sha256'] = copy.deepcopy(c['expected_checkpoint_files'])
        c['expected_identity']['context']['marker'] = marker
        self.refresh_identity(c)
        with self.assertRaisesRegex(ValueError, 'context marker status'):
            bridge.validate_config(c)

    def test_complete_inventory_and_missing_marker_fail_before_model(self):
        constructor = mock.Mock()
        path = self.checkpoint / bridge.CONTEXT_MARKER
        data = path.read_bytes()
        with offline_runtime(self.config, constructor):
            path.unlink()
            with self.assertRaises(ValueError):
                bridge.build_predictor(self.config)
            path.write_bytes(data)
            (self.checkpoint / 'extra.json').write_text('{}', encoding='utf-8')
            with mock.patch.object(pure, '_checkpoint_files') as reader:
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    bridge.build_predictor(self.config)
                reader.assert_not_called()
        constructor.assert_not_called()

    def test_cache_offline_and_metadata_checks_precede_constructor(self):
        constructor = mock.Mock()
        with offline_runtime(self.config, constructor):
            for key, value in [('HF_HUB_OFFLINE', '0'), ('HF_HUB_CACHE', str(self.snapshot))]:
                with mock.patch.dict(os.environ, {key: value}), self.assertRaises(ValueError):
                    bridge.build_predictor(self.config)
            with mock.patch.object(legacy, '_read_base_metadata', return_value={'base': 'wrong'}):
                with self.assertRaisesRegex(ValueError, 'base metadata'):
                    bridge.build_predictor(self.config)
            (self.snapshot / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                bridge.build_predictor(self.config)
        constructor.assert_not_called()

    def test_actual_build_wires_context_once_and_no_fallback(self):
        made = []
        def construct(spec, **kwargs):
            self.assertEqual(spec, self.config['spec'])
            self.assertEqual(kwargs, {key: self.config[key] for key in (
                'expected_binding', 'expected_checkpoint_files', 'expected_sources',
                'context', 'expected_context_sources')})
            runtime = ScriptedRuntime(self.config)
            made.append(runtime)
            return runtime
        with offline_runtime(self.config, construct):
            predictor = bridge.build_predictor(self.config)
            self.assertEqual(bridge.identity(predictor, self.config), self.config['expected_fingerprint'])
            self.assertEqual(len(made), 1)
        constructor = mock.Mock(side_effect=RuntimeError('private implementation detail'))
        with offline_runtime(self.config, constructor):
            with self.assertRaisesRegex(ValueError, 'no fallback') as raised:
                bridge.build_predictor(self.config)
        self.assertNotIn('private implementation detail', str(raised.exception))
        constructor.assert_called_once()

    def test_long_preflight_raw_order_and_cache_cleanup(self):
        predictor, runtime = self.predictor()
        request = fixture.request()
        self.assertEqual(predictor.predict(request), fixture.raw_prediction(self.config, request))
        self.assertEqual(runtime.calls, [(request, None)])
        self.assertTrue(all(state == '{"exit":1}' for _, state, _ in runtime.native.calls))
        self.assertEqual(runtime.native.cache, {})
        for invalid, length in [('risk', 600), (None, 1025)]:
            runtime.native.invalid_at, runtime.native.length = invalid, length
            with self.assertRaisesRegex(ValueError, 'truncated, over budget'):
                predictor.predict(request)
            self.assertEqual(len(runtime.calls), 1)
            self.assertEqual(runtime.native.cache, {})
        runtime.native.invalid_at, runtime.native.length = None, 1024
        predictor.predict(request)
        self.assertEqual(len(runtime.calls), 2)

    def test_raw_reply_and_runtime_policy_drift_never_accepted(self):
        predictor, runtime = self.predictor()
        req = fixture.request()
        for change in [
            lambda r: r['answers']['done'].update(calibrated=True),
            lambda r: r['answers']['risk'].update(temperature=0.5),
            lambda r: r.update(answers=dict(reversed(list(r['answers'].items())))),
            lambda r: r['answers']['next'].update(probabilities={'fix': 0.0, 'finish': 1.0}),
        ]:
            runtime.result = fixture.raw_prediction(self.config, req)
            change(runtime.result)
            with self.assertRaises(ValueError):
                predictor.predict(req)
            self.assertEqual(runtime.native.cache, {})
        runtime.result = None
        runtime.mutate_after = True
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            predictor.predict(req)
        runtime.mutate_after = False
        runtime.fingerprint = self.config['expected_fingerprint']
        runtime.native.context['max_encoding_tokens'] = 1000
        calls = len(runtime.calls)
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            predictor.predict(req)
        self.assertEqual(len(runtime.calls), calls)
        self.assertEqual(runtime.native.cache, {})

    def test_exception_clears_cache_and_metadata_is_defensive(self):
        predictor, runtime = self.predictor()
        runtime.predict = mock.Mock(side_effect=RuntimeError('private request'))
        with self.assertRaisesRegex(ValueError, 'no fallback') as raised:
            predictor.predict(fixture.request())
        self.assertNotIn('private request', str(raised.exception))
        runtime.predict.assert_called_once()
        self.assertEqual(runtime.native.cache, {})
        metadata = predictor.metadata
        self.assertFalse(metadata['execution_authorized'])
        self.assertFalse(metadata['calibrated'])
        self.assertEqual(metadata['semantic_quality'], 'unqualified')
        metadata['context']['max_state'] = 384
        self.assertEqual(predictor.metadata['context'], POLICY)
        with self.assertRaises(AttributeError):
            predictor.runtime.fingerprint = '0' * 64

    def test_real_f04_core_and_native_judge_preserve_review_floor(self):
        core = review_fixture.ReviewDiffTests()
        core.setUp()
        self.addCleanup(core.doCleanups)
        predictor, runtime = self.predictor()
        request = core.request([core.contract(category='security', mandatory_review=True)])
        original = copy.deepcopy(request)
        result = review_diff(request, judge_factory=lambda: NativeReviewJudge(predictor))
        hunk = result['changes'][0]['hunks'][0]
        self.assertEqual(hunk['risk_score'], 0)
        self.assertTrue(hunk['human_review_required'])
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['review_complete'])
        self.assertEqual(result['semantic_quality'], 'unqualified')
        self.assertEqual(request, original)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.native.cache, {})
        state = runtime.calls[0][0]['state']
        self.assertEqual(state['contracts'][0]['text'], core.contract_file.read_bytes().decode('utf-8'))

    def test_real_f05_core_and_native_judge_keep_evidence_guard(self):
        core = completion_fixture.CompletionTests()
        core.setUp()
        self.addCleanup(core.doCleanups)
        predictor, runtime = self.predictor()
        # Synthetic receipts are not actual test execution attestations.
        request = core.request(test=core.make_test_receipt(), diff=core.diff_receipt())
        original = copy.deepcopy(request)
        result = verify_completion(request, judge_factory=lambda: NativeCompletionJudge(predictor))
        self.assertEqual(result['claims'][0]['status'], 'supported')
        self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['semantic_quality'], 'unqualified')
        self.assertEqual(request, original)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.native.cache, {})
        self.assertEqual(runtime.calls[0][0]['state']['evidence'][0]['receipt']['output'], 'OK')
        request = core.request(test=core.make_test_receipt(status='failed', exit_code=1),
                               diff=core.diff_receipt())
        blocked_factory = mock.Mock(side_effect=AssertionError('must not load model'))
        result = verify_completion(request, judge_factory=blocked_factory)
        self.assertEqual(result['claims'][0]['status'], 'contradicted')
        blocked_factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
