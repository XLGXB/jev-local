"""Synthetic adapter checks only; no native model is loaded or scored."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from jevlocal.integration import state_json_advisory as advisory


CONFIG = Path('D:/Projects/JevLocal/runtime/kev-state-json-semdecide-01/consumer-config.json')


def envelope():
    return {'request_id': 'r', 'observation_id': 'o', 'state': {'exit': 1},
            'questions': {
                'done': {'type': 'noul', 'instructions': 'Done?'},
                'triage': {'type': 'choice', 'instructions': 'Next?',
                           'criteria': {'finish': 'Done', 'fix': 'Fix'}},
                'severity': {'type': 'score', 'instructions': 'Severity?',
                             'criteria': ['none', 'low', 'high']}}}


def prediction():
    answers = {}
    for qid, keys, fitted in (
            ('done', ['false', 'true'], False),
            ('triage', ['finish', 'fix'], False),
            ('severity', ['0', '1', '2'], True)):
        probs = {key: (1.0 if index == 0 else 0.0) for index, key in enumerate(keys)}
        answers[qid] = {'selected_key': keys[0], 'probabilities': probs,
                        'p_selected': 1.0, 'calibrated': fitted,
                        'temperature': advisory._TEMPERATURE if fitted else 1.0}
    return {'runtime_fingerprint': advisory._FINGERPRINT, 'answers': answers,
            'inference_total_ms': 1.0}


class FakeNative:
    def __init__(self):
        self.cache = {}
        self.calls = []
        self.length = 20
        self.truncated = False

    def encode(self, atom):
        self.calls.append((atom.uid, copy.deepcopy(atom.state), list(atom.keys)))
        self.cache[atom.fingerprint] = True
        return {'ids': [1] * self.length, 'state_truncated': self.truncated}


class FakeRuntime:
    def __init__(self, config):
        self.spec = copy.deepcopy(config['spec'])
        self.identity = copy.deepcopy(config['expected_identity'])
        self.fingerprint = config['expected_fingerprint']
        self.native = FakeNative()
        self.requests = []

    def predict(self, request, calibration=None):
        self.requests.append((copy.deepcopy(request), copy.deepcopy(calibration)))
        return prediction()


class StateJsonAdvisoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding='utf-8'))

    def test_real_pins_validate_without_torch_or_model(self):
        before = 'torch' in sys.modules
        original = copy.deepcopy(self.config)
        checked = advisory.validate_config(self.config)
        self.assertEqual(checked, original)
        self.assertIsNot(checked, self.config)
        checked['spec']['device'] = 'cpu'
        self.assertEqual(self.config, original)
        self.assertEqual('torch' in sys.modules, before)
        self.assertEqual(advisory.metadata(self.config), {
            'schema_version': 1, 'kind': advisory.ROUTE,
            'runtime_fingerprint': advisory._FINGERPRINT,
            'calibration_sha256': advisory._SIDECAR_SHA,
            'advisory_only': True, 'execution_authorized': False,
            'qualification_status': 'not_qualified'})

    def test_config_rejects_schema_identity_and_sidecar_drift(self):
        for change in (
            lambda c: c.update(extra=True),
            lambda c: c.update(route='native_context'),
            lambda c: c['spec'].update(device='cpu'),
            lambda c: c['expected_identity']['spec'].update(device='cpu'),
            lambda c: c['expected_sources'].update({'kev.model': '0' * 64}),
            lambda c: c['expected_checkpoint_files'].pop('head.pt'),
            lambda c: c['calibration'].update(sha256='0' * 64),
            lambda c: c['calibration'].update(path='relative.json'),
        ):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                change(config)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    advisory.validate_config(config)

    def test_source_origin_and_checkpoint_inventory_checked(self):
        with mock.patch.object(advisory, '_module_file', return_value=CONFIG):
            with self.assertRaisesRegex(ValueError, 'source hash'):
                advisory.validate_config(self.config)
        with mock.patch.object(advisory, '_checkpoint_files', return_value={}):
            with self.assertRaisesRegex(ValueError, 'checkpoint tree'):
                advisory.validate_config(self.config)
        original = advisory._read_json

        def wrong_marker(path):
            value = original(path)
            if Path(path).name == 'state_representation.json':
                value['status'] = 'unbound'
            return value

        with mock.patch.object(advisory, '_read_json', side_effect=wrong_marker):
            with self.assertRaisesRegex(ValueError, 'marker'):
                advisory.validate_config(self.config)

    def test_loaded_source_origin_normalized_but_not_substituted(self):
        name = 'decision_ft.state_json_native'
        actual = advisory._module_file(name)
        equivalent = types.ModuleType(name)
        equivalent.__file__ = str(actual).replace('\\', '/').upper()
        with mock.patch.dict(sys.modules, {name: equivalent}):
            self.assertEqual(advisory._module_file(name), actual)
        different = types.ModuleType(name)
        different.__file__ = str(CONFIG)
        with mock.patch.dict(sys.modules, {name: different}):
            with self.assertRaisesRegex(ValueError, 'origin differs'):
                advisory._module_file(name)

    def test_sidecar_byte_sha_checked_before_parse(self):
        with mock.patch.object(advisory, '_sha', side_effect=lambda path: (
                '0' * 64 if Path(path) == advisory._SIDECAR_PATH else
                advisory._SIDECAR_SHA)):
            with self.assertRaisesRegex(ValueError, 'byte SHA'):
                advisory._sidecar(self.config)

    def test_mixed_reply_policy_and_protocol(self):
        env = envelope()
        pred = prediction()
        reply = advisory.validate_prediction(env, pred, self.config)
        self.assertEqual(set(reply), {'request_id', 'observation_id',
                                     'runtime_fingerprint', 'answers', 'inference_total_ms'})
        for qid, field, wrong in (
                ('done', 'calibrated', True),
                ('triage', 'temperature', advisory._TEMPERATURE),
                ('severity', 'calibrated', False),
                ('severity', 'temperature', True),
                ('severity', 'temperature', 1.0)):
            with self.subTest(qid=qid, field=field, wrong=wrong):
                modified = copy.deepcopy(pred)
                modified['answers'][qid][field] = wrong
                with self.assertRaises(ValueError):
                    advisory.validate_prediction(env, modified, self.config)
        modified = copy.deepcopy(pred)
        modified['answers']['done']['probabilities']['false'] = 0.5
        with self.assertRaises(ValueError):
            advisory.validate_prediction(env, modified, self.config)
        modified = copy.deepcopy(pred)
        modified['answers'] = dict(reversed(list(modified['answers'].items())))
        with self.assertRaises(ValueError):
            advisory.validate_prediction(env, modified, self.config)
        modified = copy.deepcopy(pred)
        modified['answers']['triage']['probabilities'] = dict(reversed(list(
            modified['answers']['triage']['probabilities'].items())))
        with self.assertRaises(ValueError):
            advisory.validate_prediction(env, modified, self.config)

    def test_preflight_one_raw_predict_cache_finally_and_drift(self):
        config = self.config
        runtime = FakeRuntime(config)
        predictor = advisory.StateJsonAdvisoryPredictor(runtime, {'fits': {}}, config)
        self.assertEqual(advisory.identity(predictor, config), advisory._FINGERPRINT)
        env = envelope()
        request = {'state': env['state'], 'questions': env['questions']}
        result = predictor.predict(request)
        self.assertEqual(result['runtime_fingerprint'], advisory._FINGERPRINT)
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(runtime.requests[0][0], request)
        self.assertEqual([qid for qid, _state, _keys in runtime.native.calls],
                         list(request['questions']))
        self.assertTrue(all(type(state) is str and state == '{"exit":1}'
                            for _qid, state, _keys in runtime.native.calls))
        self.assertEqual(runtime.native.cache, {})
        runtime.native.length = 513
        with self.assertRaises(ValueError):
            predictor.predict(request)
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(runtime.native.cache, {})
        runtime.native.length = 20
        runtime.native.truncated = True
        with self.assertRaises(ValueError):
            predictor.predict(request)
        self.assertEqual(runtime.native.cache, {})
        runtime.native.truncated = False
        runtime.spec['device'] = 'cpu'
        with self.assertRaises(ValueError):
            advisory.identity(predictor, config)
        runtime.spec['device'] = 'cuda'
        runtime.fingerprint = '0' * 64
        with self.assertRaises(ValueError):
            advisory.identity(predictor, config)
        runtime.fingerprint = config['expected_fingerprint']
        runtime.identity['environment']['python'] = 'changed'
        with self.assertRaises(ValueError):
            advisory.identity(predictor, config)
        runtime.identity = copy.deepcopy(config['expected_identity'])
        predictor.calibration['fits']['1'] = {'status': 'fitted'}
        with self.assertRaises(ValueError):
            advisory.identity(predictor, config)

    def test_offline_rejected_before_import(self):
        fake = types.ModuleType('decision_ft.state_json_native')
        with mock.patch.object(advisory, 'validate_config', return_value=self.config), \
                mock.patch.dict(sys.modules, {'decision_ft.state_json_native': fake}), \
                mock.patch.dict('os.environ', {'HF_HUB_OFFLINE': '0'}, clear=False):
            with self.assertRaisesRegex(ValueError, 'offline'):
                advisory.build_predictor(self.config)

    def test_build_uses_one_mock_native_runtime_no_real_model(self):
        config = self.config
        constructed = []

        def make_runtime(spec, *, expected_binding, expected_checkpoint_files,
                         expected_sources):
            self.assertEqual(spec, config['spec'])
            self.assertEqual(expected_binding, config['expected_binding'])
            self.assertEqual(expected_checkpoint_files, config['expected_checkpoint_files'])
            self.assertEqual(expected_sources, config['expected_sources'])
            runtime = FakeRuntime(config)
            constructed.append(runtime)
            return runtime

        fake = types.ModuleType('decision_ft.state_json_native')
        fake.StateJsonNativeRuntime = make_runtime
        offline_hub = types.ModuleType('huggingface_hub.constants')
        offline_hub.HF_HUB_OFFLINE = True
        offline_transformers = types.ModuleType('transformers.utils.hub')
        offline_transformers._is_offline_mode = True
        flags = {name: '1' for name in (
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')}
        with mock.patch.object(advisory, 'validate_config', return_value=copy.deepcopy(config)), \
                mock.patch.dict(sys.modules, {
                    'decision_ft.state_json_native': fake,
                    'huggingface_hub.constants': offline_hub,
                    'transformers.utils.hub': offline_transformers}), \
                mock.patch.dict('os.environ', flags, clear=False):
            predictor = advisory.build_predictor(config)
        self.assertEqual(len(constructed), 1)
        self.assertIs(predictor.runtime, constructed[0])
        self.assertEqual(advisory.identity(predictor, config), advisory._FINGERPRINT)
        self.assertEqual(predictor.calibration['fits']['1']['status'], 'fitted')

    def test_online_preload_rejected_before_native_import(self):
        online = types.ModuleType('huggingface_hub.constants')
        online.HF_HUB_OFFLINE = False
        fake = types.ModuleType('decision_ft.state_json_native')
        flags = {name: '1' for name in (
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')}
        with mock.patch.object(advisory, 'validate_config', return_value=self.config), \
                mock.patch.dict(sys.modules, {
                    'huggingface_hub.constants': online,
                    'decision_ft.state_json_native': fake}), \
                mock.patch.dict('os.environ', flags, clear=False):
            with self.assertRaisesRegex(ValueError, 'imported without offline'):
                advisory.build_predictor(self.config)


if __name__ == '__main__':
    unittest.main()
