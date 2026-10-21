"""Synthetic provenance and runtime checks; never loads a native model."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from jevlocal.integration import represented_native as represented
from jevlocal.integration import state_json_advisory as pure


def request():
    return {'state': {'exit': 1}, 'questions': {
        'done': {'type': 'noul', 'instructions': 'Done?'},
        'next': {'type': 'choice', 'instructions': 'Next?',
                 'criteria': {'finish': 'Finish', 'fix': 'Fix'}},
        'risk': {'type': 'score', 'instructions': 'Risk?',
                 'criteria': ['none', 'low', 'high']},
    }}


def raw_prediction(config, req):
    answers = {}
    for qid, question in req['questions'].items():
        keys = pure._option_keys(question)
        probs = {key: float(index == 0) for index, key in enumerate(keys)}
        answers[qid] = {'selected_key': keys[0], 'probabilities': probs,
                        'p_selected': 1.0, 'calibrated': False,
                        'temperature': 1.0}
    return {'runtime_fingerprint': config['expected_fingerprint'],
            'answers': answers, 'inference_total_ms': 1.0}


class FakeNative:
    def __init__(self):
        self.cache = {}
        self.calls = []
        self.invalid_at = None
        self.length = 20

    def encode(self, atom):
        self.calls.append((atom.uid, copy.deepcopy(atom.state), list(atom.keys)))
        self.cache[atom.fingerprint] = True
        return {'ids': [1] * self.length,
                'state_truncated': atom.uid == self.invalid_at}


class FakeRuntime:
    def __init__(self, config):
        self.spec = copy.deepcopy(config['spec'])
        self.identity = copy.deepcopy(config['expected_identity'])
        self.fingerprint = config['expected_fingerprint']
        self.native = FakeNative()
        self.calls = []
        self.result = raw_prediction(config, request())
        self.mutate_after = False

    def predict(self, req, calibration=None):
        self.calls.append((copy.deepcopy(req), calibration))
        if self.mutate_after:
            self.fingerprint = '0' * 64
        return copy.deepcopy(self.result)


@contextmanager
def offline_mock_runtime(fake, config):
    # Model-build tests own cached offline state as well as environment flags.
    # Earlier full-suite tests may import real HF constants in online mode.
    hub = types.ModuleType('huggingface_hub.constants')
    hub.HF_HUB_OFFLINE = True
    hub.HF_HOME = config['expected_base_cache']['hf_home']
    hub.HF_HUB_CACHE = config['expected_base_cache']['hub_cache']
    transformers_hub = types.ModuleType('transformers.utils.hub')
    transformers_hub._is_offline_mode = True
    transformers_hub.is_offline_mode = lambda: True
    transformers_hub.constants = hub
    flags = {name: '1' for name in (
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
        'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')}
    flags.update(HF_HOME=hub.HF_HOME, HF_HUB_CACHE=hub.HF_HUB_CACHE)
    with mock.patch.dict(sys.modules, {'decision_ft.state_json_native': fake,
            'huggingface_hub.constants': hub, 'transformers.utils.hub': transformers_hub}), \
            mock.patch.dict(os.environ, flags, clear=True), \
            mock.patch.object(represented, '_read_base_metadata', return_value={
                'base': config['expected_base_cache']['base'],
                'base_revision': config['expected_base_cache']['revision']}):
        yield


class RepresentedNativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        checkpoint = Path(self.tmp.name) / 'checkpoint'
        checkpoint.mkdir()
        hf_home = Path(self.tmp.name) / 'hf-home'
        hub_cache = hf_home / 'hub'
        base = 'Qwen/Qwen3.5-0.8B-Base'
        revision = 'a' * 40
        snapshot = hub_cache / 'models--Qwen--Qwen3.5-0.8B-Base' / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'model.safetensors'):
            (snapshot / name).write_bytes(
                json.dumps({'fixture': name}).encode('ascii') if name.endswith('.json')
                else name.encode('ascii'))
        base_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in snapshot.iterdir()}
        sources = {name: pure._sha(pure._module_file(name)) for name in pure._SOURCES}
        adapters = {name: pure._sha(pure._module_file(name))
                    for name in represented._ADAPTER_SOURCES}
        binding = represented._current_binding(sources)
        marker = {'schema_version': 1,
                  'status': 'experimental_requires_explicit_consumer',
                  'binding': copy.deepcopy(binding),
                  'training_freeze_sha256': 'a' * 64, 'epoch': 2}
        for name in pure._CHECKPOINT_FILES:
            path = checkpoint / name
            if name == 'state_representation.json':
                path.write_text(json.dumps(marker), encoding='utf-8')
            else:
                path.write_bytes(name.encode())
        files = pure._checkpoint_files(checkpoint)
        spec = {'backend': 'kev', 'mode': 'native',
                'purpose': 'development_evaluation', 'device': 'cpu',
                'weights_dtype': 'fp32', 'amp': 'none',
                'checkpoint': str(checkpoint)}
        identity = {
            'representation': {'binding': copy.deepcopy(binding), 'marker': marker},
            'consumer_source_sha256': sources['decision_ft.state_json_native'],
            'checkpoint_files_sha256': copy.deepcopy(files),
            'sources_sha256': copy.deepcopy(sources),
            'spec': {key: value for key, value in spec.items() if key != 'checkpoint'},
            'environment': {'python': 'synthetic-test'},
        }
        self.config = {
            'schema_version': 1, 'route': represented.ROUTE,
            'spec': spec, 'expected_identity': identity,
            'expected_fingerprint': pure._digest(identity),
            'expected_binding': binding, 'expected_sources': sources,
            'expected_checkpoint_files': files,
            'expected_base_cache': {
                'base': base, 'revision': revision,
                'hf_home': str(hf_home), 'hub_cache': str(hub_cache),
                'files_sha256': base_files},
            'expected_adapter_sources': adapters, 'calibration': None,
        }
        self.snapshot = snapshot

    def test_pure_admission_and_copy_without_torch_import(self):
        before = 'torch' in sys.modules
        checked = represented.validate_config(self.config)
        self.assertEqual(checked, self.config)
        self.assertIsNot(checked, self.config)
        checked['expected_binding']['schema_version'] = 2
        self.assertEqual(self.config['expected_binding']['schema_version'], 1)
        self.assertEqual('torch' in sys.modules, before)

    def test_rejects_schema_calibration_identity_and_binding_drift(self):
        changes = (
            lambda c: c.update(extra=True),
            lambda c: c.update(route='state_json_advisory_v1'),
            lambda c: c.update(calibration={}),
            lambda c: c['spec'].update(amp='bf16'),
            lambda c: c.update(expected_fingerprint='0' * 64),
            lambda c: c['expected_identity']['environment'].update(python='drift'),
            lambda c: c['expected_identity'].update(consumer_source_sha256='0' * 64),
            lambda c: c['expected_binding']['policy'].update(id='other'),
            lambda c: c['expected_sources'].update({'kev.model': '0' * 64}),
            lambda c: c['expected_adapter_sources'].update(
                {'jevlocal.integration.represented_native': '0' * 64}),
            lambda c: c['expected_checkpoint_files'].pop('head.pt'),
            lambda c: c.pop('expected_base_cache'),
            lambda c: c['expected_base_cache'].update(revision='latest'),
            lambda c: c['expected_base_cache']['files_sha256'].pop('config.json'),
        )
        for change in changes:
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                change(config)
                with self.assertRaises(ValueError):
                    represented.validate_config(config)

    def test_current_binding_and_marker_are_independent_of_fingerprint(self):
        config = copy.deepcopy(self.config)
        config['expected_binding']['policy']['id'] = 'other'
        config['expected_identity']['representation']['binding'] = copy.deepcopy(config['expected_binding'])
        config['expected_fingerprint'] = pure._digest(config['expected_identity'])
        with self.assertRaisesRegex(ValueError, 'binding'):
            represented.validate_config(config)
        config = copy.deepcopy(self.config)
        marker_path = Path(config['spec']['checkpoint']) / 'state_representation.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        marker['status'] = 'other'
        marker_path.write_text(json.dumps(marker), encoding='utf-8')
        config['expected_checkpoint_files']['state_representation.json'] = pure._sha(marker_path)
        config['expected_identity']['checkpoint_files_sha256'] = copy.deepcopy(config['expected_checkpoint_files'])
        config['expected_identity']['representation']['marker'] = marker
        config['expected_fingerprint'] = pure._digest(config['expected_identity'])
        with self.assertRaisesRegex(ValueError, 'marker'):
            represented.validate_config(config)

    def test_complete_inventory_and_loaded_origin(self):
        checkpoint = Path(self.config['spec']['checkpoint'])
        (checkpoint / 'extra.bin').write_bytes(b'extra')
        with mock.patch.object(pure, '_checkpoint_files') as file_reader:
            with self.assertRaisesRegex(ValueError, 'inventory'):
                represented.validate_config(self.config)
            file_reader.assert_not_called()
        (checkpoint / 'extra.bin').unlink()
        fake = types.ModuleType('jevlocal.integration.represented_native')
        fake.__file__ = str(checkpoint / 'head.pt')
        with mock.patch.dict(sys.modules, {'jevlocal.integration.represented_native': fake}):
            with self.assertRaisesRegex(ValueError, 'origin differs'):
                represented.validate_config(self.config)

    def test_offline_flags_precede_native_import(self):
        fake = types.ModuleType('decision_ft.state_json_native')
        fake.__file__ = str(pure._module_file('decision_ft.state_json_native'))
        with mock.patch.dict(sys.modules, {'decision_ft.state_json_native': fake}), \
                mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '0'}, clear=False):
            with self.assertRaisesRegex(ValueError, 'offline'):
                represented.build_predictor(self.config)
        hub = types.ModuleType('huggingface_hub.constants')
        hub.HF_HUB_OFFLINE = False
        flags = {name: '1' for name in (
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')}
        with mock.patch.dict(sys.modules, {'decision_ft.state_json_native': fake,
                                           'huggingface_hub.constants': hub}), \
                mock.patch.dict(os.environ, flags, clear=False):
            with self.assertRaisesRegex(ValueError, 'imported without offline'):
                represented.build_predictor(self.config)

    def test_base_snapshot_inventory_hash_and_index_fail_before_runtime(self):
        fake = types.ModuleType('decision_ft.state_json_native')
        fake.__file__ = str(pure._module_file('decision_ft.state_json_native'))
        fake.StateJsonNativeRuntime = mock.Mock(side_effect=lambda *a, **k: FakeRuntime(self.config))
        shard = self.snapshot / 'model.safetensors'
        with offline_mock_runtime(fake, self.config):
            shard.write_bytes(b'mutated')
            with self.assertRaises(ValueError):
                represented.build_predictor(self.config)
            shard.unlink()
            with self.assertRaises(ValueError):
                represented.build_predictor(self.config)
        fake.StateJsonNativeRuntime.assert_not_called()

        # A complete index has precisely the declared shards; unsafe names fail.
        index = self.snapshot / 'model.safetensors.index.json'
        one = self.snapshot / 'model-00001-of-00002.safetensors'
        two = self.snapshot / 'model-00002-of-00002.safetensors'
        one.write_bytes(b'one')
        two.write_bytes(b'two')
        index.write_text(json.dumps({'metadata': {}, 'weight_map': {
            'a': one.name, 'b': two.name}}), encoding='utf-8')
        self.config['expected_base_cache']['files_sha256'] = {
            p.name: pure._sha(p) for p in self.snapshot.iterdir()}
        represented.validate_config(self.config)
        with offline_mock_runtime(fake, self.config):
            represented.build_predictor(self.config)
            fake.StateJsonNativeRuntime.assert_called_once()
            fake.StateJsonNativeRuntime.reset_mock()
            two.unlink()
            with self.assertRaises(ValueError):
                represented.build_predictor(self.config)
            two.write_bytes(b'two')
            index.write_text(json.dumps({'metadata': {}, 'weight_map': {
                'a': '../escape.safetensors'}}), encoding='utf-8')
            self.config['expected_base_cache']['files_sha256'][index.name] = pure._sha(index)
            with self.assertRaises(ValueError):
                represented.build_predictor(self.config)
        fake.StateJsonNativeRuntime.assert_not_called()

    def test_cache_env_constants_and_metadata_fail_before_runtime(self):
        fake = types.ModuleType('decision_ft.state_json_native')
        fake.__file__ = str(pure._module_file('decision_ft.state_json_native'))
        fake.StateJsonNativeRuntime = mock.Mock()
        base = self.config['expected_base_cache']
        with offline_mock_runtime(fake, self.config):
            with mock.patch.object(sys.modules['transformers.utils.hub'],
                                   '_is_offline_mode', None):
                represented._check_process_cache(base)
            for key, value in (('HF_HOME', ''), ('HF_HUB_CACHE', ''),
                               ('HF_HUB_CACHE', str(self.snapshot)),
                               ('TRANSFORMERS_CACHE', str(self.snapshot))):
                with self.subTest(key=key, value=value), \
                        mock.patch.dict(os.environ, {key: value}):
                    with self.assertRaises(ValueError):
                        represented.build_predictor(self.config)
            with mock.patch.object(sys.modules['huggingface_hub.constants'],
                                   'HF_HUB_CACHE', str(self.snapshot)):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
            with mock.patch.object(sys.modules['huggingface_hub.constants'],
                                   'HF_HOME', str(self.snapshot)):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
            replacement = types.SimpleNamespace(HF_HOME=base['hf_home'],
                                                 HF_HUB_CACHE=str(self.snapshot))
            with mock.patch.object(sys.modules['transformers.utils.hub'],
                                   'constants', replacement):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
            with mock.patch.object(sys.modules['transformers.utils.hub'],
                                   'TRANSFORMERS_CACHE', str(self.snapshot), create=True):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
            with mock.patch.object(sys.modules['transformers.utils.hub'],
                                   'is_offline_mode', return_value=False):
                with self.assertRaisesRegex(ValueError, 'imported without offline'):
                    represented.build_predictor(self.config)
            with mock.patch.object(represented, '_read_base_metadata', return_value={
                    'base': 'wrong', 'base_revision': base['revision']}):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
            with mock.patch.object(represented, '_read_base_metadata', return_value={
                    'base': base['base'], 'base_revision': 'b' * 40}):
                with self.assertRaises(ValueError):
                    represented.build_predictor(self.config)
        fake.StateJsonNativeRuntime.assert_not_called()

    def test_base_json_strict_rejection(self):
        config_file = self.snapshot / 'config.json'
        config_file.write_bytes(b'{"a":1,"a":2}')
        self.config['expected_base_cache']['files_sha256']['config.json'] = pure._sha(config_file)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            represented.validate_config(self.config)

    def test_native_initialization_error_is_safe_rejection_without_retry(self):
        fake = types.ModuleType('decision_ft.state_json_native')
        fake.__file__ = str(pure._module_file('decision_ft.state_json_native'))
        fake.StateJsonNativeRuntime = mock.Mock(side_effect=RuntimeError('private checkpoint detail'))
        with offline_mock_runtime(fake, self.config):
            with self.assertRaisesRegex(ValueError, 'no fallback') as failure:
                represented.build_predictor(self.config)
            self.assertNotIn('private checkpoint detail', str(failure.exception))
            fake.StateJsonNativeRuntime.assert_called_once()

    def _predictor(self):
        runtime = FakeRuntime(self.config)
        return represented.StateJsonRepresentedPredictor(runtime, self.config), runtime

    def test_explicit_build_uses_mock_runtime_and_checks_actual_identity(self):
        config = self.config
        made = []

        def make_runtime(spec, **kwargs):
            self.assertEqual(spec, config['spec'])
            self.assertEqual(kwargs, {
                'expected_binding': config['expected_binding'],
                'expected_checkpoint_files': config['expected_checkpoint_files'],
                'expected_sources': config['expected_sources'],
            })
            runtime = FakeRuntime(config)
            made.append(runtime)
            return runtime

        fake = types.ModuleType('decision_ft.state_json_native')
        fake.__file__ = str(pure._module_file('decision_ft.state_json_native'))
        fake.StateJsonNativeRuntime = make_runtime
        with offline_mock_runtime(fake, self.config):
            predictor = represented.build_predictor(config)
            self.assertEqual(len(made), 1)
            self.assertEqual(predictor.runtime.mode, 'native')
            self.assertEqual(predictor.runtime.backend, 'kev')
            self.assertEqual(predictor.runtime.fingerprint, config['expected_fingerprint'])
            with self.assertRaises(AttributeError):
                predictor.runtime.fingerprint = '0' * 64
            made[0].identity['environment']['python'] = 'different'
            with self.assertRaisesRegex(ValueError, 'identity drift'):
                represented.identity(predictor, config)

    def test_preflight_all_atoms_then_one_raw_forward_and_cache_clear(self):
        predictor, runtime = self._predictor()
        req = request()
        result = predictor.predict(req)
        self.assertEqual(result, raw_prediction(self.config, req))
        self.assertEqual(runtime.calls, [(req, None)])
        self.assertEqual([uid for uid, _, _ in runtime.native.calls], list(req['questions']))
        self.assertTrue(all(state == '{"exit":1}' for _, state, _ in runtime.native.calls))
        self.assertEqual(runtime.native.cache, {})
        runtime.native.invalid_at = 'risk'
        with self.assertRaisesRegex(ValueError, 'truncated'):
            predictor.predict(req)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(runtime.native.calls), 6)
        self.assertEqual(runtime.native.cache, {})
        runtime.native.invalid_at = None
        runtime.native.length = 513
        with self.assertRaisesRegex(ValueError, 'budget'):
            predictor.predict(req)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.native.cache, {})

    def test_raw_reply_policy_and_before_after_drift(self):
        predictor, runtime = self._predictor()
        req = request()
        for qid, field, value in (
            ('risk', 'calibrated', True), ('done', 'temperature', 0.5),
            ('next', 'temperature', True)):
            with self.subTest(qid=qid, field=field):
                runtime.result = raw_prediction(self.config, req)
                runtime.result['answers'][qid][field] = value
                with self.assertRaisesRegex(ValueError, 'raw prediction'):
                    predictor.predict(req)
                self.assertEqual(runtime.native.cache, {})
        runtime.result = raw_prediction(self.config, req)
        runtime.result['answers'] = dict(reversed(list(runtime.result['answers'].items())))
        with self.assertRaisesRegex(ValueError, 'question order'):
            predictor.predict(req)
        runtime.result = raw_prediction(self.config, req)
        runtime.result['answers']['next']['probabilities'] = dict(reversed(list(
            runtime.result['answers']['next']['probabilities'].items())))
        with self.assertRaisesRegex(ValueError, 'candidate order'):
            predictor.predict(req)
        runtime.result = raw_prediction(self.config, req)
        runtime.mutate_after = True
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            predictor.predict(req)
        self.assertEqual(runtime.native.cache, {})
        runtime.mutate_after = False
        runtime.fingerprint = self.config['expected_fingerprint']
        runtime.spec['device'] = 'cuda'
        before = len(runtime.calls)
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            predictor.predict(req)
        self.assertEqual(len(runtime.calls), before)
        self.assertEqual(runtime.native.cache, {})

    def test_native_prediction_failure_clears_cache_and_does_not_retry(self):
        predictor, runtime = self._predictor()
        runtime.predict = mock.Mock(side_effect=RuntimeError('private request detail'))
        with self.assertRaisesRegex(ValueError, 'no fallback') as failure:
            predictor.predict(request())
        self.assertNotIn('private request detail', str(failure.exception))
        runtime.predict.assert_called_once()
        self.assertEqual(runtime.native.cache, {})

    def test_metadata_is_defensive_and_raw(self):
        predictor, _ = self._predictor()
        self.assertIsNone(predictor.calibration)
        data = predictor.metadata
        self.assertEqual(data['kind'], represented.ROUTE)
        self.assertFalse(data['calibrated'])
        self.assertEqual(data['semantic_quality'], 'unqualified')
        self.assertEqual(data['adapter_source_sha256'], self.config['expected_adapter_sources'])
        self.assertEqual(data['expected_base_cache'], self.config['expected_base_cache'])
        data['binding']['schema_version'] = 2
        self.assertEqual(predictor.metadata['binding']['schema_version'], 1)


if __name__ == '__main__':
    unittest.main()
