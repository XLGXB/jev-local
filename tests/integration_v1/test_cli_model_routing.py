"""CPU-only CLI and real adapter interoperability; all model data is fictitious."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main
from jevlocal.integration.native_model_routing import NativeModelRoutingJudge


def request():
    return {
        'schema_version': 1, 'request_id': 'public-fixture-routing',
        'current_primary': {'model_id': 'fixture-primary', 'reasoning_effort': 'high'},
        'task': {'summary': 'Review a public example interface.', 'difficulty': 'complex',
                 'risk': 'medium', 'context_tokens': 1000, 'required_tools': ['read_file']},
        'budget': {'max_estimated_cost': 4, 'max_latency_ms': 3000,
                   'cost_unit': 'fixture-units-per-task'},
        'catalog': {'source': 'Fictitious public CPU test catalog, not current host metadata',
                    'observed_at': '2026-09-26T00:00:00Z',
                    'models': [
                        {'model_id': 'fixture-primary', 'reasoning_efforts': ['high'],
                         'description': 'Fictitious primary model', 'context_window_tokens': 8000,
                         'tools': ['read_file'], 'estimated_cost': 3, 'estimated_latency_ms': 2000},
                        {'model_id': 'fixture-small', 'reasoning_efforts': ['medium'],
                         'description': 'Fictitious alternative with incomplete metadata',
                         'context_window_tokens': None, 'tools': None,
                         'estimated_cost': None, 'estimated_latency_ms': None}]},
        'profiles': [{'id': 'primary', 'model_id': 'fixture-primary', 'reasoning_effort': 'high'},
                     {'id': 'small', 'model_id': 'fixture-small', 'reasoning_effort': 'medium'}],
    }


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


class FixturePredictor:
    def __init__(self, selected='small'):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.selected = selected
        self.requests = []

    def predict(self, value):
        self.requests.append(deepcopy(value))
        keys = value['questions']['profile']['criteria']
        return {'runtime_fingerprint': self.runtime.fingerprint, 'inference_total_ms': 1.0,
                'answers': {'profile': {'selected_key': self.selected,
                    'probabilities': {key: float(key == self.selected) for key in keys},
                    'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0}}}


class ModelRoutingCliTests(unittest.TestCase):
    def call(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_metadata_only_unknown_and_new_atomic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.json', root / 'report.json'
            value = request()
            write_json(source, value)
            before = source.read_bytes()
            with patch('jevlocal.integration.native_model_routing.build_native_model_routing_judge') as generic, \
                 patch('jevlocal.integration.native_model_routing.build_state_json_model_routing_judge') as represented:
                code, result = self.call(['advise-model-route', '--input', str(source), '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(result['status'], 'unknown')
            self.assertIn('semantic_unavailable', result['reasons'])
            self.assertIsNone(result['recommendation'])
            self.assertEqual(result['request_sha256'], digest(value))
            self.assertEqual(result['catalog_sha256'], digest(value['catalog']))
            self.assertEqual(result['catalog_provenance'], 'caller_supplied_not_host_authenticated')
            self.assertEqual(result['catalog_freshness'], 'not_verified')
            self.assertEqual(result['budget']['cost_unit'], 'fixture-units-per-task')
            self.assertEqual(set(result['profiles'][1]['unknowns']), {
                'context_capacity_unknown', 'tool_support_unknown', 'cost_unknown', 'latency_unknown'})
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            self.assertEqual(source.read_bytes(), before)
            generic.assert_not_called()
            represented.assert_not_called()
            self.assertTrue(result['main_model_review_required'])
            for key in ('execution_authorized', 'model_switch_authorized',
                        'provider_change_authorized', 'current_primary_changed', 'calibrated'):
                self.assertFalse(result[key])

    def test_cli_core_real_adapter_fake_predictor_binding_and_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = root / 'input.json', root / 'runtime.json'
            value = request()
            write_json(source, value)
            write_json(config, {'fixture': 'CPU-only'})
            for flag, builder in (('--native-runtime', 'build_native_model_routing_judge'),
                                  ('--state-json-runtime', 'build_state_json_model_routing_judge')):
                for selected in ('small', '__abstain__'):
                    with self.subTest(flag=flag, selected=selected):
                        predictor = FixturePredictor(selected)
                        judge = NativeModelRoutingJudge(predictor)
                        with patch('jevlocal.integration.native_model_routing.' + builder, return_value=judge) as build:
                            code, result = self.call(['advise-model-route', '--input', str(source), flag, str(config)])
                        self.assertEqual(code, 0)
                        build.assert_called_once_with({'fixture': 'CPU-only'})
                        self.assertEqual(len(predictor.requests), 1)
                        semantic = predictor.requests[0]
                        self.assertEqual(set(semantic['questions']['profile']['criteria']), {'primary', 'small', '__abstain__'})
                        self.assertEqual(semantic['state']['task'], value['task'])
                        self.assertEqual(semantic['state']['budget'], value['budget'])
                        binding = result['route']['input_bindings'][0]
                        prepared = {'request_id': value['request_id'], 'request_sha256': digest(value),
                                    'catalog_sha256': digest(value['catalog']), **semantic['state']}
                        self.assertEqual(binding['prepared_item_sha256'], digest(prepared))
                        self.assertEqual(binding['semantic_request_sha256'], digest(semantic))
                        self.assertEqual(result['current_primary'], value['current_primary'])
                        self.assertFalse(result['current_primary_changed'])
                        self.assertFalse(result['execution_authorized'])
                        self.assertFalse(result['model_switch_authorized'])
                        self.assertFalse(result['provider_change_authorized'])
                        self.assertEqual(result['semantic_quality'], 'unqualified')
                        self.assertEqual(result['status'], 'unknown' if selected == '__abstain__' else 'advisory')
                        self.assertEqual(result['selected_profile_id'], None if selected == '__abstain__' else 'small')

    def test_invalid_requests_rejected_before_config_or_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.json', root / 'report.json'
            for mutate in (lambda x: x['current_primary'].update(reasoning_effort='ultra'),
                           lambda x: x['budget'].update(max_estimated_cost=True),
                           lambda x: x.update(execution_authorized=True)):
                value = request()
                mutate(value)
                write_json(source, value)
                with patch('jevlocal.integration.native_model_routing.build_native_model_routing_judge') as build:
                    code, result = self.call(['advise-model-route', '--input', str(source),
                        '--native-runtime', str(root / 'missing.json'), '--output', str(output)])
                self.assertEqual(code, 2)
                self.assertEqual(result['status'], 'rejected')
                build.assert_not_called()
                self.assertFalse(output.exists())

    def test_no_eligible_profiles_never_read_either_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            value = request()
            for model in value['catalog']['models']:
                model['context_window_tokens'] = 100
            write_json(source, value)
            for flag in ('--native-runtime', '--state-json-runtime'):
                code, result = self.call(['advise-model-route', '--input', str(source), flag, str(root / 'missing.json')])
                self.assertEqual(code, 0)
                self.assertIn('no_compatible_profile', result['reasons'])
                self.assertFalse(any(profile['eligible'] for profile in result['profiles']))

    def test_existing_output_preserved_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.json', root / 'report.json'
            write_json(source, request())
            original = b'preserve-existing-output\n'
            output.write_bytes(original)
            with patch('jevlocal.integration.native_model_routing.build_native_model_routing_judge') as build:
                code, result = self.call(['advise-model-route', '--input', str(source),
                    '--native-runtime', str(root / 'missing.json'), '--output', str(output)])
            self.assertEqual(code, 2)
            build.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_duplicate_keys_and_oversized_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            for raw in ('{"schema_version":1,"schema_version":1}', ' ' * (128 * 1024 + 1)):
                source.write_text(raw, encoding='utf-8')
                code, result = self.call(['advise-model-route', '--input', str(source)])
                self.assertEqual(code, 2)
                self.assertEqual(result['status'], 'rejected')

    def test_both_configs_bounded_and_failure_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / 'input.json', root / 'runtime.json', root / 'report.json'
            write_json(source, request())
            for flag, selected in (('--native-runtime', 'build_native_model_routing_judge'),
                                   ('--state-json-runtime', 'build_state_json_model_routing_judge')):
                for oversized in (False, True):
                    if oversized:
                        config.write_text(' ' * (64 * 1024 + 1), encoding='utf-8')
                    else:
                        write_json(config, {'fixture': 'not a model'})
                    with patch('jevlocal.integration.native_model_routing.build_native_model_routing_judge',
                               side_effect=ValueError('private fixture error')) as generic, \
                         patch('jevlocal.integration.native_model_routing.build_state_json_model_routing_judge',
                               side_effect=ValueError('private fixture error')) as represented:
                        code, result = self.call(['advise-model-route', '--input', str(source), flag,
                                                  str(config), '--output', str(output)])
                    self.assertEqual(code, 2)
                    self.assertEqual(result, {'status': 'rejected', 'error_type': 'ValueError', 'execution_authorized': False})
                    self.assertFalse(output.exists())
                    self.assertEqual(generic.call_count, int(not oversized and selected == 'build_native_model_routing_judge'))
                    self.assertEqual(represented.call_count, int(not oversized and selected == 'build_state_json_model_routing_judge'))

    def test_routes_mutually_exclusive_before_input_read(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(['advise-model-route', '--input', 'missing.json',
                  '--native-runtime', 'generic.json', '--state-json-runtime', 'represented.json'])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
