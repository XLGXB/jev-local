"""Explicit local filesystem CLI checks; all files and predictors are fixtures."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    repository = root / 'repository'
    repository.mkdir()
    (repository / 'src').mkdir()
    (repository / 'src' / 'module.py').write_text('PUBLIC_FIXTURE = True\n', encoding='utf-8')
    (repository / 'README.md').write_text('Public navigation fixture.\n', encoding='utf-8')
    (repository / '.env').write_text('SYNTHETIC_CREDENTIAL_DENIAL_FIXTURE_ONLY\n', encoding='utf-8')
    value = {'schema_version': 1, 'request_id': 'navigation-cli-fixture',
             'repository_root': repository.as_posix(), 'current_directory': '.',
             'query': 'Find the source module.', 'candidate_names': None, 'max_hops': 3}
    source = root / 'input.json'
    write_json(source, value)
    return source, value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


class FixturePredictor:
    def __init__(self, abstain=False):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.requests = []
        self.abstain = abstain

    def predict(self, request):
        self.requests.append(deepcopy(request))
        candidates = request['state']['candidates']
        choice = '__abstain__' if self.abstain else next(
            item['id'] for item in candidates if item['name'] in ('src', 'module.py'))
        question = next(iter(request['questions']))
        keys = request['questions'][question]['criteria']
        return {'runtime_fingerprint': self.runtime.fingerprint, 'inference_total_ms': 1.0,
                'answers': {question: {'selected_key': choice,
                    'probabilities': {key: float(key == choice) for key in keys},
                    'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0}}}


class RepositoryNavigationCliTests(unittest.TestCase):
    def call(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_actual_directory_metadata_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            output = root / 'report.json'
            code, result = self.call(['navigate-repository', '--input', str(source), '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['reason'], 'semantic_unavailable')
            self.assertEqual(result['request_sha256'], digest(value))
            self.assertEqual({item['name'] for item in result['hops'][0]['candidates']}, {'src', 'README.md'})
            self.assertEqual(result['hops'][0]['excluded_count'], 1)
            self.assertEqual(result['repository_root'], value['repository_root'])
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            self.assertIsNone(result['selected_file'])
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['file_write_authorized'])
            self.assertFalse(result['deletion_authorized'])
            self.assertEqual(result['path_concurrency_guarantee'],
                             'windows_handle_relative_no_reparse' if os.name == 'nt' else
                             'stable_snapshot_not_handle_anchored')

    def test_real_adapter_two_hops_and_both_explicit_builders(self):
        from jevlocal.integration.native_repository_navigation import NativeRepositoryNavigationJudge
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            config = root / 'runtime.json'
            write_json(config, {'fixture': 'no real model'})
            for flag, builder in (('--native-runtime', 'build_native_repository_navigation_judge'),
                                  ('--state-json-runtime', 'build_state_json_repository_navigation_judge')):
                predictor = FixturePredictor()
                judge = NativeRepositoryNavigationJudge(predictor)
                with patch('jevlocal.integration.native_repository_navigation.' + builder, return_value=judge) as build:
                    code, result = self.call(['navigate-repository', '--input', str(source), flag, str(config)])
                self.assertEqual(code, 0)
                build.assert_called_once_with({'fixture': 'no real model'})
                self.assertEqual(result['status'], 'file_selected')
                self.assertEqual(result['selected_file'], 'src/module.py')
                self.assertEqual(len(predictor.requests), 2)
                self.assertEqual([hop['current_directory'] for hop in result['hops']], ['.', 'src'])
                for index, (hop, semantic) in enumerate(zip(result['hops'], predictor.requests), 1):
                    self.assertNotIn('repository_root', semantic['state'])
                    self.assertNotIn('snapshot_sha256', semantic['state'])
                    projected = [{key: item[key] for key in ('id', 'name', 'relative_path', 'kind')}
                                 for item in hop['candidates']]
                    prepared = {'request_id': value['request_id'], 'hop_index': index,
                                'query': value['query'], 'current_directory': hop['current_directory'],
                                'candidates': projected, 'snapshot_sha256': hop['snapshot_sha256']}
                    binding = hop['route']['input_bindings'][0]
                    self.assertEqual(binding['prepared_item_sha256'], digest(prepared))
                    self.assertEqual(binding['semantic_request_sha256'], digest(semantic))
                    self.assertIsNotNone(hop['path_evidence']['selected_candidate_identity'])
                self.assertEqual(result['semantic_quality'], 'unqualified')
                self.assertTrue(result['main_model_review_required'])
                self.assertFalse(result['execution_authorized'])

    def test_abstention_does_not_descend(self):
        from jevlocal.integration.native_repository_navigation import NativeRepositoryNavigationJudge
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            config = root / 'runtime.json'
            write_json(config, {'fixture': True})
            predictor = FixturePredictor(abstain=True)
            with patch('jevlocal.integration.native_repository_navigation.build_native_repository_navigation_judge',
                       return_value=NativeRepositoryNavigationJudge(predictor)):
                code, result = self.call(['navigate-repository', '--input', str(source), '--native-runtime', str(config)])
            self.assertEqual(code, 0)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(len(result['hops']), 1)
            self.assertEqual(len(predictor.requests), 1)
            self.assertIsNone(result['selected_file'])

    def test_invalid_or_credential_path_rejected_before_factory_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            output = root / 'report.json'
            for update in ({'current_directory': '../outside'}, {'candidate_names': ['.env']},
                           {'max_hops': True}, {'execution_authorized': True}):
                altered = {**value, **update}
                write_json(source, altered)
                code, result = self.call(['navigate-repository', '--input', str(source),
                    '--native-runtime', str(root / 'missing.json'), '--output', str(output)])
                self.assertEqual(code, 2)
                self.assertEqual(result['status'], 'rejected')
                self.assertFalse(output.exists())

    def test_overflow_never_reads_config_or_silently_truncates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            repository = Path(value['repository_root'])
            for index in range(16):
                (repository / ('file%02d.txt' % index)).touch()
            code, result = self.call(['navigate-repository', '--input', str(source),
                '--state-json-runtime', str(root / 'missing.json')])
            self.assertEqual(code, 0)
            self.assertEqual(result['status'], 'requires_candidate_subset')
            self.assertIsNone(result['selected_file'])
            self.assertIsNone(result['hops'][0]['selected_candidate_id'])

    def test_existing_report_and_input_size_or_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            output = root / 'report.json'
            output.write_text('retain me\n', encoding='utf-8')
            code, result = self.call(['navigate-repository', '--input', str(source),
                '--native-runtime', str(root / 'missing.json'), '--output', str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding='utf-8'), 'retain me\n')
            for raw in ('{"schema_version":1,"schema_version":1}', ' ' * (64 * 1024 + 1)):
                source.write_text(raw, encoding='utf-8')
                code, result = self.call(['navigate-repository', '--input', str(source)])
                self.assertEqual(code, 2)
                self.assertEqual(result['status'], 'rejected')

    def test_runtime_config_bounds_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            config, output = root / 'runtime.json', root / 'report.json'
            for flag in ('--native-runtime', '--state-json-runtime'):
                for oversize in (False, True):
                    config.write_text(' ' * (64 * 1024 + 1) if oversize else '{}', encoding='utf-8')
                    with patch('jevlocal.integration.native_repository_navigation.build_native_repository_navigation_judge',
                               side_effect=ValueError('private fixture error')) as generic, \
                         patch('jevlocal.integration.native_repository_navigation.build_state_json_repository_navigation_judge',
                               side_effect=ValueError('private fixture error')) as represented:
                        code, result = self.call(['navigate-repository', '--input', str(source), flag,
                                                  str(config), '--output', str(output)])
                    self.assertEqual(code, 2)
                    self.assertEqual(result, {'status': 'rejected', 'error_type': 'ValueError', 'execution_authorized': False})
                    self.assertFalse(output.exists())
                    self.assertEqual(generic.call_count, int(not oversize and flag == '--native-runtime'))
                    self.assertEqual(represented.call_count, int(not oversize and flag == '--state-json-runtime'))

    def test_mutually_exclusive_routes_before_input_read(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(['navigate-repository', '--input', 'missing.json',
                  '--native-runtime', 'generic.json', '--state-json-runtime', 'represented.json'])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
