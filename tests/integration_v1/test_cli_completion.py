from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_input(root, *, status='executed', exit_code=0, run_id='current'):
    now = datetime.now(timezone.utc)
    stamp = lambda n: (now + timedelta(seconds=n)).isoformat()
    subject = root / 'source.py'
    subject.write_text('print("example")\n', encoding='utf-8')
    receipt = {'schema_version': 1, 'run_id': run_id,
        'subject_files': [{'path': str(subject), 'sha256': hashlib.sha256(subject.read_bytes()).hexdigest()}],
        'started_at': stamp(-8), 'ended_at': stamp(-7), 'status': status,
        'command': ['python', '-m', 'unittest'], 'exit_code': exit_code,
        'tests': {'passed': 1, 'failed': int(exit_code != 0), 'errors': 0, 'skipped': 0},
        'output': 'synthetic fixture result; no command execution claimed by this unit test'}
    evidence = root / 'receipt.json'
    digest = write_json(evidence, receipt)
    args = {'schema_version': 1, 'workspace': str(root), 'run_id': 'current', 'started_at': stamp(-10),
        'claims': [{'id': 'claim', 'text': 'The stated test suite passed.', 'kind': 'tests_passed', 'evidence_ids': ['test']}],
        'evidence': [{'id': 'test', 'kind': 'test_run', 'path': str(evidence), 'sha256': digest}]}
    path = root / 'input.json'
    write_json(path, args)
    return path, args


class CompletionCliTests(unittest.TestCase):
    def run_cli(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_no_native_is_unknown_and_does_not_execute_receipt_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, args = make_input(root)
            before = sorted(p.name for p in root.iterdir())
            with patch('jevlocal.integration.native_completion.build_native_completion_judge') as build:
                code, result = self.run_cli(['verify-completion', '--input', str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(result['claims'][0]['status'], 'unknown')
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['action_authorized'])
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)
            build.assert_not_called()

    def test_explicit_native_is_lazy_and_output_is_new_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, args = make_input(root)
            spec, target = root / 'runtime.json', root / 'report.json'
            write_json(spec, {'fixture': True})
            seen = []
            def judge(items):
                seen.extend(items)
                return {'decisions': {x['id']: 'supported' for x in items},
                        'route': {'kind': 'fixture_only', 'semantic_quality': 'unqualified'}}
            with patch('jevlocal.integration.native_completion.build_native_completion_judge', return_value=judge) as build:
                code, result = self.run_cli(['verify-completion','--input',str(path),'--native-runtime',str(spec),'--output',str(target)])
            self.assertEqual(code, 0)
            build.assert_called_once_with({'fixture': True})
            self.assertEqual(result['claims'][0]['status'], 'supported')
            self.assertEqual(seen[0]['id'], 'claim')
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), result)
            prior = target.read_bytes()
            with patch('jevlocal.integration.native_completion.build_native_completion_judge', return_value=judge):
                code, rejected = self.run_cli(['verify-completion','--input',str(path),'--native-runtime',str(spec),'--output',str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(rejected['status'], 'rejected')
            self.assertEqual(target.read_bytes(), prior)

    def test_bad_hash_never_loads_model_or_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, args = make_input(root)
            args['evidence'][0]['sha256'] = '0' * 64
            write_json(path, args)
            target = root / 'report.json'
            with patch('jevlocal.integration.native_completion.build_native_completion_judge') as build:
                code, result = self.run_cli(['verify-completion','--input',str(path),'--native-runtime',str(root/'missing-runtime.json'),'--output',str(target)])
            self.assertEqual(code, 2)
            build.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(result['status'], 'rejected')

    def test_stale_receipt_does_not_load_even_explicit_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, args = make_input(root, run_id='previous')
            with patch('jevlocal.integration.native_completion.build_native_completion_judge') as build:
                code, result = self.run_cli(['verify-completion','--input',str(path),'--native-runtime',str(root/'missing-runtime.json')])
            self.assertEqual(code, 0)
            self.assertEqual(result['claims'][0]['status'], 'unknown')
            build.assert_not_called()

    def test_current_failure_contradicts_test_success_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, args = make_input(root, status='failed', exit_code=1)
            code, result = self.run_cli(['verify-completion','--input',str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(result['claims'][0]['status'], 'contradicted')


    def test_raw_input_file_budget_before_json_parse_or_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'oversized.json'
            path.write_text(' ' * (128 * 1024 + 1), encoding='utf-8')
            with patch('jevlocal.integration.native_completion.build_native_completion_judge') as build:
                code, result = self.run_cli(['verify-completion', '--input', str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'rejected')
            build.assert_not_called()


    def test_state_json_flag_selects_only_explicit_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = make_input(root)
            config = root / 'represented.json'
            write_json(config, {'route': 'fixture raw route'})
            def judge(items):
                return {'decisions': {item['id']: 'supported' for item in items},
                        'route': {'kind': 'fixture_only', 'calibrated': False,
                                  'semantic_quality': 'unqualified'}}
            with patch('jevlocal.integration.native_completion.build_state_json_completion_judge',
                       return_value=judge) as represented, \
                 patch('jevlocal.integration.native_completion.build_native_completion_judge') as generic:
                code, result = self.run_cli(['verify-completion', '--input', str(source),
                    '--state-json-runtime', str(config)])
            self.assertEqual(code, 0)
            represented.assert_called_once_with({'route': 'fixture raw route'})
            generic.assert_not_called()
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['action_authorized'])

    def test_state_json_ineligible_and_invalid_inputs_do_not_read_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = make_input(root, run_id="previous")
            with patch('jevlocal.integration.native_completion.build_state_json_completion_judge') as build:
                code, result = self.run_cli(['verify-completion', '--input', str(source),
                    '--state-json-runtime', str(root / 'missing-config.json')])
            self.assertEqual(code, 0)
            self.assertEqual(result['claims'][0]['status'], 'unknown')
            build.assert_not_called()
            source, value = make_input(root)
            value['evidence'][0]['sha256'] = '0' * 64
            write_json(source, value)
            with patch('jevlocal.integration.native_completion.build_state_json_completion_judge') as build:
                code, result = self.run_cli(['verify-completion', '--input', str(source),
                    '--state-json-runtime', str(root / 'missing-config.json')])
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'rejected')
            build.assert_not_called()

    def test_state_json_failure_and_oversized_config_do_not_fallback_or_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = make_input(root)
            config, output = root / 'config.json', root / 'report.json'
            write_json(config, {'route': 'fixture'})
            with patch('jevlocal.integration.native_completion.build_state_json_completion_judge',
                       side_effect=ValueError('fixture admission failure')) as represented, \
                 patch('jevlocal.integration.native_completion.build_native_completion_judge') as generic:
                code, result = self.run_cli(['verify-completion', '--input', str(source),
                    '--state-json-runtime', str(config), '--output', str(output)])
            self.assertEqual(code, 2)
            represented.assert_called_once()
            generic.assert_not_called()
            self.assertFalse(output.exists())
            config.write_text(' ' * (64 * 1024 + 1), encoding='utf-8')
            with patch('jevlocal.integration.native_completion.build_state_json_completion_judge') as represented:
                code, result = self.run_cli(['verify-completion', '--input', str(source),
                    '--state-json-runtime', str(config), '--output', str(output)])
            self.assertEqual(code, 2)
            represented.assert_not_called()
            self.assertFalse(output.exists())

    def test_native_routes_are_mutually_exclusive_before_input_read(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(['verify-completion', '--input', 'missing-input.json',
                  '--native-runtime', 'legacy.json', '--state-json-runtime', 'represented.json'])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
