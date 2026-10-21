from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, with_contract=True):
    baseline = root / 'before.py'
    current = root / 'module.py'
    baseline.write_text('allow = False\n', encoding='utf-8')
    current.write_text('allow = True\n', encoding='utf-8')
    contract = root / 'policy.txt'
    contract.write_text('Permission changes require explicit human review. A model is not approval.\n', encoding='utf-8')
    value = {'schema_version': 1, 'workspace': str(root), 'request_id': 'review-cli',
        'changes': [{'path': 'module.py',
                     'before': {'path': str(baseline), 'sha256': sha(baseline)},
                     'after': {'path': str(current), 'sha256': sha(current)}}],
        'contracts': [{'id': 'approval', 'path': str(contract), 'sha256': sha(contract),
                       'applies_to': ['module.py'], 'category': 'permissions',
                       'mandatory_review': True, 'reviewer': 'human'}] if with_contract else []}
    source = root / 'review.json'
    write_json(source, value)
    return source, value


class ReviewCliTests(unittest.TestCase):
    def call(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_no_native_review_is_unknown_never_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            before = (root / 'module.py').read_bytes()
            with patch('jevlocal.integration.native_review.build_native_review_judge') as build:
                code, result = self.call(['review-diff', '--input', str(source)])
            self.assertEqual(code, 0)
            self.assertEqual(result['status'], 'reviewed')
            hunk = result['changes'][0]['hunks'][0]
            self.assertEqual(hunk['status'], 'unknown')
            self.assertIsNone(hunk['risk_score'])
            self.assertTrue(hunk['human_review_required'])
            self.assertTrue(hunk['review_required'])
            for key in ('action_authorized', 'execution_authorized', 'merge_authorized', 'review_complete'):
                self.assertFalse(result[key])
            self.assertEqual(before, (root / 'module.py').read_bytes())
            build.assert_not_called()

    def test_low_native_fixture_cannot_cancel_human_floor_and_new_output_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            runtime, output = root / 'runtime.json', root / 'report.json'
            write_json(runtime, {'fixture': 'not a native model'})
            seen = []
            def judge(items):
                seen.extend(items)
                return {'decisions': {item['id']: {'risk_score': 0, 'categories': dict.fromkeys(
                    ('security', 'permissions', 'interface', 'persistence'), False)} for item in items},
                    'route': {'kind': 'fixture_only', 'semantic_quality': 'unqualified'}}
            with patch('jevlocal.integration.native_review.build_native_review_judge', return_value=judge) as build:
                code, result = self.call(['review-diff','--input',str(source),'--native-runtime',str(runtime),'--output',str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]['file_path'], 'module.py')
            hunk = result['changes'][0]['hunks'][0]
            self.assertEqual(hunk['risk_score'], 0)
            self.assertTrue(hunk['human_review_required'])
            self.assertFalse(result['merge_authorized'])
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            original = output.read_bytes()
            with patch('jevlocal.integration.native_review.build_native_review_judge') as build:
                code, rejection = self.call(['review-diff','--input',str(source),'--native-runtime',str(runtime),'--output',str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(rejection['status'], 'rejected')
            self.assertEqual(output.read_bytes(), original)
            build.assert_not_called()

    def test_missing_contract_does_not_load_even_explicit_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root, with_contract=False)
            with patch('jevlocal.integration.native_review.build_native_review_judge') as build:
                code, result = self.call(['review-diff','--input',str(source),'--native-runtime',str(root/'missing-runtime.json')])
            self.assertEqual(code, 0)
            self.assertEqual(result['changes'][0]['hunks'][0]['status'], 'unknown')
            self.assertTrue(result['changes'][0]['hunks'][0]['review_required'])
            build.assert_not_called()

    def test_modified_current_file_rejected_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            (root / 'module.py').write_text('another change\n', encoding='utf-8')
            output = root / 'report.json'
            with patch('jevlocal.integration.native_review.build_native_review_judge') as build:
                code, result = self.call(['review-diff','--input',str(source),'--native-runtime',str(root/'missing-runtime.json'),'--output',str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'rejected')
            self.assertFalse(output.exists())
            build.assert_not_called()

    def test_duplicate_keys_and_oversized_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            for raw in ('{"schema_version":1,"schema_version":1}', ' '*(128*1024+1)):
                source.write_text(raw, encoding='utf-8')
                with patch('jevlocal.integration.native_review.build_native_review_judge') as build:
                    code, result = self.call(['review-diff','--input',str(source)])
                self.assertEqual(code, 2)
                self.assertEqual(result['status'], 'rejected')
                build.assert_not_called()


    def test_state_json_flag_selects_only_explicit_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            config = root / 'represented.json'
            write_json(config, {'route': 'fixture raw route'})
            def judge(items):
                return {'decisions': {item['id']: {'risk_score': 0, 'categories': dict.fromkeys(('security', 'permissions', 'interface', 'persistence'), False)} for item in items},
                        'route': {'kind': 'fixture_only', 'calibrated': False,
                                  'semantic_quality': 'unqualified'}}
            with patch('jevlocal.integration.native_review.build_state_json_review_judge',
                       return_value=judge) as represented, \
                 patch('jevlocal.integration.native_review.build_native_review_judge') as generic:
                code, result = self.call(['review-diff', '--input', str(source),
                    '--state-json-runtime', str(config)])
            self.assertEqual(code, 0)
            represented.assert_called_once_with({'route': 'fixture raw route'})
            generic.assert_not_called()
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['action_authorized'])

    def test_state_json_ineligible_and_invalid_inputs_do_not_read_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root, with_contract=False)
            with patch('jevlocal.integration.native_review.build_state_json_review_judge') as build:
                code, result = self.call(['review-diff', '--input', str(source),
                    '--state-json-runtime', str(root / 'missing-config.json')])
            self.assertEqual(code, 0)
            self.assertEqual(result['changes'][0]['hunks'][0]['status'], 'unknown')
            build.assert_not_called()
            source, value = fixture(root)
            value['changes'][0]['after']['sha256'] = '0' * 64
            write_json(source, value)
            with patch('jevlocal.integration.native_review.build_state_json_review_judge') as build:
                code, result = self.call(['review-diff', '--input', str(source),
                    '--state-json-runtime', str(root / 'missing-config.json')])
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'rejected')
            build.assert_not_called()

    def test_state_json_failure_and_oversized_config_do_not_fallback_or_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, value = fixture(root)
            config, output = root / 'config.json', root / 'report.json'
            write_json(config, {'route': 'fixture'})
            with patch('jevlocal.integration.native_review.build_state_json_review_judge',
                       side_effect=ValueError('fixture admission failure')) as represented, \
                 patch('jevlocal.integration.native_review.build_native_review_judge') as generic:
                code, result = self.call(['review-diff', '--input', str(source),
                    '--state-json-runtime', str(config), '--output', str(output)])
            self.assertEqual(code, 2)
            represented.assert_called_once()
            generic.assert_not_called()
            self.assertFalse(output.exists())
            config.write_text(' ' * (64 * 1024 + 1), encoding='utf-8')
            with patch('jevlocal.integration.native_review.build_state_json_review_judge') as represented:
                code, result = self.call(['review-diff', '--input', str(source),
                    '--state-json-runtime', str(config), '--output', str(output)])
            self.assertEqual(code, 2)
            represented.assert_not_called()
            self.assertFalse(output.exists())

    def test_native_routes_are_mutually_exclusive_before_input_read(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(['review-diff', '--input', 'missing-input.json',
                  '--native-runtime', 'legacy.json', '--state-json-runtime', 'represented.json'])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
