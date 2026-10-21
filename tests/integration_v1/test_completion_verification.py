"""Focused F05 evidence guards; all receipts are caller-supplied test fixtures."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration import completion_verification as core
from jevlocal.integration.completion_verification import verify_completion


def stamp(offset=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.subject = self.workspace / 'source.py'
        self.subject.write_text('print(1)\n', encoding='utf-8')
        self.calls = []
        self.run_id = 'run-1'
        self.window = stamp(-120)

    def add_evidence(self, evidence, kind, receipt):
        path = self.workspace / (kind + '-' + str(len(evidence)) + '.json')
        path.write_text(json.dumps(receipt), encoding='utf-8')
        item = {'id': 'e' + str(len(evidence)), 'kind': kind,
                'path': str(path), 'sha256': digest(path)}
        evidence.append(item)
        return item['id']

    def subject_binding(self):
        return {'path': str(self.subject), 'sha256': digest(self.subject)}

    def make_test_receipt(self, **changes):
        item = {'schema_version': 1, 'run_id': self.run_id,
                'subject_files': [self.subject_binding()],
                'started_at': stamp(-30), 'ended_at': stamp(-20),
                'status': 'executed', 'command': ['python', '-m', 'unittest'],
                'exit_code': 0, 'tests': {'passed': 1, 'failed': 0,
                                          'errors': 0, 'skipped': 0},
                'output': 'OK'}
        item.update(changes)
        return item

    def diff_receipt(self, **changes):
        item = {'schema_version': 1, 'run_id': self.run_id,
                'subject_files': [self.subject_binding()],
                'observed_at': stamp(-10), 'diff': '+print(1)'}
        item.update(changes)
        return item

    def request(self, *, kind='completion', test=None, diff=None):
        evidence = []
        ids = []
        if test is not None:
            ids.append(self.add_evidence(evidence, 'test_run', test))
        if diff is not None:
            ids.append(self.add_evidence(evidence, 'diff', diff))
        return {'schema_version': 1, 'workspace': str(self.workspace),
                'run_id': self.run_id, 'started_at': self.window,
                'claims': [{'id': 'c1', 'text': 'Tests passed and change present',
                            'kind': kind, 'evidence_ids': ids}], 'evidence': evidence}

    def factory(self, label='supported'):
        def make():
            self.calls.append('factory')
            def judge(items):
                self.calls.append(copy.deepcopy(items))
                return {'decisions': {item['id']: label for item in items}, 'route': {}}
            return judge
        return make

    def test_supported_needs_complete_current_evidence_and_passes_full_semantics(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        before = copy.deepcopy(request)
        result = verify_completion(request, judge_factory=self.factory())
        self.assertEqual(request, before)
        self.assertEqual(result['claims'][0]['status'], 'supported')
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['semantic_quality'], 'unqualified')
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['action_authorized'])
        self.assertEqual(result['provenance'],
                         'caller_supplied_receipts_not_execution_attestation')
        self.assertEqual(result['binding_observation_scope'],
                         'current_at_evidence_validation_only')
        self.assertIn('evidence_checked_at', result)
        item = self.calls[1][0]
        self.assertEqual(set(item), {'id', 'text', 'kind', 'evidence'})
        self.assertEqual({e['kind'] for e in item['evidence']}, {'test_run', 'diff'})
        self.assertEqual(item['evidence'][0]['receipt']['command'],
                         ['python', '-m', 'unittest'])

    def test_missing_reference_never_calls_factory(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        request['claims'][0]['evidence_ids'].append('absent')
        result = verify_completion(request, judge_factory=self.factory())
        self.assertEqual(result['claims'][0]['status'], 'unknown')
        self.assertEqual(self.calls, [])

    def test_unknown_guards_override_supported_and_failure(self):
        variants = [
            (self.make_test_receipt(run_id='previous'), self.diff_receipt()),
            (self.make_test_receipt(started_at=stamp(-180), ended_at=stamp(-170)), self.diff_receipt()),
            (self.make_test_receipt(status='not_run', exit_code=None), self.diff_receipt()),
            (self.make_test_receipt(status='skipped', exit_code=None), self.diff_receipt()),
            (self.make_test_receipt(tests={'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}), self.diff_receipt()),
            (self.make_test_receipt(tests={'passed': 1, 'failed': 0, 'errors': 0, 'skipped': 1}), self.diff_receipt()),
            (self.make_test_receipt(status='failed', exit_code=1), self.diff_receipt(observed_at=stamp(-180))),
        ]
        for test, diff in variants:
            self.calls.clear()
            with self.subTest(test=test['status'], diff=diff['observed_at']):
                result = verify_completion(self.request(test=test, diff=diff),
                                           judge_factory=self.factory())
                self.assertEqual(result['claims'][0]['status'], 'unknown')
                self.assertEqual(self.calls, [])

    def test_current_failed_execution_contradicts_before_model(self):
        request = self.request(test=self.make_test_receipt(status='failed', exit_code=1),
                               diff=self.diff_receipt())
        result = verify_completion(request, judge_factory=self.factory())
        self.assertEqual(result['claims'][0]['status'], 'contradicted')
        self.assertEqual(self.calls, [])

    def test_no_judge_is_unknown(self):
        result = verify_completion(self.request(test=self.make_test_receipt(),
                                                diff=self.diff_receipt()))
        self.assertEqual(result['claims'][0]['status'], 'unknown')
        self.assertIn('semantic_unavailable', result['claims'][0]['reasons'])

    def test_slash_paths_are_accepted_and_claim_text_changes_request_hash(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        request['workspace'] = request['workspace'].replace('\\', '/')
        for item in request['evidence']:
            file = Path(item['path'])
            receipt = json.loads(file.read_text(encoding='utf-8'))
            receipt['subject_files'][0]['path'] = receipt['subject_files'][0]['path'].replace('\\', '/')
            file.write_text(json.dumps(receipt), encoding='utf-8')
            item['path'] = item['path'].replace('\\', '/')
            item['sha256'] = digest(file)
        first = verify_completion(request, judge_factory=self.factory())
        request['claims'][0]['text'] = 'A materially different completion claim'
        second = verify_completion(request, judge_factory=self.factory())
        self.assertEqual(first['claims'][0]['status'], 'supported')
        self.assertEqual(second['claims'][0]['status'], 'supported')
        self.assertNotEqual(first['request_sha256'], second['request_sha256'])
        self.assertEqual(first['run_id'], self.run_id)

    def test_failed_and_skipped_receipt_fields_visible_without_output_body(self):
        for receipt in (self.make_test_receipt(status='failed', exit_code=3,
                                               tests={'passed': 2, 'failed': 1,
                                                      'errors': 0, 'skipped': 0},
                                               output='private failure output'),
                        self.make_test_receipt(status='skipped', exit_code=None,
                                               output='private skip output')):
            with self.subTest(status=receipt['status']):
                request = self.request(test=receipt, diff=self.diff_receipt())
                result = verify_completion(request, judge_factory=self.factory())
                binding = result['claims'][0]['evidence_bindings'][0]
                self.assertEqual(binding['status'], receipt['status'])
                self.assertEqual(binding['exit_code'], receipt['exit_code'])
                self.assertEqual(binding['tests'], receipt['tests'])
                self.assertEqual(binding['command'], receipt['command'])
                self.assertEqual(binding['run_id'], receipt['run_id'])
                self.assertEqual(binding['started_at'], receipt['started_at'])
                self.assertEqual(binding['ended_at'], receipt['ended_at'])
                self.assertNotIn('output', binding)
                self.assertEqual(result['claims'][0]['evidence_bindings'][1]['observed_at'],
                                 json.loads(Path(request['evidence'][1]['path']).read_text())['observed_at'])

    def test_outside_path_rejected_before_any_outside_stat(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        outside = self.workspace.parent / 'not-an-evidence-file.json'
        request['evidence'][0]['path'] = str(outside)
        original = core._check_components
        def guard(path):
            if path == outside:
                raise AssertionError('outside file was inspected')
            return original(path)
        with patch.object(core, '_check_components', side_effect=guard):
            with self.assertRaises(ValueError):
                verify_completion(request, judge_factory=self.factory())

    def test_noncanonical_local_paths_rejected(self):
        original = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        path = original['evidence'][0]['path'].replace('\\', '/')
        variants = ('//server/share/receipt.json', '//?/C:/receipt.json',
                    path + ':stream', path.replace('/test_run-', '//test_run-'),
                    path.replace('/test_run-', '/./test_run-'),
                    path.replace('/test_run-', '/../test_run-'))
        for bad in variants:
            with self.subTest(path=bad):
                request = copy.deepcopy(original)
                request['evidence'][0]['path'] = bad
                with self.assertRaises(ValueError):
                    verify_completion(request, judge_factory=self.factory())

    def test_tampering_current_subject_and_unused_evidence_reject_before_factory(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        request['evidence'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            verify_completion(request, judge_factory=self.factory())
        self.assertEqual(self.calls, [])
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        self.add_evidence(request['evidence'], 'diff', self.diff_receipt())
        request['evidence'][-1]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            verify_completion(request, judge_factory=self.factory())
        self.assertEqual(self.calls, [])
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        self.subject.write_text('print(2)\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            verify_completion(request, judge_factory=self.factory())
        self.assertEqual(self.calls, [])

    def test_judge_mutation_cannot_change_bindings_or_decision_ids(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        def factory():
            def judge(items):
                items[0]['id'] = 'impostor'
                items[0]['evidence'][0]['receipt']['subject_files'].clear()
                return {'decisions': {'c1': 'supported'}, 'route': {}}
            return judge
        result = verify_completion(request, judge_factory=factory)
        self.assertEqual(result['claims'][0]['id'], 'c1')
        self.assertEqual(len(result['claims'][0]['evidence_bindings'][0]['subject_files']), 1)

    def test_tool_result_failure_contradicts_but_command_never_runs(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        marker = self.workspace / 'must-not-exist'
        tool = {'schema_version': 1, 'run_id': self.run_id,
                'subject_files': [self.subject_binding()],
                'started_at': stamp(-30), 'ended_at': stamp(-20),
                'status': 'failed', 'command': ['python', '-c',
                                               'open(%r,"w").write("bad")' % str(marker)],
                'exit_code': 1, 'output': 'failure'}
        request['claims'][0]['evidence_ids'].append(
            self.add_evidence(request['evidence'], 'tool_result', tool))
        result = verify_completion(request, judge_factory=self.factory())
        self.assertEqual(result['claims'][0]['status'], 'contradicted')
        self.assertFalse(marker.exists())
        self.assertEqual(self.calls, [])

    def test_shape_bool_duplicate_future_nonfinite_and_path_rejected(self):
        cases = []
        original = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        for change in (lambda r: r.update(schema_version=True),
                       lambda r: r['claims'][0]['evidence_ids'].append('e0'),
                       lambda r: r['evidence'][0].update(path=str(self.workspace.parent / 'else.json')),
                       lambda r: r.update(extra=1),
                       lambda r: r['claims'][0].update(text=float('nan'))):
            request = copy.deepcopy(original)
            change(request)
            cases.append(request)
        for request in cases:
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    verify_completion(request, judge_factory=self.factory())
        self.assertEqual(self.calls, [])
        future = self.request(test=self.make_test_receipt(ended_at=stamp(3600)),
                              diff=self.diff_receipt())
        with self.assertRaises(ValueError):
            verify_completion(future, judge_factory=self.factory())
        duplicate = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        path = Path(duplicate['evidence'][0]['path'])
        path.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
        duplicate['evidence'][0]['sha256'] = digest(path)
        with self.assertRaises(ValueError):
            verify_completion(duplicate, judge_factory=self.factory())

    def test_model_invalid_response_rejected(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        for response in ({'decisions': {}, 'route': {}},
                         {'decisions': {'c1': 'supported', 'extra': 'supported'}, 'route': {}},
                         {'decisions': {'c1': 'yes'}, 'route': {}},
                         {'decisions': {'c1': 'supported'}, 'route': [],},
                         {'decisions': {'c1': 'supported'}, 'route': {}, 'extra': 1}):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    verify_completion(request, judge_factory=lambda response=response:
                                      lambda items: response)

    def test_symlink_evidence_rejected(self):
        request = self.request(test=self.make_test_receipt(), diff=self.diff_receipt())
        link = self.workspace / 'link.json'
        try:
            link.symlink_to(request['evidence'][0]['path'])
        except (OSError, NotImplementedError):
            self.skipTest('symlink unavailable')
        request['evidence'][0]['path'] = str(link)
        with self.assertRaises(ValueError):
            verify_completion(request, judge_factory=self.factory())
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
