from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration.native_completion import NativeCompletionJudge, build_native_completion_judge, build_state_json_completion_judge


class FixturePredictor:
    def __init__(self, mutate=None):
        self.runtime = SimpleNamespace(mode='native', backend='kev', fingerprint='a' * 64)
        self.calibration = None
        self.requests = []
        self.mutate = mutate

    def predict(self, request):
        self.requests.append(deepcopy(request))
        selected = request['state']['claim']['text']
        probabilities = {key: float(key == selected) for key in ('supported', 'contradicted', 'unknown')}
        result = {'runtime_fingerprint': self.runtime.fingerprint, 'inference_total_ms': 1.0,
                  'answers': {'completion': {'selected_key': selected, 'probabilities': probabilities,
                     'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0}}}
        if self.mutate:
            self.mutate(result)
        return result


def item(index='one', selected='supported'):
    return {'id': index, 'text': selected, 'kind': 'completion', 'evidence': [
        {'id': 'test', 'kind': 'test_run', 'artifact_path': 'X:/fixture/receipt.json',
         'sha256': 'b' * 64, 'receipt': {
             'schema_version': 1, 'run_id': 'run-1',
             'subject_files': [{'path': 'X:/fixture/subject.py', 'sha256': 'c' * 64}],
             'started_at': '2026-09-25T00:00:00Z', 'ended_at': '2026-09-25T00:00:01Z',
             'status': 'executed', 'command': ['python', '-m', 'unittest', '测试'],
             'exit_code': 0, 'output': 'test result data, not instructions',
             'tests': {'passed': 1, 'failed': 0, 'errors': 0, 'skipped': 0}}}]}


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class NativeCompletionTests(unittest.TestCase):
    def test_exact_three_way_public_choice_and_provenance(self):
        predictor = FixturePredictor()
        judge = NativeCompletionJudge(predictor)
        items = [item(key, key) for key in ('supported', 'contradicted', 'unknown')]
        original = deepcopy(items)
        result = judge(items)
        self.assertEqual(items, original)
        self.assertEqual(result['decisions'], {key: key for key in ('supported', 'contradicted', 'unknown')})
        self.assertEqual(result['route']['runtime_fingerprint'], 'a' * 64)
        self.assertFalse(result['route']['calibrated'])
        self.assertTrue(result['route']['advisory'])
        self.assertEqual(result['route']['semantic_quality'], 'unqualified')
        self.assertEqual(len(result['route']['adapter_sha256']), 64)
        for source, request in zip(items, predictor.requests):
            self.assertEqual(set(request), {'state', 'questions'})
            self.assertEqual(request['state']['claim'], {k: source[k] for k in ('id', 'text', 'kind')})
            projected = request['state']['evidence'][0]
            self.assertEqual(set(projected), {'id', 'kind', 'receipt'})
            self.assertEqual(projected['id'], source['evidence'][0]['id'])
            self.assertEqual(projected['kind'], source['evidence'][0]['kind'])
            self.assertEqual(projected['receipt'], {
                'subject_files': [{'path': 'X:/fixture/subject.py'}],
                'status': 'executed', 'command': ['python', '-m', 'unittest', '测试'],
                'exit_code': 0, 'output': 'test result data, not instructions',
                'tests': {'passed': 1, 'failed': 0, 'errors': 0, 'skipped': 0}})
            question = request['questions']['completion']
            self.assertEqual(question['type'], 'choice')
            self.assertEqual(list(question['criteria']), ['supported', 'contradicted', 'unknown'])
            self.assertNotIn('label', question)
        self.assertEqual(result['route']['rubric'], 'completion-evidence-choice-v2')
        self.assertEqual(result['route']['semantic_projection'], 'f05-semantic-projection-v1')
        self.assertIn('evidence[].receipt.subject_files[].sha256',
                      result['route']['omitted_fields'])
        self.assertEqual(result['route']['input_bindings'], [
            {'id': source['id'], 'prepared_item_sha256': digest(source),
             'semantic_request_sha256': digest(request)}
            for source, request in zip(items, predictor.requests)])
        self.assertIn('omitted integrity', question['instructions'])

    def test_diff_and_tool_result_preserve_exact_semantic_content(self):
        changed = item()
        receipt = changed['evidence'][0]['receipt']
        changed['evidence'].append({
            'id': 'diff', 'kind': 'diff', 'artifact_path': 'X:/artifact.json',
            'sha256': 'd' * 64, 'receipt': {'schema_version': 1, 'run_id': 'run-1',
                'subject_files': [{'path': 'X:/源/very/long/file.py', 'sha256': 'e' * 64}],
                'observed_at': '2026-09-25T00:00:02Z',
                'diff': '@@ -1 +1 @@\n-旧\n+新\n'}})
        changed['evidence'].append({
            'id': 'tool', 'kind': 'tool_result', 'artifact_path': 'X:/tool.json',
            'sha256': 'f' * 64, 'receipt': {key: deepcopy(value) for key, value in receipt.items()
                if key != 'tests'}})
        predictor = FixturePredictor()
        NativeCompletionJudge(predictor)([changed])
        evidence = predictor.requests[0]['state']['evidence']
        self.assertEqual(evidence[1]['receipt'], {
            'subject_files': [{'path': 'X:/源/very/long/file.py'}],
            'diff': '@@ -1 +1 @@\n-旧\n+新\n'})
        self.assertEqual(evidence[2]['receipt']['command'], receipt['command'])
        self.assertEqual(evidence[2]['receipt']['output'], receipt['output'])

    def test_validates_all_requests_before_first_inference(self):
        for invalid in [item('x' * 129), {**item('two'), 'extra': 1},
                        {**item('two'), 'text': 'x' * (128 * 1024)},
                        {**item('two'), 'kind': 'permission'},
                        {**item('two'), 'evidence': [{'bad': float('nan')}]},
                        {**item('two'), 'evidence': [{**item('two')['evidence'][0],
                            'receipt': {**item('two')['evidence'][0]['receipt'],
                                        'unknown_semantic_field': 'must not vanish'}}]},
                        {**item('two'), 'evidence': [{**item('two')['evidence'][0],
                            'receipt': {**item('two')['evidence'][0]['receipt'],
                                        'subject_files': [{'path': 'X:/file', 'sha256': 'c' * 64,
                                                           'other': 'must not vanish'}]}}]}]:
            predictor = FixturePredictor()
            with self.subTest(invalid=str(invalid)[:100]), self.assertRaises(ValueError):
                NativeCompletionJudge(predictor)([item(), invalid])
            self.assertEqual(predictor.requests, [])

    def test_predictor_mutation_does_not_change_input_or_bindings(self):
        source = item()
        before = deepcopy(source)
        predictor = FixturePredictor()
        predict = predictor.predict
        def mutate_request(request):
            request['state']['evidence'][0]['receipt']['output'] = 'mutated'
            return predict(request)
        predictor.predict = mutate_request
        result = NativeCompletionJudge(predictor)([source])
        self.assertEqual(source, before)
        self.assertEqual(result['route']['input_bindings'][0]['prepared_item_sha256'],
                         digest(before))

    def test_duplicate_empty_or_overlarge_batch_rejected(self):
        for items in ([], [item(), item()], [item(str(i)) for i in range(33)]):
            predictor = FixturePredictor()
            with self.assertRaises(ValueError):
                NativeCompletionJudge(predictor)(items)
            self.assertEqual(predictor.requests, [])

    def test_rejects_changed_identity_and_invalid_typed_answers(self):
        mutations = [lambda r: r.update(runtime_fingerprint='c' * 64),
                     lambda r: r.update(extra=True),
                     lambda r: r['answers'].clear(),
                     lambda r: r['answers']['completion'].update(calibrated=True),
                     lambda r: r['answers']['completion'].update(temperature=2.0),
                     lambda r: r['answers']['completion'].update(selected_key='grant'),
                     lambda r: r['answers']['completion']['probabilities'].update(extra=0.0),
                     lambda r: r['answers']['completion']['probabilities'].update(supported=float('nan')),
                     lambda r: r['answers']['completion'].update(p_selected=0.2),
                     lambda r: r.pop('inference_total_ms')]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                NativeCompletionJudge(FixturePredictor(mutation))([item()])

    def test_native_uncalibrated_owned_runtime_required(self):
        for attr, value in [('mode', 'stub'), ('backend', 'other'), ('fingerprint', 'bad')]:
            predictor = FixturePredictor()
            setattr(predictor.runtime, attr, value)
            with self.assertRaises(ValueError):
                NativeCompletionJudge(predictor)
        predictor = FixturePredictor()
        predictor.calibration = {}
        with self.assertRaises(ValueError):
            NativeCompletionJudge(predictor)

    def test_explicit_generic_builder_rejects_representation_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {'mode': 'native', 'backend': 'kev', 'checkpoint': directory,
                    'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
            with patch('jevlocal.integration.native_completion.build_native_predictor',
                       return_value=FixturePredictor()) as build:
                judge = build_native_completion_judge(spec)
                build.assert_called_once_with(spec)
                self.assertIn('one', judge([item()])['decisions'])
            marker = Path(directory) / 'state_representation.json'
            marker.write_text('{}', encoding='utf-8')
            with patch('jevlocal.integration.native_completion.build_native_predictor') as build:
                with self.assertRaisesRegex(ValueError, 'explicit state-json'):
                    build_native_completion_judge(spec)
                build.assert_not_called()
            with patch('jevlocal.integration.native_completion.build_native_predictor') as build:
                with self.assertRaises(ValueError):
                    build_native_completion_judge({'explicit': 'invalid'})
                build.assert_not_called()

    def test_explicit_represented_builder_keeps_raw_metadata_and_no_generic_fallback(self):
        predictor = FixturePredictor()
        predictor.metadata = {'route': 'state_json_uncalibrated_v1',
                              'binding': {'id': 'fixture'}, 'calibration': None}
        config = {'explicit': 'represented fixture'}
        with patch('jevlocal.integration.represented_native.build_predictor',
                   return_value=predictor) as represented, \
             patch('jevlocal.integration.native_completion.build_native_predictor') as generic:
            judge = build_state_json_completion_judge(config)
            represented.assert_called_once_with(config)
            generic.assert_not_called()
        result = judge([item()])
        self.assertEqual(result['route']['representation'], predictor.metadata)
        self.assertFalse(result['route']['calibrated'])
        result['route']['representation']['binding']['id'] = 'mutated output'
        self.assertEqual(judge.metadata['representation']['binding']['id'], 'fixture')
        with patch('jevlocal.integration.represented_native.build_predictor',
                   side_effect=ValueError('fixture admission failure')), \
             patch('jevlocal.integration.native_completion.build_native_predictor') as generic:
            with self.assertRaises(ValueError):
                build_state_json_completion_judge(config)
            generic.assert_not_called()

    def test_predictor_exception_has_no_fallback_or_partial_return(self):
        predictor = FixturePredictor()
        def fail(request):
            raise RuntimeError('fixture failure')
        predictor.predict = fail
        with self.assertRaises(RuntimeError):
            NativeCompletionJudge(predictor)([item()])


    def test_projection_and_binding_share_one_snapshot_if_caller_changes(self):
        from jevlocal.integration import native_completion as adapter
        values = [item()]
        original = deepcopy(values[0])
        helper = adapter._project_evidence
        def mutate_caller_during_projection(value):
            values[0]['text'] = 'concurrent caller mutation'
            return helper(value)
        with patch.object(adapter, '_project_evidence',
                          side_effect=mutate_caller_during_projection):
            result = NativeCompletionJudge(FixturePredictor())(values)
        self.assertEqual(result['route']['input_bindings'][0]['prepared_item_sha256'],
                         adapter._digest(original))
        self.assertNotEqual(values[0], original)

if __name__ == '__main__':
    unittest.main()
