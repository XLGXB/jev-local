from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration.native_review import NativeReviewJudge, build_native_review_judge, build_state_json_review_judge


class FixturePredictor:
    def __init__(self, mutate=None):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.requests = []
        self.mutate = mutate

    def predict(self, request):
        self.requests.append(deepcopy(request))
        answers = {}
        for key, selected in [('risk', '2'), ('security', 'true'),
                              ('permissions', 'false'), ('interface', 'false'),
                              ('persistence', 'true')]:
            keys = ('0', '1', '2', '3') if key == 'risk' else ('false', 'true')
            probabilities = {candidate: float(candidate == selected) for candidate in keys}
            answers[key] = {'selected_key': selected, 'probabilities': probabilities,
                            'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0}
        result = {'runtime_fingerprint': self.runtime.fingerprint,
                  'inference_total_ms': 1.0, 'answers': answers}
        if self.mutate:
            self.mutate(result)
        return result


def item(ident='hunk-1'):
    return {'id': ident, 'file_path': 'src/example.py', 'change_sha256': 'b' * 64,
            'hunk_sha256': 'c' * 64,
            'hunk': {'before_start': 1, 'before_count': 1, 'after_start': 1,
                     'after_count': 1, 'lines': [{'kind': 'add', 'before_line': None,
                                                  'after_line': 1, 'text': 'value = 1'}]},
            'contracts': []}


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class NativeReviewTests(unittest.TestCase):
    def test_exact_typed_decisions_and_advisory_provenance_without_input_mutation(self):
        predictor = FixturePredictor()
        judge = NativeReviewJudge(predictor)
        items = [item()]
        original = deepcopy(items)
        result = judge(items)
        self.assertEqual(items, original)
        self.assertEqual(result['decisions'], {'hunk-1': {'risk_score': 2, 'categories': {
            'security': True, 'permissions': False, 'interface': False, 'persistence': True}}})
        self.assertEqual(result['route']['rubric'], 'review-risk-noul-score-v2')
        self.assertEqual(result['route']['semantic_projection'], 'f04-semantic-projection-v1')
        self.assertEqual(result['route']['omitted_fields'],
                         ['change_sha256', 'hunk_sha256', 'contracts[].path',
                          'contracts[].sha256'])
        self.assertEqual(result['route']['kind'], 'native_local')
        self.assertFalse(result['route']['calibrated'])
        self.assertTrue(result['route']['advisory'])
        self.assertEqual(result['route']['semantic_quality'], 'unqualified')
        self.assertEqual(len(result['route']['adapter_sha256']), 64)
        request = predictor.requests[0]
        self.assertEqual(set(request), {'state', 'questions'})
        self.assertEqual(set(request['state']), {'file_path', 'hunk', 'contracts'})
        self.assertEqual(request['state']['file_path'], items[0]['file_path'])
        self.assertEqual(request['state']['hunk'], items[0]['hunk'])
        self.assertEqual(result['route']['input_bindings'], [
            {'id': 'hunk-1', 'prepared_item_sha256': digest(items[0]),
             'semantic_request_sha256': digest(request)}])
        self.assertEqual(set(request['questions']), {'risk', 'security', 'permissions', 'interface', 'persistence'})
        self.assertEqual(request['questions']['risk']['type'], 'score')
        self.assertEqual(request['questions']['security']['type'], 'noul')
        bodies = {(question['type'], question['instructions'], str(question['criteria']))
                  for question in request['questions'].values()}
        self.assertEqual(len(bodies), 5)
        for category in ('security', 'permissions', 'interface', 'persistence'):
            question = request['questions'][category]
            self.assertIn(category, question['instructions'])
            self.assertIn(category, question['criteria']['true'])
            self.assertIn('omitted integrity', question['instructions'])

    def test_contract_exact_text_and_semantic_fields_preserved(self):
        source = item()
        source['contracts'] = [{'id': 'contract', 'path': 'X:/contract.txt',
            'sha256': 'd' * 64, 'category': 'security', 'mandatory_review': True,
            'reviewer': 'human', 'text': '全文\n不删节\n', 'line_start': 1,
            'line_end': 2}]
        predictor = FixturePredictor()
        result = NativeReviewJudge(predictor)([source])
        self.assertEqual(predictor.requests[0]['state']['contracts'], [{
            'id': 'contract', 'category': 'security', 'mandatory_review': True,
            'reviewer': 'human', 'text': '全文\n不删节\n', 'line_start': 1,
            'line_end': 2}])
        self.assertEqual(result['route']['input_bindings'][0]['prepared_item_sha256'],
                         digest(source))

    def test_predictor_mutation_cannot_mutate_caller_input(self):
        predictor = FixturePredictor()
        predict = predictor.predict

        def mutate_request(request):
            request['state']['hunk']['lines'][0]['text'] = 'predictor mutation'
            request['state']['contracts'].append({'id': 'mutated'})
            return predict(request)

        predictor.predict = mutate_request
        items = [item()]
        original = deepcopy(items)
        NativeReviewJudge(predictor)(items)
        self.assertEqual(items, original)

    def test_entire_batch_is_validated_before_first_inference(self):
        bad_cases = [dict(item(), extra=True), dict(item('bad'), hunk_sha256=''),
                     dict(item('bad'), contracts={'not': 'a list'}),
                     dict(item('bad'), hunk={'text': 'x' * (128 * 1024)}),
                     dict(item('bad'), hunk={**item()['hunk'], 'other': 'not dropped'}),
                     dict(item('bad'), contracts=[{'id': 'c', 'path': 'p',
                         'sha256': 'd' * 64, 'category': 'security',
                         'mandatory_review': True, 'reviewer': 'human',
                         'text': 'full', 'line_start': 1, 'line_end': 1,
                         'unexpected_semantic': 'not dropped'}])]
        for invalid in bad_cases:
            predictor = FixturePredictor()
            with self.subTest(invalid=str(invalid)[:90]), self.assertRaises(ValueError):
                NativeReviewJudge(predictor)([item(), invalid])
            self.assertEqual(predictor.requests, [])
        predictor = FixturePredictor()
        with self.assertRaises(ValueError):
            NativeReviewJudge(predictor)([item(), item()])
        self.assertEqual(predictor.requests, [])

    def test_rejects_reply_identity_shape_score_boolean_and_calibration_drift(self):
        mutations = [lambda r: r.update(runtime_fingerprint='d' * 64),
                     lambda r: r.update(extra=True),
                     lambda r: r['answers'].pop('risk'),
                     lambda r: r['answers']['risk'].update(selected_key='4'),
                     lambda r: r['answers']['security'].update(selected_key='yes'),
                     lambda r: r['answers']['security'].update(calibrated=True),
                     lambda r: r['answers']['risk'].update(temperature=2.0)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                NativeReviewJudge(FixturePredictor(mutation))([item()])

    def test_requires_native_uncalibrated_identity(self):
        for attr, value in [('mode', 'stub'), ('backend', 'remote'), ('fingerprint', 'bad')]:
            predictor = FixturePredictor()
            setattr(predictor.runtime, attr, value)
            with self.assertRaises(ValueError):
                NativeReviewJudge(predictor)
        predictor = FixturePredictor()
        predictor.calibration = {}
        with self.assertRaises(ValueError):
            NativeReviewJudge(predictor)

    def test_explicit_generic_builder_rejects_representation_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {'mode': 'native', 'backend': 'kev', 'checkpoint': directory,
                    'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
            with patch('jevlocal.integration.native_review.build_native_predictor',
                       return_value=FixturePredictor()) as build:
                judge = build_native_review_judge(spec)
                build.assert_called_once_with(spec)
                self.assertIn('hunk-1', judge([item()])['decisions'])
            marker = Path(directory) / 'state_representation.json'
            marker.write_text('{}', encoding='utf-8')
            with patch('jevlocal.integration.native_review.build_native_predictor') as build:
                with self.assertRaisesRegex(ValueError, 'explicit state-json'):
                    build_native_review_judge(spec)
                build.assert_not_called()
            with patch('jevlocal.integration.native_review.build_native_predictor') as build:
                with self.assertRaises(ValueError):
                    build_native_review_judge({'explicit': 'invalid'})
                build.assert_not_called()

    def test_explicit_represented_builder_keeps_raw_metadata_and_no_generic_fallback(self):
        predictor = FixturePredictor()
        predictor.metadata = {'route': 'state_json_uncalibrated_v1',
                              'binding': {'id': 'fixture'}, 'calibration': None}
        config = {'explicit': 'represented fixture'}
        with patch('jevlocal.integration.represented_native.build_predictor',
                   return_value=predictor) as represented, \
             patch('jevlocal.integration.native_review.build_native_predictor') as generic:
            judge = build_state_json_review_judge(config)
            represented.assert_called_once_with(config)
            generic.assert_not_called()
        result = judge([item()])
        self.assertEqual(result['route']['representation'], predictor.metadata)
        self.assertFalse(result['route']['calibrated'])
        result['route']['representation']['binding']['id'] = 'mutated output'
        self.assertEqual(judge.metadata['representation']['binding']['id'], 'fixture')
        with patch('jevlocal.integration.represented_native.build_predictor',
                   side_effect=ValueError('fixture admission failure')), \
             patch('jevlocal.integration.native_review.build_native_predictor') as generic:
            with self.assertRaises(ValueError):
                build_state_json_review_judge(config)
            generic.assert_not_called()

    def test_native_failure_has_no_fallback(self):
        predictor = FixturePredictor()
        predictor.predict = lambda request: (_ for _ in ()).throw(RuntimeError('fixture failure'))
        with self.assertRaises(RuntimeError):
            NativeReviewJudge(predictor)([item()])


    def test_projection_and_binding_share_one_snapshot_if_caller_changes(self):
        from jevlocal.integration import native_review as adapter
        values = [item()]
        original = deepcopy(values[0])
        helper = adapter._project_contracts
        def mutate_caller_during_projection(value):
            values[0]['file_path'] = 'concurrent caller mutation'
            return helper(value)
        with patch.object(adapter, '_project_contracts',
                          side_effect=mutate_caller_during_projection):
            result = NativeReviewJudge(FixturePredictor())(values)
        self.assertEqual(result['route']['input_bindings'][0]['prepared_item_sha256'],
                         adapter._digest(original))
        self.assertNotEqual(values[0], original)

if __name__ == '__main__':
    unittest.main()

