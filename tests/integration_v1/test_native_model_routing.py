"""CPU-only fake Predictor checks for the bounded native F03 adapter."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration.native_model_routing import (
    NativeModelRoutingJudge, build_native_model_routing_judge,
    build_state_json_model_routing_judge,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def item():
    return {
        'request_id': 'route-1', 'request_sha256': 'b' * 64,
        'catalog_sha256': 'c' * 64,
        'current_primary': {'model_id': 'primary', 'reasoning_effort': 'ultra'},
        'task': {'summary': 'Assess a complex interface change', 'difficulty': 'complex',
                 'risk': 'high', 'context_tokens': 10000,
                 'required_tools': ['read_file']},
        'budget': {'max_estimated_cost': 5.0, 'max_latency_ms': 3000,
                   'cost_unit': 'estimated_task_units'},
        'candidates': [{
            'id': 'profile-a', 'model_id': 'model-a', 'reasoning_effort': 'high',
            'description': 'Useful on complex interfaces',
            'context_window_tokens': 16000, 'tools': ['read_file'],
            'estimated_cost': 2.0, 'estimated_latency_ms': 2000,
            'unknowns': [],
        }],
    }


class FixturePredictor:
    def __init__(self, selected='profile-a', mutate=None):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.metadata = {'route': 'state_json_uncalibrated_v1', 'binding': {'id': 'fixture'}}
        self.selected = selected
        self.mutate = mutate
        self.requests = []

    def predict(self, request):
        self.requests.append(deepcopy(request))
        keys = list(request['questions']['profile']['criteria'])
        probabilities = {key: float(key == self.selected) for key in keys}
        result = {'runtime_fingerprint': self.runtime.fingerprint,
                  'inference_total_ms': 1.0,
                  'answers': {'profile': {
                      'selected_key': self.selected, 'probabilities': probabilities,
                      'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0,
                  }}}
        if self.mutate:
            self.mutate(result)
        return result


class NativeModelRoutingTests(unittest.TestCase):
    def test_one_bound_choice_projection_and_no_execution(self):
        predictor = FixturePredictor()
        source = item()
        original = deepcopy(source)
        result = NativeModelRoutingJudge(predictor)(source)
        self.assertEqual(source, original)
        self.assertEqual(len(predictor.requests), 1)
        self.assertEqual(result['selected_profile_id'], 'profile-a')
        route = result['route']
        self.assertEqual(route['rubric'], 'model-routing-choice-v1')
        self.assertEqual(route['kind'], 'native_local')
        self.assertEqual(route['runtime_fingerprint'], 'a' * 64)
        self.assertEqual(len(route['adapter_sha256']), 64)
        self.assertEqual(route['semantic_quality'], 'unqualified')
        self.assertTrue(route['advisory'])
        self.assertFalse(route['calibrated'])
        request = predictor.requests[0]
        self.assertEqual(set(request), {'state', 'questions'})
        self.assertEqual(request['questions']['profile']['type'], 'choice')
        self.assertEqual(set(request['questions']['profile']['criteria']),
                         {'profile-a', '__abstain__'})
        self.assertEqual(request['state'], {
            'current_primary': source['current_primary'], 'task': source['task'],
            'budget': source['budget'], 'candidates': source['candidates']})
        self.assertNotIn('request_sha256', json.dumps(request))
        self.assertNotIn(source['request_sha256'], json.dumps(request))
        self.assertEqual(route['input_bindings'], [{
            'request_id': 'route-1', 'prepared_item_sha256': digest(source),
            'semantic_request_sha256': digest(request)}])

    def test_abstain_maps_to_null(self):
        result = NativeModelRoutingJudge(FixturePredictor('__abstain__'))(item())
        self.assertIsNone(result['selected_profile_id'])

    def test_reply_drift_rejected(self):
        mutations = [
            lambda r: r.update(runtime_fingerprint='d' * 64),
            lambda r: r.update(extra=True),
            lambda r: r['answers'].pop('profile'),
            lambda r: r['answers']['profile'].update(selected_key='not-eligible'),
            lambda r: r['answers']['profile'].update(calibrated=True),
            lambda r: r['answers']['profile'].update(temperature=2.0),
            lambda r: r['answers']['profile']['probabilities'].update({'profile-a': float('nan')}),
            lambda r: r['answers']['profile'].update(p_selected=0.3),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                NativeModelRoutingJudge(FixturePredictor(mutate=mutation))(item())

    def test_invalid_prepared_shape_and_closed_set_rejected_before_inference(self):
        cases = [dict(item(), extra=True), dict(item(), candidates=[]),
                 dict(item(), candidates=item()['candidates'] * 16),
                 dict(item(), request_sha256='not-sha')]
        duplicate = item()
        duplicate['candidates'].append(deepcopy(duplicate['candidates'][0]))
        cases.append(duplicate)
        reserved = item()
        reserved['candidates'][0]['id'] = '__abstain__'
        cases.append(reserved)
        malformed = item()
        malformed['candidates'][0]['unknowns'] = ['outside-code']
        cases.append(malformed)
        for field, value in [('context_window_tokens', 9999),
                             ('tools', []), ('estimated_cost', 5.1),
                             ('estimated_latency_ms', 3001)]:
            screened = item()
            screened['candidates'][0][field] = value
            cases.append(screened)
        mismatched_unknown = item()
        mismatched_unknown['candidates'][0]['estimated_cost'] = None
        cases.append(mismatched_unknown)
        bad_type = item()
        bad_type['budget']['max_estimated_cost'] = True
        cases.append(bad_type)
        bad_type = item()
        bad_type['candidates'][0]['reasoning_effort'] = []
        cases.append(bad_type)
        for invalid in cases:
            predictor = FixturePredictor()
            with self.subTest(invalid=str(invalid)[:80]), self.assertRaises(ValueError):
                NativeModelRoutingJudge(predictor)(invalid)
            self.assertEqual(predictor.requests, [])

    def test_valid_unknown_metadata_is_projected_not_invented(self):
        source = item()
        candidate = source['candidates'][0]
        candidate.update(context_window_tokens=None, tools=None,
                         estimated_cost=None, estimated_latency_ms=None,
                         unknowns=['context_capacity_unknown', 'tool_support_unknown',
                                   'cost_unknown', 'latency_unknown'])
        predictor = FixturePredictor()
        result = NativeModelRoutingJudge(predictor)(source)
        self.assertEqual(result['selected_profile_id'], 'profile-a')
        self.assertEqual(predictor.requests[0]['state']['candidates'][0], candidate)

    def test_caller_mutation_during_predict_does_not_change_binding(self):
        source = item()
        original = deepcopy(source)
        predictor = FixturePredictor()
        predict = predictor.predict

        def mutate_caller(request):
            source['task']['summary'] = 'changed concurrently'
            request['state']['task']['summary'] = 'predictor mutation'
            return predict(request)

        predictor.predict = mutate_caller
        result = NativeModelRoutingJudge(predictor)(source)
        self.assertEqual(result['route']['input_bindings'][0]['prepared_item_sha256'],
                         digest(original))
        self.assertNotEqual(source, original)

    def test_native_identity_and_uncalibrated_required(self):
        for attr, value in [('mode', 'stub'), ('backend', 'remote'),
                            ('fingerprint', 'bad')]:
            predictor = FixturePredictor()
            setattr(predictor.runtime, attr, value)
            with self.assertRaises(ValueError):
                NativeModelRoutingJudge(predictor)
        predictor = FixturePredictor()
        predictor.calibration = {}
        with self.assertRaises(ValueError):
            NativeModelRoutingJudge(predictor)

    def test_explicit_generic_and_represented_builders_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {'mode': 'native', 'backend': 'kev', 'checkpoint': directory,
                    'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
            with patch('jevlocal.integration.native_model_routing.build_native_predictor',
                       return_value=FixturePredictor()) as generic:
                self.assertEqual(build_native_model_routing_judge(spec)(item())[
                    'selected_profile_id'], 'profile-a')
                generic.assert_called_once_with(spec)
            (Path(directory) / 'state_representation.json').write_text('{}', encoding='utf-8')
            with patch('jevlocal.integration.native_model_routing.build_native_predictor') as generic:
                with self.assertRaisesRegex(ValueError, 'explicit state-json'):
                    build_native_model_routing_judge(spec)
                generic.assert_not_called()
        config = {'fixture': 'represented'}
        predictor = FixturePredictor()
        with patch('jevlocal.integration.represented_native.build_predictor',
                   return_value=predictor) as represented, \
             patch('jevlocal.integration.native_model_routing.build_native_predictor') as generic:
            judge = build_state_json_model_routing_judge(config)
            represented.assert_called_once_with(config)
            generic.assert_not_called()
        result = judge(item())
        self.assertEqual(result['route']['representation'], predictor.metadata)
        result['route']['representation']['binding']['id'] = 'mutated'
        self.assertEqual(judge.metadata['representation']['binding']['id'], 'fixture')
        with patch('jevlocal.integration.represented_native.build_predictor',
                   side_effect=RuntimeError('fixture load failed')), \
             patch('jevlocal.integration.native_model_routing.build_native_predictor') as generic:
            with self.assertRaises(RuntimeError):
                build_state_json_model_routing_judge(config)
            generic.assert_not_called()

    def test_predictor_failure_is_not_retried_or_fallback(self):
        predictor = FixturePredictor()
        calls = []

        def fail(request):
            calls.append(request)
            raise RuntimeError('fixture failure')

        predictor.predict = fail
        with self.assertRaises(RuntimeError):
            NativeModelRoutingJudge(predictor)(item())
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
