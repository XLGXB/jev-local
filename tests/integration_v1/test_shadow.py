import copy
import json
from types import SimpleNamespace
import unittest

from jevlocal.integration.shadow import LocalShadow, assess_feedback, request_digest


def envelope():
    return dict(request_id='r1', observation_id='o1', state={'fixture': 'private text, not in receipt'},
        questions={'action': {'type': 'choice', 'instructions': 'Which fixture action fits?',
                             'criteria': {'read': 'Read the file', 'write': 'Write the file'}}})


class Predictor:
    def __init__(self, selected='read', fail=False):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='c' * 64)
        self.selected, self.fail, self.requests = selected, fail, []

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.fail:
            raise RuntimeError('private error details')
        return dict(runtime_fingerprint=self.runtime.fingerprint, inference_total_ms=0.0,
            answers={'action': dict(selected_key=self.selected,
                probabilities={key: 0.8 if key == self.selected else 0.2 for key in ('read', 'write')},
                p_selected=0.8, calibrated=False, temperature=1.0)})


def fixture_verifier(request, evidence):
    # Test host code, not a serialized "verified=true" flag.
    payload = json.loads(evidence)
    if payload.get('fixture_bytes') != 'checked':
        raise ValueError('actual fixture check failed')
    return dict(request_sha256=request_digest(request), outcome=payload['outcome'],
                executed=payload['executed'], verified_labels=payload['labels'])


def feedback(*, outcome='success', executed='write', truth=None):
    return json.dumps(dict(fixture_bytes='checked', outcome=outcome,
                          executed={'action': executed}, labels={} if truth is None else {'action': truth})).encode()


class ShadowTests(unittest.TestCase):
    def test_predictor_mutation_cannot_change_frozen_observation(self):
        class MutatingPredictor(Predictor):
            def predict(self, request):
                result = super().predict(request)
                request['state'].clear()
                request['questions'].clear()
                return result
        request = envelope()
        observation = LocalShadow(MutatingPredictor()).observe(request)
        result = assess_feedback(observation, feedback(truth='read'), verifier=fixture_verifier, verifier_id='fixture-v1')
        self.assertEqual(result['request_sha256'], request_digest(request))
        self.assertTrue(result['feedback']['comparisons']['action']['correct'])

    def test_same_snapshot_read_only_receipt_and_independent_correctness(self):
        request = envelope()
        predictor = Predictor()
        observation = LocalShadow(predictor).observe(request)
        request['state']['fixture'] = 'later state'
        self.assertEqual(predictor.requests[0]['state']['fixture'], 'private text, not in receipt')
        result = assess_feedback(observation, feedback(truth='write'), verifier=fixture_verifier, verifier_id='fixture-v1')
        self.assertNotIn('private text', json.dumps(result))
        self.assertNotIn('later state', json.dumps(result))
        item = result['feedback']['comparisons']['action']
        self.assertFalse(item['correct'])
        self.assertFalse(item['matches_executed'])
        self.assertEqual(item['local_action_outcome'], 'unobserved')
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['cloud_data_used'])
        self.assertNotIn('feedback', observation.receipt())

    def test_success_without_independent_label_is_not_correctness(self):
        observation = LocalShadow(Predictor('write')).observe(envelope())
        result = assess_feedback(observation, feedback(), verifier=fixture_verifier, verifier_id='fixture-v1')
        item = result['feedback']['comparisons']['action']
        self.assertTrue(item['matches_executed'])
        self.assertEqual(item['local_action_outcome'], 'success')
        self.assertIsNone(item['correct'])

    def test_failed_actual_action_does_not_label_unexecuted_local_alternative(self):
        observation = LocalShadow(Predictor('read')).observe(envelope())
        result = assess_feedback(observation, feedback(outcome='failure'), verifier=fixture_verifier, verifier_id='fixture-v1')
        item = result['feedback']['comparisons']['action']
        self.assertIsNone(item['correct'])
        self.assertEqual(item['local_action_outcome'], 'unobserved')

    def test_unknown_and_unavailable_stay_unknown_without_private_error_echo(self):
        observation = LocalShadow(Predictor(fail=True)).observe(envelope())
        result = assess_feedback(observation, feedback(outcome='unknown'), verifier=fixture_verifier, verifier_id='fixture-v1')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['feedback']['comparisons']['action']['correct'])
        self.assertNotIn('private error details', json.dumps(result))
        known = assess_feedback(observation, feedback(outcome='unknown', truth='read'),
                                verifier=fixture_verifier, verifier_id='fixture-v1')
        self.assertEqual(known['feedback']['comparisons']['action']['verified_key'], 'read')
        self.assertIsNone(known['feedback']['comparisons']['action']['correct'])

    def test_known_proposition_does_not_invent_known_action_outcome(self):
        observation = LocalShadow(Predictor('read')).observe(envelope())
        result = assess_feedback(observation, feedback(outcome='unknown', executed='read', truth='read'),
                                 verifier=fixture_verifier, verifier_id='fixture-v1')
        item = result['feedback']['comparisons']['action']
        self.assertTrue(item['correct'])
        self.assertTrue(item['matches_executed'])
        self.assertEqual(item['local_action_outcome'], 'unknown')
        self.assertFalse(result['training_allowed'])

    def test_feedback_requires_executed_program_check_and_valid_candidates(self):
        observation = LocalShadow(Predictor()).observe(envelope())
        with self.assertRaises(ValueError):
            assess_feedback(observation, b'{"verified":true}', verifier=fixture_verifier, verifier_id='fixture-v1')
        with self.assertRaises(ValueError):
            assess_feedback(observation, feedback(truth='invented'), verifier=fixture_verifier, verifier_id='fixture-v1')

    def test_stale_binding_and_candidate_order_drift_rejected(self):
        request = envelope()
        original = request_digest(request)
        request['questions']['action']['criteria'] = {'write': 'Write the file', 'read': 'Read the file'}
        self.assertNotEqual(original, request_digest(request))
        observation = LocalShadow(Predictor()).observe(envelope())
        def stale(request, evidence):
            verdict = fixture_verifier(request, evidence)
            verdict['request_sha256'] = '0' * 64
            return verdict
        with self.assertRaises(ValueError):
            assess_feedback(observation, feedback(), verifier=stale, verifier_id='fixture-v1')

    def test_receipt_mutation_does_not_change_observation_and_cloud_route_rejected(self):
        predictor = Predictor()
        observation = LocalShadow(predictor).observe(envelope())
        receipt = observation.receipt()
        receipt['answers'].clear()
        self.assertTrue(observation.receipt()['answers'])
        predictor.runtime.mode = 'cloud'
        with self.assertRaises(ValueError):
            LocalShadow(predictor)


if __name__ == '__main__':
    unittest.main()
