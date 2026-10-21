"""Bounded host-side shadow coordination, without real native inference."""

import copy
import json
import threading
import time
from types import SimpleNamespace
import unittest

from jevlocal.integration.shadow import request_digest
from jevlocal.integration.shadow_host import ShadowHost


def envelope(request_id='r1'):
    return dict(request_id=request_id, observation_id='o1',
                state={'fixture': 'private original'},
                questions={'action': {'type': 'choice', 'instructions': 'Choose one',
                                      'criteria': {'read': 'Read', 'write': 'Write'}}})


def fixture_verifier(request, evidence):
    data = json.loads(evidence)
    if data.get('checked') is not True:
        raise ValueError('program did not check fixture')
    return dict(request_sha256=request_digest(request), outcome=data['outcome'],
                executed=data['executed'], verified_labels=data['labels'])


def evidence(*, outcome='success', executed='write', truth='read'):
    return json.dumps(dict(checked=True, outcome=outcome,
                           executed={'action': executed}, labels={'action': truth})).encode()


class Predictor:
    def __init__(self, gate=None, backend='laya'):
        self.runtime = SimpleNamespace(mode='native', backend=backend,
                                       fingerprint='a' * 64)
        self.gate = gate
        self.started = threading.Event()
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def predict(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(copy.deepcopy(request))
            self.started.set()
        try:
            if self.gate is not None and not self.gate.wait(2):
                raise TimeoutError('test gate was not released')
            return dict(runtime_fingerprint=self.runtime.fingerprint,
                        inference_total_ms=0.0,
                        answers={'action': dict(selected_key='read',
                                                probabilities={'read': 0.8, 'write': 0.2},
                                                p_selected=0.8, calibrated=False,
                                                temperature=1.0)})
        finally:
            with self.lock:
                self.active -= 1


class ShadowHostTests(unittest.TestCase):
    def host(self, predictor=None, **kwargs):
        host = ShadowHost(predictor or Predictor(), **kwargs)
        self.addCleanup(host.close, timeout_s=1)
        return host

    def test_nonblocking_snapshot_and_feedback_are_bound_to_original(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor)
        self.addCleanup(gate.set)
        original = envelope()
        digest = request_digest(original)
        started = time.monotonic()
        ticket = host.begin(original)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(predictor.started.wait(1))
        original['state']['fixture'] = 'later mutation'
        original['questions']['action']['criteria'] = {'write': 'Write', 'read': 'Read'}
        # Live host work is independent of the still-running prediction.
        live_actions = ['write exactly once']
        self.assertEqual(len(live_actions), 1)
        gate.set()
        observation = host.collect(ticket)
        self.assertEqual(observation.receipt()['status'], 'observed')
        self.assertEqual(observation.receipt()['request_sha256'], digest)
        self.assertEqual(predictor.requests[0]['state']['fixture'], 'private original')
        result = host.feedback(ticket, evidence(), verifier=fixture_verifier,
                               verifier_id='fixture-v1')
        self.assertEqual(result['feedback']['comparisons']['action']['correct'], True)
        self.assertEqual(result['feedback']['comparisons']['action']['matches_executed'], False)
        self.assertNotIn('private original', json.dumps(result))
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['cloud_data_used'])

    def test_single_worker_saturation_and_stable_collection(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=2)
        self.addCleanup(gate.set)
        first = host.begin(envelope('first'))
        self.assertTrue(predictor.started.wait(1))
        second = host.begin(envelope('second'))
        rejected = host.begin(envelope('third'))
        self.assertEqual(host.collect(rejected).receipt()['unavailable_reason'], 'saturated')
        gate.set()
        first_result = host.collect(first)
        self.assertIs(first_result, host.collect(first))
        self.assertEqual(host.collect(second).receipt()['status'], 'observed')
        self.assertEqual(len(predictor.requests), 2)
        self.assertEqual(predictor.max_active, 1)

    def test_queue_deadline_expires_before_inference(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=2)
        self.addCleanup(gate.set)
        first = host.begin(envelope('first'))
        self.assertTrue(predictor.started.wait(1))
        queued = host.begin(envelope('queued'), timeout_s=0.03)
        self.assertEqual(host.collect(queued).receipt()['unavailable_reason'], 'queue_expired')
        gate.set()
        host.collect(first)
        self.assertEqual(len(predictor.requests), 1)

    def test_worker_skips_expired_queue_without_early_collect(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=3)
        self.addCleanup(gate.set)
        first = host.begin(envelope('first'))
        self.assertTrue(predictor.started.wait(1))
        expired = host.begin(envelope('expired'), timeout_s=0.03)
        barrier = host.begin(envelope('barrier'))
        self.assertFalse(gate.wait(0.05))
        gate.set()
        self.assertEqual(host.collect(first).receipt()['status'], 'observed')
        self.assertEqual(host.collect(barrier).receipt()['status'], 'observed')
        self.assertEqual(host.collect(expired).receipt()['unavailable_reason'], 'queue_expired')
        self.assertEqual(len(predictor.requests), 2)
        self.assertEqual(predictor.requests[1]['state']['fixture'], 'private original')

    def test_completed_before_deadline_remains_available_for_late_collect(self):
        host = self.host(max_pending=2)
        ticket = host.begin(envelope('first'), timeout_s=0.2)
        barrier = host.begin(envelope('barrier'))
        self.assertEqual(host.collect(barrier).receipt()['status'], 'observed')
        # The second completion is a worker-order barrier; the first result is
        # already finalized. Collection itself happens beyond the first deadline.
        self.assertFalse(threading.Event().wait(0.25))
        self.assertEqual(host.collect(ticket).receipt()['status'], 'observed')

    def test_inference_timeout_discards_late_answer(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=1)
        self.addCleanup(gate.set)
        ticket = host.begin(envelope(), timeout_s=0.03)
        self.assertTrue(predictor.started.wait(1))
        observation = host.collect(ticket)
        self.assertEqual(observation.receipt()['unavailable_reason'], 'timeout')
        self.assertEqual(observation.receipt()['answers'], {})
        self.assertEqual(host.collect(host.begin(envelope('blocked'))).receipt()['unavailable_reason'],
                         'saturated')
        gate.set()
        self.assertTrue(host.close(timeout_s=1)['drained'])
        self.assertIs(host.collect(ticket), observation)

    def test_running_cancel_retains_capacity_until_native_returns(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=1)
        self.addCleanup(gate.set)
        ticket = host.begin(envelope())
        self.assertTrue(predictor.started.wait(1))
        host.cancel(ticket)
        self.assertEqual(host.collect(ticket).receipt()['unavailable_reason'], 'cancelled')
        rejected = host.begin(envelope('blocked'))
        self.assertEqual(host.collect(rejected).receipt()['unavailable_reason'], 'saturated')
        gate.set()
        self.assertTrue(host.close(timeout_s=1)['drained'])
        self.assertEqual(len(predictor.requests), 1)

    def test_queued_cancel_prevents_inference_and_frees_slot(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor, max_pending=2)
        self.addCleanup(gate.set)
        first = host.begin(envelope('first'))
        self.assertTrue(predictor.started.wait(1))
        queued = host.begin(envelope('queued'))
        host.cancel(queued)
        replacement = host.begin(envelope('replacement'))
        gate.set()
        host.collect(first)
        self.assertEqual(host.collect(queued).receipt()['unavailable_reason'], 'cancelled')
        self.assertEqual(host.collect(replacement).receipt()['status'], 'observed')
        self.assertEqual(len(predictor.requests), 2)

    def test_close_is_bounded_and_repeated_close_drains_running_work(self):
        gate = threading.Event()
        predictor = Predictor(gate)
        host = self.host(predictor)
        self.addCleanup(gate.set)
        ticket = host.begin(envelope())
        self.assertTrue(predictor.started.wait(1))
        started = time.monotonic()
        first = host.close(timeout_s=0.02)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(first, {'closed': True, 'drained': False, 'pending': 1})
        self.assertEqual(host.collect(ticket).receipt()['unavailable_reason'], 'cancelled')
        rejected = host.begin(envelope('closed'))
        self.assertEqual(host.collect(rejected).receipt()['unavailable_reason'], 'closed')
        gate.set()
        self.assertEqual(host.close(timeout_s=1),
                         {'closed': True, 'drained': True, 'pending': 0})

    def test_foreign_ticket_and_cancel_after_collect(self):
        host = self.host()
        other = self.host()
        ticket = host.begin(envelope())
        observation = host.collect(ticket)
        self.assertEqual(observation.receipt()['status'], 'observed')
        host.cancel(ticket)
        self.assertIs(host.collect(ticket), observation)
        for operation in (other.collect, other.cancel):
            with self.assertRaises(ValueError):
                operation(ticket)
        with self.assertRaises(ValueError):
            other.feedback(ticket, evidence(), verifier=fixture_verifier,
                           verifier_id='fixture-v1')

    def test_cancel_completed_but_uncollected_result_discards_future_use(self):
        host = self.host(max_pending=2)
        first = host.begin(envelope('first'))
        barrier = host.begin(envelope('barrier'))
        self.assertEqual(host.collect(barrier).receipt()['status'], 'observed')
        host.cancel(first)
        observation = host.collect(first)
        self.assertEqual(observation.receipt()['unavailable_reason'], 'cancelled')
        self.assertEqual(observation.receipt()['answers'], {})
        self.assertIs(host.collect(first), observation)

    def test_close_discards_completed_but_uncollected_result(self):
        host = self.host(max_pending=2)
        first = host.begin(envelope('first'))
        barrier = host.begin(envelope('barrier'))
        self.assertEqual(host.collect(barrier).receipt()['status'], 'observed')
        self.assertTrue(host.close(timeout_s=1)['drained'])
        self.assertEqual(host.collect(first).receipt()['unavailable_reason'], 'cancelled')

    def test_feedback_after_unavailable_has_unknown_local_correctness(self):
        host = self.host(max_pending=1)
        host.close(timeout_s=1)
        ticket = host.begin(envelope())
        result = host.feedback(ticket, evidence(), verifier=fixture_verifier,
                               verifier_id='fixture-v1')
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['request_sha256'], request_digest(envelope()))
        comparison = result['feedback']['comparisons']['action']
        self.assertIsNone(comparison['correct'])
        self.assertIsNone(comparison['matches_executed'])
        self.assertIsNone(comparison['local_key'])

    def test_failed_prediction_keeps_actual_feedback_unlabelled(self):
        class FailingPredictor(Predictor):
            def predict(self, request):
                raise RuntimeError('private predictor failure details')

        host = self.host(FailingPredictor())
        ticket = host.begin(envelope())
        result = host.feedback(ticket, evidence(), verifier=fixture_verifier,
                               verifier_id='fixture-v1')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['feedback']['comparisons']['action']['correct'])
        self.assertNotIn('private predictor failure details', json.dumps(result))

    def test_invalid_bounds_envelope_and_cloud_predictor_rejected(self):
        for bad in (0, 17, True, 1.0):
            with self.subTest(max_pending=bad), self.assertRaises(ValueError):
                ShadowHost(Predictor(), max_pending=bad)
        cloud = Predictor()
        cloud.runtime.mode = 'cloud'
        with self.assertRaises(ValueError):
            ShadowHost(cloud)
        host = self.host()
        for bad in (0, -1, 61, float('inf'), True):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                host.begin(envelope(), timeout_s=bad)
        with self.assertRaises(ValueError):
            host.begin({'state': {}, 'questions': {}})
        with self.assertRaises(ValueError):
            host.close(timeout_s=float('nan'))


if __name__ == '__main__':
    unittest.main()
