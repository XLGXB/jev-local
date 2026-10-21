"""Frozen passive-shadow boundary tests; synthetic inputs only."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from jevlocal.integration import context_shadow, context_tools
from jevlocal.integration.native_context import _CRITERIA, _INSTRUCTIONS
from jevlocal.integration.shadow import request_digest


QUERY = 'fix timeout'
SAFE = [dict(id='a', source='local.txt', start_line=10,
             text='other\nfix timeout\n'),
        dict(id='b', source='second.txt', start_line=1, text='unrelated\n')]
SPEC = dict(mode='native', backend='laya', checkpoint='C:/no-such-test-model',
            device='cpu', amp='none', weights_dtype='fp32')


def primary(store_dir=None):
    return context_tools._rank_prepared_evidence(QUERY, copy.deepcopy(SAFE),
                                                 max_chars=4000, store_dir=store_dir)


class ContextShadowTests(unittest.TestCase):
    def test_verify_primary_lines_bindings_and_store(self):
        with tempfile.TemporaryDirectory() as directory:
            result = primary(directory)
            self.assertEqual(context_shadow.verify_primary(
                QUERY, SAFE, result, store_dir=directory)['status'], 'verified')
            for key, value in [('source', 'wrong'), ('quote', 'wrong'),
                               ('start_line', 20), ('restore_start_line', 1),
                               ('sha256', '0' * 64)]:
                bad = copy.deepcopy(result)
                bad['results'][0][key] = value
                self.assertEqual(context_shadow.verify_primary(
                    QUERY, SAFE, bad, store_dir=directory)['status'], 'failed', key)
            bad = copy.deepcopy(result)
            bad['results'].append(copy.deepcopy(bad['results'][0]))
            self.assertEqual(context_shadow.verify_primary(
                QUERY, SAFE, bad, store_dir=directory)['status'], 'failed')
            row = result['results'][0]
            (Path(directory) / (row['sha256'] + '.txt')).write_text('tampered', encoding='utf-8')
            self.assertEqual(context_shadow.verify_primary(
                QUERY, SAFE, result, store_dir=directory)['status'], 'failed')

    def test_unavailable_primary_does_not_invent_labels(self):
        feedback = context_shadow.verify_primary(QUERY, SAFE, {'status': 'requires_larger_budget'})
        self.assertEqual(feedback['status'], 'not_available')
        self.assertNotIn('verified_labels', feedback)

    def test_invalid_native_startup_does_not_change_primary(self):
        expected = primary()
        job = context_shadow.start_shadow(QUERY, SAFE, SPEC, timeout_s=3.0)
        sidecar = job.finish(expected)
        self.assertEqual(expected, primary())
        self.assertEqual(sidecar['availability']['status'], 'unavailable')
        self.assertEqual(sidecar['mechanical_feedback']['status'], 'verified')
        self.assertEqual(sidecar['host_top'], expected['results'][0]['id'])
        self.assertIsNone(sidecar['shadow_top'])
        self.assertFalse(sidecar['execution_authorized'])
        self.assertFalse(sidecar['training_allowed'])
        self.assertFalse(sidecar['cloud_data_used'])
        self.assertNotIn('verified_labels', json.dumps(sidecar))
        self.assertNotIn(QUERY, json.dumps(sidecar))
        self.assertNotIn('other', json.dumps(sidecar))
        job.close()
        job.close()

    def test_explicit_unavailable_factory_does_not_spawn(self):
        with patch.object(context_shadow.subprocess, 'Popen') as spawn:
            job = context_shadow.unavailable_shadow(QUERY, SAFE, SPEC)
            sidecar = job.finish(primary())
        spawn.assert_not_called()
        self.assertEqual(sidecar['availability']['reason'], 'startup_failed')
        self.assertEqual(sidecar['mechanical_feedback']['status'], 'verified')

    def test_sanitized_snapshot_is_copied_and_same_interpreter_is_used(self):
        class Exited:
            pid = 12345
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        with tempfile.TemporaryDirectory() as checkpoint:
            spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            query, safe = context_tools._prepare_rank_evidence(
                'fix timeout token=TOP_SECRET',
                [dict(id='a', source='fixture', text='fix timeout token=TOP_SECRET')],
                max_chars=4000)
            with patch.object(context_shadow.subprocess, 'Popen', return_value=Exited()) as spawn:
                job = context_shadow.start_shadow(query, safe, spec)
            self.assertEqual(spawn.call_args.args[0][0], sys.executable)
            self.assertEqual(spawn.call_args.kwargs['stdout'], context_shadow.subprocess.DEVNULL)
            payload = (job._temp_path / 'payload.json').read_text(encoding='utf-8')
            self.assertNotIn('TOP_SECRET', payload)
            safe[0]['text'] = 'changed after start'
            self.assertNotIn('changed after start', payload)
            result = context_tools._rank_prepared_evidence(query, job._safe, max_chars=4000)
            self.assertEqual(job.finish(result)['mechanical_feedback']['status'], 'verified')

    def test_capacity_does_not_reject_128_candidate_primary(self):
        large_safe = [dict(id=f'c{i}', source='fixture', start_line=1,
                           text='x' * 8500) for i in range(128)]
        job = context_shadow.start_shadow('x', large_safe, SPEC, timeout_s=1)
        self.assertEqual(job.finish({'status': 'requires_larger_budget'})['availability']['status'],
                         'unavailable')

    def test_timeout_is_bounded_and_reaped(self):
        class Hung:
            def __init__(self):
                self.returncode = None
                self.killed = False
                self.reaped = False
            def wait(self, timeout=None):
                if self.killed:
                    self.reaped = True
                    self.returncode = -9
                    return -9
                raise __import__('subprocess').TimeoutExpired('fixture', timeout)
            def kill(self):
                self.killed = True
            def poll(self):
                return self.returncode
        hung = Hung()
        clock = SimpleNamespace(now=100.0)
        clock.monotonic = lambda: clock.now
        with tempfile.TemporaryDirectory() as checkpoint:
            valid_spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            with (patch.object(context_shadow, 'time', clock),
                  patch.object(context_shadow.subprocess, 'Popen', return_value=hung)):
                job = context_shadow.start_shadow(QUERY, SAFE, valid_spec, timeout_s=1)
                self.assertIs(job._process, hung)
                clock.now = 102.0
                sidecar = job.finish(primary())
        self.assertEqual(sidecar['availability']['status'], 'unavailable')
        self.assertTrue(hung.killed)
        self.assertTrue(hung.reaped)
        self.assertTrue(sidecar['lifecycle']['timed_out'])
        self.assertTrue(sidecar['lifecycle']['reaped'])
        self.assertFalse(job._temp_path.exists())

    def test_observation_binding_probability_and_rubric(self):
        envelope = context_shadow._envelope(QUERY, SAFE[0])
        self.assertEqual(envelope['questions']['relevance']['instructions'], _INSTRUCTIONS)
        self.assertEqual(envelope['questions']['relevance']['criteria'], _CRITERIA)
        answer = dict(selected_key='true', probabilities={'false': 0.2, 'true': 0.8},
                      p_selected=0.8, calibrated=False, temperature=1.0)
        record = dict(schema_version=1, request_id=envelope['request_id'],
                      observation_id=envelope['observation_id'],
                      request_sha256=request_digest(envelope),
                      candidate_keys={'relevance': ['false', 'true']}, backend='laya',
                      runtime_fingerprint='a' * 64, status='observed',
                      answers={'relevance': answer}, shadow=True,
                      execution_authorized=False, training_allowed=False,
                      cloud_data_used=False, shadow_elapsed_ms=1.0)
        payload = dict(version=1, receipts=[dict(id='a', record=record)],
                       snapshot_sha256=context_shadow._snapshot_hash(QUERY, [SAFE[0]]))
        context_shadow._validate_output(payload, QUERY, [SAFE[0]], 'laya')
        bad_version = copy.deepcopy(payload)
        bad_version['version'] = True
        with self.assertRaises(ValueError):
            context_shadow._validate_output(bad_version, QUERY, [SAFE[0]], 'laya')
        unavailable = copy.deepcopy(payload)
        unavailable_record = unavailable['receipts'][0]['record']
        unavailable_record['status'] = 'unavailable'
        unavailable_record['answers'] = {}
        del unavailable_record['shadow_elapsed_ms']
        unavailable_record['unavailable_reason'] = 'timeout'
        context_shadow._validate_output(unavailable, QUERY, [SAFE[0]], 'laya')
        for mutate in (
            lambda p: p['receipts'][0]['record'].update(observation_id='b'),
            lambda p: p['receipts'][0]['record']['answers']['relevance']['probabilities'].update(true=1.5),
            lambda p: p['receipts'][0]['record'].update(runtime_fingerprint='wrong'),
        ):
            bad = copy.deepcopy(payload)
            mutate(bad)
            with self.assertRaises(ValueError):
                context_shadow._validate_output(bad, QUERY, [SAFE[0]], 'laya')

    def test_unreaped_process_and_cleanup_failure_can_retry(self):
        class Unreaped:
            pid = 34567
            attempts = 0
            def poll(self):
                return None
            def kill(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError('fixture kill failure')
            def wait(self, timeout=None):
                if self.attempts == 0:
                    raise __import__('subprocess').TimeoutExpired('fixture', timeout)
                return -9
        proc = Unreaped()
        clock = SimpleNamespace(now=100.0)
        clock.monotonic = lambda: clock.now
        with tempfile.TemporaryDirectory() as checkpoint:
            spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            with (patch.object(context_shadow, 'time', clock),
                  patch.object(context_shadow.subprocess, 'Popen', return_value=proc)):
                job = context_shadow.start_shadow(QUERY, SAFE, spec, timeout_s=1)
                self.assertIs(job._process, proc)
                clock.now = 102.0
                sidecar = job.finish(primary())
        self.assertEqual(sidecar['availability']['reason'], 'cleanup_failed')
        self.assertFalse(sidecar['lifecycle']['closed'])
        self.assertFalse(sidecar['lifecycle']['reaped'])
        self.assertTrue(job._temp_path.exists())
        self.assertIs(job._process, proc)
        self.assertEqual(job.close(), {'closed': True, 'reaped': True,
                                       'temporary_cleaned': True})
        self.assertFalse(job._temp_path.exists())
        class Exited:
            pid = 56790
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        with tempfile.TemporaryDirectory() as checkpoint:
            spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            with patch.object(context_shadow.subprocess, 'Popen', return_value=Exited()):
                job = context_shadow.start_shadow(QUERY, SAFE, spec, timeout_s=1)
        with patch.object(context_shadow.shutil, 'rmtree', side_effect=OSError('fixture')):
            sidecar = job.finish(primary())
        self.assertEqual(sidecar['availability']['reason'], 'cleanup_failed')
        self.assertTrue(sidecar['lifecycle']['reaped'])
        self.assertFalse(sidecar['lifecycle']['temporary_cleaned'])
        self.assertTrue(job._temp_path.exists())
        self.assertTrue(job.close()['closed'])
        self.assertFalse(job._temp_path.exists())

    def test_post_kill_wait_timeout_can_retry(self):
        class SlowReap:
            pid = 56789
            waits = 0
            def poll(self):
                return None
            def kill(self):
                pass
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise __import__('subprocess').TimeoutExpired('fixture', timeout)
                return -9
        clock = SimpleNamespace(now=100.0)
        clock.monotonic = lambda: clock.now
        with tempfile.TemporaryDirectory() as checkpoint:
            spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            proc = SlowReap()
            with (patch.object(context_shadow, 'time', clock),
                  patch.object(context_shadow.subprocess, 'Popen', return_value=proc)):
                job = context_shadow.start_shadow(QUERY, SAFE, spec, timeout_s=1)
                self.assertIs(job._process, proc)
                clock.now = 102.0
                sidecar = job.finish(primary())
        self.assertEqual(sidecar['availability']['reason'], 'cleanup_failed')
        self.assertTrue(job._temp_path.exists())
        self.assertFalse(sidecar['lifecycle']['temporary_cleaned'])
        self.assertEqual(job.close(), {'closed': True, 'reaped': True,
                                       'temporary_cleaned': True})
        self.assertFalse(job._temp_path.exists())

    def test_late_completion_and_pre_spawn_deadline_are_unavailable(self):
        clock = SimpleNamespace(now=100.0)
        clock.monotonic = lambda: clock.now
        class Late:
            pid = 45678
            def poll(self):
                return 0
            def wait(self, timeout=None):
                clock.now = 102.0
                return 0
        with tempfile.TemporaryDirectory() as checkpoint:
            spec = {**SPEC, 'checkpoint': str(Path(checkpoint).resolve())}
            with (patch.object(context_shadow, 'time', clock),
                  patch.object(context_shadow.subprocess, 'Popen', return_value=Late())):
                job = context_shadow.start_shadow(QUERY, SAFE, spec, timeout_s=1)
                self.assertIsNotNone(job._process)
                sidecar = job.finish(primary())
            self.assertEqual(sidecar['availability']['reason'], 'timeout')
            self.assertTrue(sidecar['lifecycle']['timed_out'])
            original = context_shadow._json_bytes
            clock.now = 100.0
            def over_deadline_bytes(value):
                clock.now = 102.0
                return original(value)
            with (patch.object(context_shadow, 'time', clock),
                  patch.object(context_shadow, '_json_bytes', side_effect=over_deadline_bytes),
                  patch.object(context_shadow.subprocess, 'Popen') as spawn):
                job = context_shadow.start_shadow(QUERY, SAFE, spec, timeout_s=1)
            spawn.assert_not_called()
            self.assertIsNone(job._process)
            self.assertEqual(job.finish(primary())['availability']['reason'], 'timeout')


if __name__ == '__main__':
    unittest.main()
