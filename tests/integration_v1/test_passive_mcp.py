import asyncio
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from jevlocal.integration import passive_session
from jevlocal.integration import context_shadow
from jevlocal.integration.context_shadow import unavailable_shadow
from jevlocal.integration.evidence_scope import candidate_digest
from jevlocal.integration.context_tools import rank_evidence
from jevlocal.integration.scoped_invocation import prepare_invocation
from jevlocal.integration.shadow import request_digest


class PassiveSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = {'mode': 'native', 'backend': 'laya',
                     'checkpoint': str(self.root), 'device': 'cpu',
                     'amp': 'none', 'weights_dtype': 'fp32'}
        self.records = self.root / 'records'
        self.session = passive_session.PassiveSession(self.spec, self.records)
        self.addCleanup(self.session.close)
        candidate = {'id': 'a', 'source': 'local note',
                     'text': 'Relevant alpha\npassword=test-secret\n'}
        self.args = {'query': 'Relevant alpha', 'candidates': [candidate],
                     'target': {'object_id': 'object-a', 'revision': 'r1'},
                     'bindings': {'a': {'object_id': 'object-a', 'revision': 'r1',
                                        'candidate_sha256': candidate_digest(candidate)}},
                     'max_chars': 12000, 'rank_max_chars': 6000}

    def _wait(self, identity):
        for _ in range(100):
            result = self.session.collect(identity)
            if result['status'] != 'pending':
                return result
            time.sleep(.01)
        self.fail('finalizer did not finish')

    def test_observed_probabilities_never_change_offline_primary(self):
        candidates = [
            {'id': 'a', 'source': 'alpha.txt', 'text': 'Unrelated note\n'},
            {'id': 'b', 'source': 'beta.txt', 'text': 'Relevant alpha evidence\n'},
            {'id': 'c', 'source': 'gamma.txt', 'text': 'Alpha evidence with detail\n'},
        ]
        self.args['candidates'] = candidates
        self.args['bindings'] = {
            candidate['id']: {'object_id': 'object-a', 'revision': 'r1',
                              'candidate_sha256': candidate_digest(candidate)}
            for candidate in candidates
        }
        derived = prepare_invocation(self.args)['rank_input']
        self.assertIsNotNone(derived)
        baseline = rank_evidence(derived['query'], derived['candidates'],
                                 max_chars=derived['max_chars'])
        self.assertEqual(baseline['status'], 'ok')
        baseline_bytes = json.dumps(baseline, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')
        primary_ids = [row['id'] for row in baseline['results']]
        self.assertEqual(set(primary_ids), {'a', 'b', 'c'})

        for profile in ('constant_high', 'constant_low', 'seeded_random'):
            with self.subTest(profile=profile):
                rng = random.Random(20260923)
                probabilities = {
                    candidate['id']: (.9 if profile == 'constant_high' else
                                      .1 if profile == 'constant_low' else
                                      round(rng.uniform(.1, .9), 6))
                    for candidate in candidates
                }

                class FakeProcess:
                    pid = 91001

                    def poll(self):
                        return 0

                    def wait(self, timeout=None):
                        return 0

                def fake_popen(argv, **kwargs):
                    payload = json.loads(Path(argv[4]).read_text(encoding='utf-8'))
                    receipts = []
                    for candidate in payload['safe_candidates']:
                        probability = probabilities[candidate['id']]
                        selected = 'true' if probability >= .5 else 'false'
                        envelope = context_shadow._envelope(
                            payload['query'], candidate, payload['binding'])
                        receipts.append({'id': candidate['id'], 'record': {
                            'schema_version': 1,
                            'request_id': envelope['request_id'],
                            'observation_id': candidate['id'],
                            'request_sha256': request_digest(envelope),
                            'candidate_keys': {'relevance': ['false', 'true']},
                            'backend': 'laya', 'runtime_fingerprint': 'a' * 64,
                            'status': 'observed', 'answers': {'relevance': {
                                'selected_key': selected,
                                'probabilities': {'false': round(1 - probability, 6),
                                                  'true': probability},
                                'p_selected': max(probability, 1 - probability),
                                'calibrated': False, 'temperature': 1.0}},
                            'shadow': True, 'execution_authorized': False,
                            'training_allowed': False, 'cloud_data_used': False,
                            'shadow_elapsed_ms': 1.0}})
                    result = {'version': 2, 'binding': payload['binding'],
                              'snapshot_sha256': context_shadow._snapshot_hash(
                                  payload['query'], payload['safe_candidates']),
                              'receipts': receipts}
                    Path(argv[6]).write_text(json.dumps(result), encoding='utf-8')
                    return FakeProcess()

                with patch.object(context_shadow.subprocess, 'Popen',
                                  side_effect=fake_popen) as launch:
                    started = self.session.start(**self.args)
                    self.assertEqual(started['observer'], 'started')
                    done = self._wait(started['invocation_id'])
                launch.assert_called_once()
                self.assertEqual(done['status'], 'complete')
                self.assertEqual(done['primary'], baseline)
                self.assertEqual(json.dumps(done['primary'], ensure_ascii=False,
                                            allow_nan=False, separators=(',', ':')).encode('utf-8'),
                                 baseline_bytes)
                self.assertEqual([row['id'] for row in done['primary']['results']], primary_ids)
                sidecar = done['sidecar']
                self.assertEqual(sidecar['availability'], {
                    'status': 'observed', 'observed_count': 3,
                    'candidate_count': 3, 'reason': None})
                observed = {item['id']: item['record']['answers']['relevance']
                            for item in sidecar['native_receipts']}
                self.assertEqual(set(observed), set(probabilities))
                for candidate_id, probability in probabilities.items():
                    self.assertEqual(observed[candidate_id]['probabilities']['true'], probability)
                    self.assertEqual(observed[candidate_id]['selected_key'],
                                     'true' if probability >= .5 else 'false')
                self.assertEqual(
                    sidecar['shadow_ranking'],
                    [{'id': candidate_id, 'probability_true': probability}
                     for candidate_id, probability in sorted(
                         probabilities.items(), key=lambda item: (-item[1], item[0]))])
                self.assertEqual([row['id'] for row in sidecar['host_ranking']], primary_ids)
                self.assertEqual(sidecar['mechanical_feedback']['status'], 'verified')

    def test_primary_returns_before_gate_and_capacity_has_fresh_bound_receipt(self):
        gate = threading.Event()
        started = threading.Event()

        class FakeJob:
            def __init__(self, query, candidates, spec, binding):
                self.query, self.candidates, self.spec, self.binding = query, candidates, spec, binding

            def finish(self, primary, *, cancel_event=None):
                started.set()
                gate.wait(1)
                reason = 'cancelled' if cancel_event.is_set() else 'worker_failed'
                return unavailable_shadow(self.query, self.candidates, self.spec,
                                          reason=reason, binding=self.binding).finish(primary)

        def fake_start(query, candidates, spec, *, binding, **kwargs):
            return FakeJob(query, candidates, spec, binding)

        with patch.object(passive_session, 'start_shadow', side_effect=fake_start):
            first = self.session.start(**self.args)
            self.assertEqual(first['status'], 'pending')
            self.assertEqual(first['primary']['method'], 'lexical')
            self.assertTrue(started.wait(1))
            second = self.session.start(**self.args)
            self.assertNotEqual(first['invocation_id'], second['invocation_id'])
            self.assertEqual(second['observer'], 'capacity')
            self.assertEqual(second['primary'], first['primary'])
            self.assertEqual(self.session.replay(second['invocation_id'])['model_calls'], 0)
            self.assertEqual(self.session.collect(second['invocation_id'])['sidecar']['availability']['reason'], 'capacity')
            self.assertEqual(self.session.cancel(first['invocation_id'])['status'], 'cancel_requested')
            gate.set()
            done = self._wait(first['invocation_id'])
            self.assertEqual(done['sidecar']['availability']['reason'], 'cancelled')
            self.assertEqual(self.session.replay(first['invocation_id'])['actions_replayed'], 0)
            self.assertNotIn('test-secret', json.dumps(done))
            self.assertNotIn('test-secret', (self.records / first['invocation_id'] /
                                             'invocation.json').read_text(encoding='utf-8'))

    def test_no_match_zero_observer_and_old_record_read_only(self):
        self.args['bindings']['a']['revision'] = 'other'
        with patch.object(passive_session, 'start_shadow', side_effect=AssertionError('native call')):
            first = self.session.start(**self.args)
        self.assertEqual(first['status'], 'complete')
        self.assertEqual(first['primary']['status'], 'no_matches')
        other = passive_session.PassiveSession(self.spec, self.records)
        try:
            self.assertEqual(other.collect(first['invocation_id'])['status'], 'complete')
            self.assertEqual(other.replay(first['invocation_id'])['model_calls'], 0)
            self.assertIsNone(other.collect(first['invocation_id'])['sidecar'])
        finally:
            other.close()

    def test_interrupted_restart_and_swapped_record_rejected(self):
        gate = threading.Event()

        class FakeJob:
            def finish(self, primary, *, cancel_event=None):
                gate.wait(1)
                return unavailable_shadow(self_query, self_candidates, self_spec,
                                          reason='cancelled', binding=self_binding).finish(primary)

        self_query = self.args['query']
        self_candidates = [{'id': 'a', 'source': 'local note', 'start_line': 1,
                            'text': 'Relevant alpha\n[REDACTED]\n'}]
        self_spec = self.spec
        self_binding = None
        def fake_start(query, candidates, spec, *, binding, **kwargs):
            nonlocal self_query, self_candidates, self_spec, self_binding
            self_query, self_candidates, self_spec, self_binding = query, candidates, spec, binding
            return FakeJob()

        with patch.object(passive_session, 'start_shadow', side_effect=fake_start):
            first = self.session.start(**self.args)
            other = passive_session.PassiveSession(self.spec, self.records)
            try:
                self.assertEqual(other.collect(first['invocation_id'])['status'], 'interrupted')
                self.assertEqual(other.replay(first['invocation_id'])['status'], 'interrupted')
                with self.assertRaises(ValueError):
                    other.collect('../escape')
            finally:
                other.close()
            self.session.cancel(first['invocation_id'])
            gate.set()
            self._wait(first['invocation_id'])
            second = self.session.start(**self.args)
            self._wait(second['invocation_id'])
            first_sidecar = self.records / first['invocation_id'] / 'sidecar.json'
            second_sidecar = self.records / second['invocation_id'] / 'sidecar.json'
            first_bytes, second_bytes = first_sidecar.read_bytes(), second_sidecar.read_bytes()
            first_sidecar.write_bytes(second_bytes)
            second_sidecar.write_bytes(first_bytes)
            with self.assertRaises(ValueError):
                self.session.replay(first['invocation_id'])

    def test_publication_failure_keeps_incomplete_record_without_native_launch(self):
        original = passive_session._write_once

        def fail_primary(path, value):
            if path.name == 'primary.json':
                raise OSError('private fixture failure')
            return original(path, value)

        with patch.object(passive_session, '_write_once', side_effect=fail_primary), \
             patch.object(passive_session, 'start_shadow', side_effect=AssertionError('native call')):
            with self.assertRaises(OSError):
                self.session.start(**self.args)
        dirs = list(self.records.iterdir())
        self.assertEqual(len(dirs), 1)
        self.assertTrue((dirs[0] / 'invocation.json').exists())
        self.assertFalse((dirs[0] / 'primary.json').exists())
        identity = dirs[0].name
        self.assertEqual(self.session.collect(identity)['status'], 'interrupted')

    def test_cancel_event_is_finalizer_owned_and_reaps_fake_process(self):
        import subprocess

        class Hung:
            pid = 91234

            def __init__(self):
                self.killed = False
                self.reaped = False

            def poll(self):
                return -9 if self.killed else None

            def wait(self, timeout=None):
                if not self.killed:
                    time.sleep(min(timeout or .01, .01))
                    raise subprocess.TimeoutExpired('fake', timeout)
                self.reaped = True
                return -9

            def kill(self):
                self.killed = True

        fake = Hung()
        query = 'Relevant alpha'
        candidates = [{'id': 'a', 'source': 'local note', 'start_line': 1,
                       'text': 'Relevant alpha\n'}]
        from jevlocal.integration.context_tools import _rank_prepared_evidence
        with patch.object(context_shadow.subprocess, 'Popen', return_value=fake):
            job = context_shadow.start_shadow(query, candidates, self.spec, timeout_s=3)
        event = threading.Event()
        result = []
        thread = threading.Thread(target=lambda: result.append(job.finish(
            _rank_prepared_evidence(query, candidates, max_chars=6000),
            cancel_event=event)))
        thread.start()
        time.sleep(.03)
        event.set()
        thread.join(1.5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(fake.killed)
        self.assertTrue(fake.reaped)
        self.assertEqual(result[0]['availability']['reason'], 'cancelled')
        self.assertTrue(result[0]['lifecycle']['closed'])

    def test_sidecar_publication_is_atomic_and_collect_never_reads_partial(self):
        finish_gate = threading.Event()
        link_entered = threading.Event()
        link_gate = threading.Event()

        class FakeJob:
            def __init__(self, query, candidates, spec, binding):
                self.query, self.candidates, self.spec, self.binding = query, candidates, spec, binding

            def finish(self, primary, *, cancel_event=None):
                finish_gate.wait(1)
                return unavailable_shadow(self.query, self.candidates, self.spec,
                                          reason='worker_failed', binding=self.binding).finish(primary)

        with patch.object(passive_session, 'start_shadow',
                          side_effect=lambda q, c, s, *, binding, **kw: FakeJob(q, c, s, binding)):
            started = self.session.start(**self.args)
            real_link = passive_session.os.link

            def gated_link(source, destination, *args, **kwargs):
                if Path(destination).name == 'sidecar.json':
                    link_entered.set()
                    link_gate.wait(1)
                return real_link(source, destination, *args, **kwargs)

            with patch.object(passive_session.os, 'link', side_effect=gated_link):
                finish_gate.set()
                self.assertTrue(link_entered.wait(1))
                self.assertFalse((self.records / started['invocation_id'] / 'sidecar.json').exists())
                link_gate.set()
                self.assertEqual(self._wait(started['invocation_id'])['status'], 'complete')

    def test_cleanup_failure_quarantines_capacity_and_close_reports_interrupted(self):
        def fake_start(query, candidates, spec, *, binding, **kwargs):
            class FakeJob:
                def finish(self, primary, *, cancel_event=None):
                    sidecar = unavailable_shadow(query, candidates, spec,
                                                reason='worker_failed', binding=binding).finish(primary)
                    sidecar['availability']['reason'] = 'cleanup_failed'
                    sidecar['lifecycle']['closed'] = False
                    sidecar['lifecycle']['temporary_cleaned'] = False
                    return sidecar
            return FakeJob()

        with patch.object(passive_session, 'start_shadow', side_effect=fake_start) as spawn:
            first = self.session.start(**self.args)
            result = self._wait(first['invocation_id'])
            self.assertEqual(result['sidecar']['availability']['reason'], 'cleanup_failed')
            second = self.session.start(**self.args)
            self.assertEqual(second['observer'], 'capacity')
            spawn.assert_called_once()
            self.assertEqual(self.session.close()['status'], 'interrupted')

    def test_finish_exception_closes_same_owner_and_leaves_interrupted_record(self):
        closed = threading.Event()

        class BrokenJob:
            _pid = None

            def finish(self, primary, *, cancel_event=None):
                raise RuntimeError('private fixture error')

            def close(self):
                closed.set()
                return {'closed': True, 'reaped': False, 'temporary_cleaned': True}

        with patch.object(passive_session, 'start_shadow', return_value=BrokenJob()):
            first = self.session.start(**self.args)
            self.assertTrue(closed.wait(1))
            self.assertEqual(self._wait(first['invocation_id'])['status'], 'interrupted')
            self.assertFalse((self.records / first['invocation_id'] / 'sidecar.json').exists())
            self.assertNotIn('private fixture error',
                             (self.records / first['invocation_id'] / 'invocation.json').read_text())

    def test_cancel_wins_after_observed_finish_before_sidecar_publication(self):
        observed_ready = threading.Event()
        release_finish = threading.Event()

        class FinishedJob:
            def __init__(self, query, candidates, spec, binding):
                self.query, self.candidates, self.spec, self.binding = query, candidates, spec, binding

            def finish(self, primary, *, cancel_event=None):
                sidecar = unavailable_shadow(self.query, self.candidates, self.spec,
                                              reason='worker_failed', binding=self.binding).finish(primary)
                candidate = self.candidates[0]
                envelope = context_shadow._envelope(self.query, candidate, self.binding)
                answer = {'selected_key': 'true', 'probabilities': {'false': .2, 'true': .8},
                          'p_selected': .8, 'calibrated': False, 'temperature': 1.0}
                receipt = {'schema_version': 1, 'request_id': envelope['request_id'],
                           'observation_id': candidate['id'],
                           'request_sha256': request_digest(envelope),
                           'candidate_keys': {'relevance': ['false', 'true']},
                           'backend': 'laya', 'runtime_fingerprint': 'a' * 64,
                           'status': 'observed', 'answers': {'relevance': answer},
                           'shadow': True, 'execution_authorized': False,
                           'training_allowed': False, 'cloud_data_used': False,
                           'shadow_elapsed_ms': 1.0}
                sidecar['native_receipts'] = [{'id': candidate['id'], 'record': receipt}]
                sidecar['availability'] = {'status': 'observed', 'observed_count': 1,
                                           'candidate_count': 1, 'reason': None}
                sidecar['shadow_ranking'] = [{'id': candidate['id'], 'probability_true': .8}]
                sidecar['shadow_top'] = candidate['id']
                sidecar['lifecycle'] = {'pid': 90123, 'exit_code': 0, 'timed_out': False,
                                        'reaped': True, 'closed': True,
                                        'temporary_cleaned': True}
                observed_ready.set()
                release_finish.wait(1)
                return sidecar

        with patch.object(passive_session, 'start_shadow',
                          side_effect=lambda q, c, s, *, binding, **kw: FinishedJob(q, c, s, binding)):
            started = self.session.start(**self.args)
            identity = started['invocation_id']
            self.assertTrue(observed_ready.wait(1))
            self.assertEqual(self.session.cancel(identity)['status'], 'cancel_requested')
            release_finish.set()
            result = self._wait(identity)
            self.assertEqual(result['sidecar']['availability']['reason'], 'cancelled')
            self.assertEqual(result['sidecar']['native_receipts'], [])
            self.assertEqual(result['sidecar']['shadow_ranking'], [])
            self.assertIsNone(result['sidecar']['shadow_top'])
            self.assertEqual(result['sidecar']['lifecycle']['pid'], 90123)
            self.assertEqual(self.session.replay(identity)['model_calls'], 0)

    def test_publication_wins_before_cancel_returns_complete(self):
        published = threading.Event()
        release_publication = threading.Event()

        class FinishedJob:
            def __init__(self, query, candidates, spec, binding):
                self.query, self.candidates, self.spec, self.binding = query, candidates, spec, binding

            def finish(self, primary, *, cancel_event=None):
                return unavailable_shadow(self.query, self.candidates, self.spec,
                                          reason='worker_failed', binding=self.binding).finish(primary)

        original_write = passive_session._write_once

        def pause_after_publication(path, value):
            original_write(path, value)
            if path.name == 'sidecar.json':
                published.set()
                release_publication.wait(1)

        with patch.object(passive_session, 'start_shadow',
                          side_effect=lambda q, c, s, *, binding, **kw: FinishedJob(q, c, s, binding)), \
             patch.object(passive_session, '_write_once', side_effect=pause_after_publication):
            first = self.session.start(**self.args)
            self.assertTrue(published.wait(1))
            response = []
            caller = threading.Thread(target=lambda: response.append(
                self.session.cancel(first['invocation_id'])))
            caller.start()
            release_publication.set()
            caller.join(1)
            self.assertFalse(caller.is_alive())
            self.assertEqual(response[0]['status'], 'complete')
            self.assertEqual(self.session.collect(first['invocation_id'])['sidecar'][
                'availability']['reason'], 'worker_failed')

    def test_reparse_and_invalid_id_rejected(self):
        with self.assertRaises(ValueError):
            self.session.collect('not-a-uuid')
        link = self.root / 'link'
        link.mkdir()
        real_is_symlink = Path.is_symlink
        with patch.object(Path, 'is_symlink',
                          lambda path: path == link or real_is_symlink(path)):
            with self.assertRaises(ValueError):
                passive_session.PassiveSession(self.spec, link)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional MCP SDK required')
class PassiveMcpSurfaceTests(unittest.TestCase):
    def test_explicit_surface_and_schema(self):
        from jevlocal.integration.mcp_server import create_server
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {'mode': 'native', 'backend': 'laya', 'checkpoint': directory,
                    'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
            default = create_server(root / 'store')
            self.assertEqual(len(asyncio.run(default.list_tools())), 5)
            server = create_server(root / 'store', passive_runtime=spec,
                                   passive_records=root / 'records')
            tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
            self.assertEqual(set(tools) - {tool.name for tool in asyncio.run(default.list_tools())},
                             {'passive_start', 'passive_collect', 'passive_cancel', 'passive_replay'})
            self.assertFalse(tools['passive_start'].annotations.idempotentHint)
            self.assertFalse(tools['passive_cancel'].annotations.readOnlyHint)
            self.assertTrue(tools['passive_replay'].annotations.readOnlyHint)
            for tool in tools.values():
                schema = json.dumps(tool.inputSchema)
                for forbidden in ('runtime', 'model', 'checkpoint', 'path', 'callback', 'known_secrets'):
                    self.assertNotIn(forbidden, schema)
            with self.assertRaises(ValueError):
                create_server(root / 'store', native_runtime=spec,
                              passive_runtime=spec, passive_records=root / 'other')


if __name__ == '__main__':
    unittest.main()
