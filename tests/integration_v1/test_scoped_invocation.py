"""Explicit scoped invocation tests; no model or browser is loaded."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration import context_shadow, context_tools
from jevlocal.integration.cli import main
from jevlocal.integration.evidence_scope import candidate_digest
from jevlocal.integration.scoped_invocation import prepare_invocation, replay_invocation
from jevlocal.integration.shadow import request_digest


def scope(other_object='B'):
    candidates = [
        {'id': 'a', 'source': 'one.txt', 'start_line': 3,
         'text': 'fix timeout\npassword=private-value\n'},
        {'id': 'b', 'source': 'two.txt', 'text': 'other\n'}]
    return {'query': 'fix timeout', 'candidates': candidates,
            'target': {'object_id': 'A', 'revision': 'r1'},
            'bindings': {
                'a': {'object_id': 'A', 'revision': 'r1',
                      'candidate_sha256': candidate_digest(candidates[0])},
                'b': {'object_id': other_object, 'revision': 'r1',
                      'candidate_sha256': candidate_digest(candidates[1])}},
            'max_chars': 30000, 'rank_max_chars': 5000}


class ScopedInvocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'scope.json'
        self.runtime = self.root / 'runtime.json'
        self.manifest = self.root / 'manifest.json'
        self.primary = self.root / 'primary.json'
        self.shadow = self.root / 'shadow.json'
        self.source.write_text(json.dumps(scope()), encoding='utf-8')
        self.runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
            'checkpoint': str(self.root), 'device': 'cpu', 'amp': 'none',
            'weights_dtype': 'fp32'}), encoding='utf-8')

    def argv(self):
        return ['scoped-rank-evidence', '--input', str(self.source),
                '--shadow-runtime', str(self.runtime),
                '--manifest-output', str(self.manifest),
                '--primary-output', str(self.primary),
                '--shadow-output', str(self.shadow)]

    def invoke(self, argv=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(self.argv() if argv is None else argv)
        return code, out.getvalue(), err.getvalue()

    def test_primary_first_full_manifest_and_pure_replay(self):
        events = []
        def start(query, safe, spec, *, timeout_s, binding, manifest, rank_input):
            events.append('start')
            self.assertTrue(self.manifest.exists())
            self.assertFalse(self.primary.exists())
            self.assertEqual(manifest['counts']['total'], 2)
            self.assertEqual([x['id'] for x in safe], ['a'])
            self.assertNotIn('private-value', json.dumps((query, safe, manifest, rank_input)))
            real = context_shadow.unavailable_shadow(query, safe, spec, binding=binding)
            class Job:
                def close(self):
                    events.append('close')
                    return real.close()
            job = Job()
            def finish(primary):
                events.append('finish')
                self.assertEqual(json.loads(self.primary.read_text()), primary)
                return real.finish(primary)
            job.finish = finish
            return job
        with patch.object(context_shadow, 'start_shadow', side_effect=start):
            code, out, err = self.invoke()
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(events, ['start', 'finish', 'close'])
        original = json.loads(self.source.read_text())
        record = json.loads(self.manifest.read_text())
        primary = json.loads(self.primary.read_text())
        sidecar = json.loads(self.shadow.read_text())
        self.assertEqual(json.loads(out), primary)
        self.assertEqual(primary, context_tools.rank_evidence(**record['rank_input']))
        self.assertEqual(record['manifest']['counts'],
                         {'total': 2, 'match': 1, 'mismatch': 1, 'unknown': 0})
        with patch.object(context_shadow.subprocess, 'Popen', side_effect=AssertionError('model call')):
            report = replay_invocation(original, record, primary, sidecar)
        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(report['replay']['version'], 2)
        self.assertFalse(report['origin_authenticated'])
        self.assertNotIn('private-value', self.manifest.read_text() + self.shadow.read_text())

        other = scope(other_object='C')
        self.assertEqual(prepare_invocation(other)['rank_input'], record['rank_input'])
        with self.assertRaises(ValueError):
            replay_invocation(other, record, primary, sidecar)
        for key in ('invocation_id', 'manifest_sha256', 'rank_input_sha256'):
            bad = copy.deepcopy(record)
            bad['binding'][key] = ('0' * 64 if key != 'invocation_id' else
                                   '00000000-0000-4000-8000-000000000000')
            with self.assertRaises(ValueError):
                replay_invocation(original, bad, primary, sidecar)
        bad = copy.deepcopy(sidecar)
        bad['binding']['invocation_id'] = '00000000-0000-4000-8000-000000000000'
        with self.assertRaises(ValueError):
            replay_invocation(original, record, primary, bad)
        bad = copy.deepcopy(record)
        bad['rank_input']['query'] = 'switched'
        with self.assertRaises(ValueError):
            replay_invocation(original, bad, primary, sidecar)

    def test_no_matches_zero_observer_calls(self):
        data = scope()
        data['target']['object_id'] = 'none'
        self.source.write_text(json.dumps(data), encoding='utf-8')
        with patch.object(context_shadow, 'start_shadow', side_effect=AssertionError('observer')):
            code, out, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['status'], 'no_matches')
        self.assertFalse(self.shadow.exists())
        self.assertEqual(replay_invocation(data, json.loads(self.manifest.read_text()),
            json.loads(self.primary.read_text()))['model_calls'], 0)

    def test_no_matches_cli_subprocess_has_no_model_launch(self):
        data = scope()
        data['target']['object_id'] = 'none'
        self.source.write_text(json.dumps(data), encoding='utf-8')
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
        proc = subprocess.run([sys.executable, '-m', 'jevlocal.integration.cli', *self.argv()],
                              capture_output=True, text=True, timeout=15, env=env,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['status'], 'no_matches')
        self.assertFalse(self.shadow.exists())
        replay = subprocess.run([sys.executable, '-m', 'jevlocal.integration.cli',
            'replay-scoped-rank-evidence', '--input', str(self.source),
            '--manifest', str(self.manifest), '--primary', str(self.primary)],
            capture_output=True, text=True, timeout=15, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)['model_calls'], 0)

    def test_timeout_retains_primary_and_replays_unavailable(self):
        test = self
        def start(query, safe, spec, *, timeout_s, binding, manifest, rank_input):
            real = context_shadow.unavailable_shadow(query, safe, spec, binding=binding)
            class Job:
                def finish(self, primary):
                    self_primary = json.loads(self_path.read_text())
                    test.assertEqual(self_primary, primary)
                    sidecar = real.finish(primary)
                    sidecar['availability']['reason'] = 'timeout'
                    sidecar['lifecycle']['timed_out'] = True
                    return sidecar
                def close(self):
                    return real.close()
            self_path = self.primary
            return Job()
        # A synthetic timeout verifies publication order; the existing v1
        # process tests cover actual kill/reap behavior.
        with patch.object(context_shadow, 'start_shadow', side_effect=start):
            code, _, _ = self.invoke()
        self.assertEqual(code, 0)
        report = replay_invocation(json.loads(self.source.read_text()),
            json.loads(self.manifest.read_text()), json.loads(self.primary.read_text()),
            json.loads(self.shadow.read_text()))
        self.assertEqual(report['replay']['availability']['reason'], 'timeout')

    def test_output_conflict_and_partial_publication(self):
        self.shadow.write_text('owned', encoding='utf-8')
        with patch.object(context_shadow, 'start_shadow', side_effect=AssertionError('observer')):
            code, _, _ = self.invoke()
        self.assertEqual(code, 2)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(self.shadow.read_text(), 'owned')
        self.shadow.unlink()
        from jevlocal.integration import cli
        actual = cli._write_new_atomic
        def fail_primary(path, value):
            if Path(path) == self.primary:
                raise OSError('private publication error')
            return actual(path, value)
        with patch.object(cli, '_write_new_atomic', side_effect=fail_primary), \
                patch.object(context_shadow, 'start_shadow', side_effect=lambda q, s, spec, **kw:
                    context_shadow.unavailable_shadow(q, s, spec, binding=kw['binding'])):
            code, out, err = self.invoke()
        self.assertEqual(code, 2)
        self.assertTrue(self.manifest.exists())
        self.assertFalse(self.primary.exists())
        self.assertFalse(self.shadow.exists())
        self.assertNotIn('private publication error', out + err)

    def test_child_envelope_and_receipt_binding(self):
        record = prepare_invocation(scope())
        derived = record['rank_input']
        query, safe = context_tools._prepare_rank_evidence(**derived)
        binding = record['binding']
        envelope = context_shadow._envelope(query, safe[0], binding)
        self.assertEqual(envelope['state'],
                         context_shadow._envelope(query, safe[0])['state'])
        answer = {'selected_key': 'true', 'probabilities': {'false': 0.2, 'true': 0.8},
                  'p_selected': 0.8, 'calibrated': False, 'temperature': 1.0}
        receipt = {'schema_version': 1, 'request_id': envelope['request_id'],
                   'observation_id': safe[0]['id'], 'request_sha256': request_digest(envelope),
                   'candidate_keys': {'relevance': ['false', 'true']}, 'backend': 'laya',
                   'runtime_fingerprint': 'a' * 64, 'status': 'observed',
                   'answers': {'relevance': answer}, 'shadow': True,
                   'execution_authorized': False, 'training_allowed': False,
                   'cloud_data_used': False, 'shadow_elapsed_ms': 1.0}
        output = {'version': 2, 'snapshot_sha256': context_shadow._snapshot_hash(query, safe),
                  'binding': binding, 'receipts': [{'id': safe[0]['id'], 'record': receipt}]}
        context_shadow._validate_output(output, query, safe, 'laya', binding=binding)
        bad = copy.deepcopy(output)
        bad['receipts'][0]['record']['request_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            context_shadow._validate_output(bad, query, safe, 'laya', binding=binding)

    def test_fresh_binding_changes_receipts_not_predictor_input(self):
        from decision_deploy.common import atoms_for
        from jevlocal.integration.shadow import LocalShadow
        records = [prepare_invocation(scope()), prepare_invocation(scope())]
        query, safe = context_tools._prepare_rank_evidence(**records[0]['rank_input'])
        envelopes = [context_shadow._envelope(query, safe[0], record['binding'])
                     for record in records]
        requests = []
        class Predictor:
            runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
            def predict(self, request):
                requests.append(copy.deepcopy(request))
                atoms_for(request)  # Exercise the real public model-input boundary.
                return {'runtime_fingerprint': 'a' * 64, 'inference_total_ms': 0.0,
                    'answers': {'relevance': {'selected_key': 'true',
                        'probabilities': {'false': 0.2, 'true': 0.8},
                        'p_selected': 0.8, 'calibrated': False, 'temperature': 1.0}}}
        local = LocalShadow(Predictor())
        receipts = [local.observe(envelope).receipt() for envelope in envelopes]
        self.assertTrue(all(x['status'] == 'observed' for x in receipts))
        self.assertEqual(requests[0], requests[1])
        legacy = context_shadow._envelope(query, safe[0])
        self.assertEqual(requests[0], {key: legacy[key] for key in ('state', 'questions')})
        self.assertNotEqual(envelopes[0]['request_id'], envelopes[1]['request_id'])
        self.assertNotEqual(receipts[0]['request_sha256'], receipts[1]['request_sha256'])
        output = {'version': 2, 'snapshot_sha256': context_shadow._snapshot_hash(query, safe),
                  'binding': records[1]['binding'],
                  'receipts': [{'id': safe[0]['id'], 'record': receipts[0]}]}
        with self.assertRaises(ValueError):
            context_shadow._validate_output(output, query, safe, 'laya',
                                            binding=records[1]['binding'])

    def test_actual_projection_mismatch_rejected_before_publication_or_start(self):
        actual_prepare = context_tools._prepare_rank_evidence
        def changed(**arguments):
            query, safe = actual_prepare(**arguments)
            return query + ' changed', safe
        with patch.object(context_tools, '_prepare_rank_evidence', side_effect=changed), \
                patch.object(context_shadow, 'start_shadow') as start:
            code, out, _ = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)['status'], 'rejected')
        start.assert_not_called()
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.primary.exists())
        self.assertFalse(self.shadow.exists())


if __name__ == '__main__':
    unittest.main()
