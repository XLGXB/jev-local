"""Explicit skill-description shadow tests; no native model is loaded."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from jevlocal.integration import context_shadow, context_tools
from jevlocal.integration.cli import main
from jevlocal.integration.scoped_invocation import validate_snapshot
from jevlocal.integration.shadow import request_digest
from jevlocal.integration.skill_shadow import (prepare_skill_projection,
                                                verify_skill_primary)


class SkillShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.json'
        self.runtime = self.root / 'runtime.json'
        self.primary = self.root / 'primary.json'
        self.shadow = self.root / 'shadow.json'
        self.arguments = {'task': 'repair timeout', 'enabled_skills': [
            {'id': 'b', 'description': 'unrelated charts'},
            {'id': 'a', 'description': 'repair timeout; token=private-value'},
            {'id': 'c', 'description': 'timeout diagnostics'}],
            'max_chars': 5000}
        self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
        self.runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
            'checkpoint': str(self.root), 'device': 'cpu', 'amp': 'none',
            'weights_dtype': 'fp32'}), encoding='utf-8')

    def flags(self):
        return ['--shadow-runtime', str(self.runtime), '--shadow-output', str(self.shadow)]

    def invoke(self, *, flags=(), start=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(context_shadow, 'start_shadow', side_effect=start) if start else patch.object(
                context_shadow, 'start_shadow', side_effect=AssertionError('unexpected model start')):
            with redirect_stdout(out), redirect_stderr(err):
                code = main(['context', 'suggest_skills', '--input', str(self.input), *flags])
        return code, out.getvalue(), err.getvalue()

    def test_default_equivalence_and_empty_set_no_spawn(self):
        code, out, err = self.invoke()
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), context_tools.suggest_skills(**self.arguments))
        self.arguments['enabled_skills'] = []
        self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
        code, out, err = self.invoke(flags=self.flags())
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), context_tools.suggest_skills(**self.arguments))
        sidecar = json.loads(self.shadow.read_text())
        self.assertEqual(sidecar['availability']['reason'], 'no_candidates')
        self.assertIsNone(sidecar['projection'])
        self.assertEqual(sidecar['primary_feedback']['status'], 'verified')

    def test_primary_bytes_and_order_unchanged_for_native_variations(self):
        expected = json.dumps(context_tools.suggest_skills(**self.arguments),
                              ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        for values in ({'a': .9, 'b': .8, 'c': .7},
                       {'a': .1, 'b': .2, 'c': .3},
                       {'a': .2, 'b': .99, 'c': .8}):
            with self.subTest(values=values):
                events = []
                test = self
                def start(query, safe, spec, *, timeout_s, binding, manifest, rank_input):
                    events.append('start')
                    self.assertNotIn('private-value', json.dumps((query, safe, manifest, rank_input)))
                    validate_snapshot(binding, manifest, rank_input, query, safe)
                    temp = self.root / 'observer'
                    temp.mkdir()
                    receipts = []
                    for candidate in safe:
                        envelope = context_shadow._envelope(query, candidate, binding)
                        p = values[candidate['id']]
                        selected = 'true' if p >= .5 else 'false'
                        answer = {'selected_key': selected,
                            'probabilities': {'false': 1-p, 'true': p},
                            'p_selected': max(p, 1-p), 'calibrated': False,
                            'temperature': 1.0}
                        receipts.append({'id': candidate['id'], 'record': {
                            'schema_version': 1, 'request_id': envelope['request_id'],
                            'observation_id': candidate['id'],
                            'request_sha256': request_digest(envelope),
                            'candidate_keys': {'relevance': ['false', 'true']},
                            'backend': 'laya', 'runtime_fingerprint': 'a' * 64,
                            'status': 'observed', 'answers': {'relevance': answer},
                            'shadow': True, 'execution_authorized': False,
                            'training_allowed': False, 'cloud_data_used': False,
                            'shadow_elapsed_ms': 1.0}})
                    payload = {'version': 2, 'snapshot_sha256':
                               context_shadow._snapshot_hash(query, safe),
                               'binding': binding, 'receipts': receipts}
                    (temp / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
                    class Process:
                        pid = 12345
                        def poll(self): return 0
                        def wait(self, timeout): return 0
                    real = context_shadow._Job(query, safe, 'laya', time.monotonic(),
                        10.0, temp_path=temp, process=Process(), binding=binding)
                    class Job:
                        def finish(self, projected_primary):
                            events.append('finish')
                            self_ref = json.loads(test.primary.read_text())
                            test.assertEqual(self_ref, context_tools.suggest_skills(**test.arguments))
                            return real.finish(projected_primary)
                        def close(self):
                            events.append('close')
                            return real.close()
                    return Job()
                with patch.object(context_shadow, 'start_shadow', side_effect=start):
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = main(['context', 'suggest_skills', '--input', str(self.input),
                                     *self.flags(), '--output', str(self.primary)])
                self.assertEqual((code, err.getvalue()), (0, ''))
                self.assertEqual(out.getvalue(), expected)
                self.assertEqual(json.loads(self.primary.read_text()), json.loads(expected))
                sidecar = json.loads(self.shadow.read_text())
                self.assertEqual(sidecar['primary_feedback']['status'], 'verified')
                self.assertEqual(sidecar['host_ranking'], json.loads(expected)['suggestions'])
                self.assertFalse(sidecar['cloud_data_used'])
                self.assertTrue(sidecar['shadow'])
                self.assertEqual(sidecar['projection_feedback_scope'],
                                 'projection_only_not_skill_primary')
                self.assertEqual(sidecar['projection']['context_shadow']['shadow_top'],
                                 max(values, key=values.get))
                self.assertEqual(events, ['start', 'finish', 'close'])
                self.primary.unlink(); self.shadow.unlink()

    def test_fresh_formal_binding_and_foreign_receipt_rejected(self):
        task, safe = context_tools._prepare_suggest_skills(**self.arguments)
        first = prepare_skill_projection(task, safe)
        second = prepare_skill_projection(task, safe)
        self.assertNotEqual(first[0]['binding']['invocation_id'],
                            second[0]['binding']['invocation_id'])
        for record, query, candidates in (first, second):
            self.assertEqual(record['manifest']['counts']['match'], len(safe))
            validate_snapshot(record['binding'], record['manifest'],
                              record['rank_input'], query, candidates)
        record, query, candidates = second
        # Snapshot hashes intentionally stay equal; the UUID binds each native
        # request and is checked against its receipt by the real validator.
        answer = {'selected_key': 'true', 'probabilities': {'false': .2, 'true': .8},
                  'p_selected': .8, 'calibrated': False, 'temperature': 1.0}
        def native_record(candidate, binding):
            envelope = context_shadow._envelope(query, candidate, binding)
            return {'schema_version': 1, 'request_id': envelope['request_id'],
                    'observation_id': candidate['id'],
                    'request_sha256': request_digest(envelope),
                    'candidate_keys': {'relevance': ['false', 'true']},
                    'backend': 'laya', 'runtime_fingerprint': 'a' * 64,
                    'status': 'observed', 'answers': {'relevance': answer},
                    'shadow': True, 'execution_authorized': False,
                    'training_allowed': False, 'cloud_data_used': False,
                    'shadow_elapsed_ms': 1.0}
        valid = {'version': 2, 'snapshot_sha256': context_shadow._snapshot_hash(query, candidates),
                 'binding': record['binding'], 'receipts': [
                     {'id': candidate['id'], 'record': native_record(candidate, record['binding'])}
                     for candidate in candidates]}
        context_shadow._validate_output(valid, query, candidates, 'laya',
                                        binding=record['binding'])
        bad = copy.deepcopy(valid)
        bad['receipts'][1]['id'] = bad['receipts'][0]['id']
        with self.assertRaises(ValueError):
            context_shadow._validate_output(bad, query, candidates, 'laya',
                                            binding=record['binding'])
        bad = copy.deepcopy(valid)
        bad['receipts'][1]['id'] = 'unknown'
        with self.assertRaises(ValueError):
            context_shadow._validate_output(bad, query, candidates, 'laya',
                                            binding=record['binding'])
        foreign = {'version': 2, 'snapshot_sha256': context_shadow._snapshot_hash(query, candidates),
                   'binding': second[0]['binding'],
                   'receipts': [{'id': candidate['id'],
                       'record': native_record(candidate, first[0]['binding'])}
                       for candidate in candidates]}
        with self.assertRaises(ValueError):
            context_shadow._validate_output(foreign, query, candidates,
                                            'laya', binding=second[0]['binding'])

    def test_invalid_ids_and_paths_rejected_before_start(self):
        for skills in ([{'id': 'a', 'description': 'one'},
                        {'id': 'a', 'description': 'two'}],
                       [{'id': 'bad/id', 'description': 'one'}]):
            self.arguments['enabled_skills'] = skills
            self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
            code, _, _ = self.invoke(flags=self.flags())
            self.assertEqual(code, 2)
            self.assertFalse(self.shadow.exists())
        self.arguments['enabled_skills'] = [{'id': 'a', 'description': 'one'}]
        self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
        self.shadow.write_text('owner', encoding='utf-8')
        code, _, _ = self.invoke(flags=self.flags())
        self.assertEqual(code, 2)
        self.assertEqual(self.shadow.read_text(), 'owner')
        self.shadow.unlink()
        code, _, _ = self.invoke(flags=['--shadow-runtime', str(self.runtime),
                                     '--shadow-output', str(self.input)])
        self.assertEqual(code, 2)
        self.assertFalse(self.shadow.exists())

    def test_unavailable_and_primary_verifier(self):
        def start(query, safe, spec, **kw):
            return context_shadow.unavailable_shadow(query, safe, spec,
                binding=kw['binding'])
        code, out, err = self.invoke(flags=self.flags(), start=start)
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), context_tools.suggest_skills(**self.arguments))
        receipt = json.loads(self.shadow.read_text())
        self.assertEqual(receipt['availability']['status'], 'unavailable')
        self.assertEqual(receipt['primary_feedback']['status'], 'verified')
        task, safe = context_tools._prepare_suggest_skills(**self.arguments)
        wrong = context_tools.suggest_skills(**self.arguments)
        wrong['suggestions'] = list(reversed(wrong['suggestions']))
        self.assertEqual(verify_skill_primary(task, safe, wrong,
                                             max_chars=5000)['status'], 'failed')
        self.assertEqual(verify_skill_primary(task, safe, {'status': True},
                                             max_chars=5000)['status'], 'failed')

    def test_large_unicode_projection_capacity_preserves_lexical_primary(self):
        self.arguments = {'task': '汉', 'enabled_skills': [
            {'id': f's{i:03}', 'description': '汉' * 4096} for i in range(128)],
            'max_chars': 50000}
        self.input.write_text(json.dumps(self.arguments, ensure_ascii=False), encoding='utf-8')
        expected = context_tools.suggest_skills(**self.arguments)
        self.assertEqual(expected['status'], 'ok')
        code, out, err = self.invoke(flags=self.flags())
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), expected)
        sidecar = json.loads(self.shadow.read_text())
        self.assertEqual(sidecar['availability']['reason'], 'capacity')
        self.assertEqual(sidecar['primary_feedback']['status'], 'verified')
        self.assertEqual(sidecar['host_ranking'], expected['suggestions'])

    def test_invalid_child_output_and_timeout_close_owned_process(self):
        expected = context_tools.suggest_skills(**self.arguments)
        for mode in ('invalid_output', 'timeout'):
            with self.subTest(mode=mode):
                temp = self.root / ('job-' + mode)
                temp.mkdir()
                if mode == 'invalid_output':
                    (temp / 'result.json').write_text('{bad child bytes', encoding='utf-8')
                class Process:
                    pid = 12345
                    killed = False
                    def poll(self):
                        return None if not self.killed else -9
                    def kill(self):
                        self.killed = True
                    def wait(self, timeout):
                        if mode == 'timeout' and not self.killed:
                            raise subprocess.TimeoutExpired('fake-child', timeout)
                        return -9 if self.killed else 0
                proc = Process()
                def start(query, safe, spec, *, timeout_s, binding, manifest, rank_input):
                    validate_snapshot(binding, manifest, rank_input, query, safe)
                    return context_shadow._Job(query, safe, 'laya', time.monotonic(),
                        0.01 if mode == 'timeout' else 10.0,
                        temp_path=temp, process=proc, binding=binding)
                code, out, err = self.invoke(flags=self.flags(), start=start)
                self.assertEqual((code, err), (0, ''))
                self.assertEqual(json.loads(out), expected)
                receipt = json.loads(self.shadow.read_text())
                self.assertEqual(receipt['availability']['reason'], mode)
                lifecycle = receipt['projection']['context_shadow']['lifecycle']
                self.assertTrue(lifecycle['closed'])
                self.assertTrue(lifecycle['temporary_cleaned'])
                self.assertFalse(temp.exists())
                self.shadow.unlink()

    def test_startup_failure_keeps_host_primary(self):
        def start(*_args, **_kwargs):
            raise OSError('private startup detail')
        code, out, err = self.invoke(flags=self.flags(), start=start)
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), context_tools.suggest_skills(**self.arguments))
        receipt = json.loads(self.shadow.read_text())
        self.assertEqual(receipt['availability']['reason'], 'startup_failed')
        self.assertNotIn('private startup detail', out + err + self.shadow.read_text())

    def test_concurrent_sidecar_collision_never_overwrites(self):
        def start(query, safe, spec, **kw):
            real = context_shadow.unavailable_shadow(query, safe, spec,
                binding=kw['binding'])
            class Job:
                def finish(self, projected_primary):
                    self_shadow.write_text('other owner', encoding='utf-8')
                    return real.finish(projected_primary)
                def close(self):
                    return real.close()
            self_shadow = self.shadow
            return Job()
        code, out, err = self.invoke(flags=self.flags(), start=start)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), context_tools.suggest_skills(**self.arguments))
        self.assertEqual(self.shadow.read_text(), 'other owner')
        self.assertEqual(json.loads(err)['status'], 'shadow_receipt_not_written')


if __name__ == '__main__':
    unittest.main()
