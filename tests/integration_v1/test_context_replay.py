import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from jevlocal.integration.context_tools import _prepare_rank_evidence, rank_evidence
from jevlocal.integration.context_shadow import _Job, _digest, _envelope
from jevlocal.integration.shadow import request_digest
from jevlocal.integration.context_replay import load_replay, read_replay_files


def fixture(*, observed=True, stored=None, budget=12000):
    request = {'query': 'explain missing module', 'max_chars': budget,
        'candidates': [
            {'id': 'failure', 'source': 'fixture.stderr', 'start_line': 10,
             'text': 'Traceback\nModuleNotFoundError: missing module\n'},
            {'id': 'health', 'source': 'fixture.stdout', 'text': 'healthy\r\n'}]}
    if stored is not None:
        request['store_dir'] = str(stored)
    query, safe = _prepare_rank_evidence(**request)
    primary = rank_evidence(**request)
    job = _Job(query, safe, 'laya', time.monotonic(), 0,
               unavailable_reason='startup_failed')
    sidecar = job.finish(primary, store_dir=stored)
    if observed:
        for candidate in safe:
            env = _envelope(query, candidate)
            sidecar['native_receipts'].append({'id': candidate['id'], 'record': {
                'schema_version': 1, 'request_id': env['request_id'],
                'observation_id': candidate['id'], 'request_sha256': request_digest(env),
                'candidate_keys': {'relevance': ['false', 'true']},
                'backend': 'laya', 'runtime_fingerprint': 'a' * 64,
                'status': 'observed', 'answers': {'relevance': {
                    'selected_key': 'true', 'probabilities': {'false': 0.2, 'true': 0.8},
                    'p_selected': 0.8, 'calibrated': False, 'temperature': 1.0}},
                'shadow_elapsed_ms': 1.0, 'shadow': True, 'execution_authorized': False,
                'training_allowed': False, 'cloud_data_used': False}})
        sidecar['availability'] = {'status': 'observed', 'observed_count': 2,
                                  'candidate_count': 2, 'reason': None}
        sidecar['shadow_ranking'] = [{'id': x['id'], 'probability_true': 0.8} for x in safe]
        sidecar['shadow_top'] = 'failure'
        sidecar['lifecycle'] = {'pid': 123, 'exit_code': 0, 'timed_out': False,
                               'reaped': True, 'closed': True, 'temporary_cleaned': True}
    return request, primary, sidecar


def independent_fixture_verifier(request, evidence):
    # A narrow test oracle. It is not a general relevance judge or JSON labels.
    if (request['state']['query'] != 'explain missing module' or
            request['state']['candidate']['id'] != 'failure' or
            request['state']['candidate']['text'] != evidence.decode('utf-8') or
            b'ModuleNotFoundError: missing module' not in evidence):
        raise ValueError('different proposition or evidence')
    return {'request_sha256': request_digest(request), 'outcome': 'unknown',
            'executed': {}, 'verified_labels': {'relevance': 'true'}}


class ContextReplayTests(unittest.TestCase):
    def test_replay_freezes_input_and_does_not_run_or_write(self):
        req, primary, sidecar = fixture()
        with patch('subprocess.Popen', side_effect=AssertionError('no child')), \
                patch('jevlocal.integration.context_tools._store', side_effect=AssertionError('no store')):
            replay = load_replay(req, primary, sidecar)
        req['candidates'][0]['text'] = 'changed'
        primary['results'].clear()
        sidecar['native_receipts'].clear()
        report = replay.report()
        self.assertEqual(report['status'], 'consistent')
        self.assertEqual(report['availability']['observed_count'], 2)
        self.assertEqual(report['store_check'], 'not_requested')
        self.assertFalse(report['origin_authenticated'])
        self.assertFalse(report['event_time_known'])
        self.assertFalse(report['training_allowed'])
        self.assertNotIn('ModuleNotFoundError', json.dumps(report))
        report['availability'].clear()
        self.assertEqual(replay.report()['availability']['observed_count'], 2)

    def test_unknown_action_can_have_independent_proposition_truth(self):
        replay = load_replay(*fixture())
        result = replay.assess('failure', b'Traceback\nModuleNotFoundError: missing module\n',
            verifier=independent_fixture_verifier, verifier_id='fixture-only-v1')
        self.assertTrue(result['feedback']['comparisons']['relevance']['correct'])
        self.assertEqual(result['feedback']['outcome'], 'unknown')
        self.assertEqual(result['feedback']['comparisons']['relevance']['local_action_outcome'], 'unobserved')
        self.assertFalse(result['training_allowed'])
        self.assertEqual(result['replay_binding']['snapshot_sha256'], replay.report()['snapshot_sha256'])

    def test_missing_observation_is_not_fabricated_for_feedback(self):
        replay = load_replay(*fixture(observed=False))
        self.assertEqual(replay.report()['availability']['candidate_count'], 2)
        self.assertEqual(replay.report()['availability']['observed_count'], 0)
        with self.assertRaises(ValueError):
            replay.assess('failure', b'anything', verifier=independent_fixture_verifier, verifier_id='fixture')

    def test_each_input_binding_change_rejected(self):
        for key in ('query', 'text', 'source', 'start_line', 'order'):
            req, primary, sidecar = fixture()
            if key == 'query': req[key] += ' changed'
            elif key == 'order': req['candidates'].reverse()
            elif key == 'start_line': req['candidates'][0][key] += 1
            else: req['candidates'][0][key] += ' changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_replay(req, primary, sidecar)

    def test_primary_recomputed_not_merely_hash_checked(self):
        for key, value in [('quote', 'wrong'), ('score', 0.123), ('stored', True), ('start_line', True)]:
            req, primary, sidecar = fixture()
            primary['results'][0][key] = value
            sidecar['primary_result_sha256'] = _digest(primary)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_replay(req, primary, sidecar)

    def test_sidecar_derived_fields_and_flags_rejected(self):
        for key, value in [('snapshot_sha256', '0' * 64), ('version', True),
                ('training_allowed', True), ('cloud_data_used', True), ('execution_authorized', True),
                ('shadow_top', 'health'), ('host_top', 'health'), ('primary_bindings', []),
                ('candidates', []), ('host_ranking', []), ('shadow_ranking', [])]:
            req, primary, sidecar = fixture()
            sidecar[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_replay(req, primary, sidecar)

    def test_native_receipt_validation_reused(self):
        changes = [('request_sha256', '0' * 64), ('observation_id', 'wrong'),
            ('runtime_fingerprint', 'b' * 64), ('answers', {}), ('schema_version', True),
            ('training_allowed', True), ('backend', 'cloud')]
        for key, value in changes:
            req, primary, sidecar = fixture()
            sidecar['native_receipts'][0]['record'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_replay(req, primary, sidecar)
        req, primary, sidecar = fixture()
        sidecar['native_receipts'].append(copy.deepcopy(sidecar['native_receipts'][0]))
        with self.assertRaises(ValueError): load_replay(req, primary, sidecar)

    def test_availability_and_lifecycle_cannot_contradict_receipts(self):
        for section, key, value in [('availability', 'observed_count', 1),
                ('availability', 'candidate_count', True), ('lifecycle', 'reaped', False),
                ('lifecycle', 'timed_out', True), ('lifecycle', 'exit_code', 1)]:
            req, primary, sidecar = fixture()
            sidecar[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_replay(req, primary, sidecar)

    def test_store_paths_are_not_implicit_read_authority(self):
        with tempfile.TemporaryDirectory() as td:
            req, primary, sidecar = fixture(stored=Path(td) / 'store')
            req['store_dir'] = 'Z:/unrelated-private-path'
            with patch('jevlocal.integration.context_tools.restore_lines', side_effect=AssertionError('not authorized')):
                self.assertEqual(load_replay(req, primary, sidecar).report()['store_check'], 'not_requested')
            self.assertEqual(load_replay(req, primary, sidecar, store_dir=Path(td)/'store').report()['store_check'], 'verified')
            digest = primary['results'][0]['sha256']
            (Path(td)/'store'/(digest+'.txt')).write_text('corrupt', encoding='utf-8')
            with self.assertRaises(ValueError): load_replay(req, primary, sidecar, store_dir=Path(td)/'store')

    def test_budget_failure_remains_in_availability_denominator(self):
        replay = load_replay(*fixture(budget=35))
        self.assertFalse(replay.report()['citation_content_verified'])
        self.assertEqual(replay.report()['availability']['candidate_count'], 2)

    def test_cli_only_internal_error_receipt_is_not_a_valid_observation(self):
        req, primary, _ = fixture()
        with self.assertRaises(ValueError):
            load_replay(req, primary, {'schema_version': 1, 'status': 'unavailable', 'reason': 'observer_internal_error'})

    def test_unavailable_receipts_remain_unavailable_with_known_truth(self):
        req, primary, sidecar = fixture()
        for item in sidecar['native_receipts']:
            item['record'].update(status='unavailable', answers={}, unavailable_reason='inference_failed')
        sidecar['availability'].update(status='unavailable', observed_count=0, reason='native_unavailable')
        sidecar.update(shadow_ranking=[], shadow_top=None)
        result = load_replay(req, primary, sidecar).assess('failure',
            b'Traceback\nModuleNotFoundError: missing module\n',
            verifier=independent_fixture_verifier, verifier_id='fixture-only-v1')
        self.assertIsNone(result['feedback']['comparisons']['relevance']['correct'])
        self.assertEqual(result['feedback']['comparisons']['relevance']['verified_key'], 'true')

    def test_partial_availability_does_not_drop_unavailable_candidate(self):
        req, primary, sidecar = fixture()
        sidecar['native_receipts'][1]['record'].update(status='unavailable', answers={},
                                                     unavailable_reason='inference_failed')
        sidecar['availability']['observed_count'] = 1
        sidecar['shadow_ranking'] = sidecar['shadow_ranking'][:1]
        report = load_replay(req, primary, sidecar).report()
        self.assertEqual(report['availability']['candidate_count'], 2)
        self.assertEqual(report['availability']['observed_count'], 1)

    def test_reported_mechanical_reasons_cannot_echo_arbitrary_source_text(self):
        req, primary, sidecar = fixture()
        sidecar['mechanical_feedback'].update(status='failed', reasons=['private source text'])
        with self.assertRaises(ValueError): load_replay(req, primary, sidecar)

    def test_bool_temperature_is_not_numeric_one(self):
        req, primary, sidecar = fixture()
        sidecar['native_receipts'][0]['record']['answers']['relevance']['temperature'] = True
        with self.assertRaises(ValueError): load_replay(req, primary, sidecar)

    def test_file_reader_strict_json_bounds_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/name for name in ('input.json', 'primary.json', 'sidecar.json')]
            for path, obj in zip(paths, fixture()):
                path.write_text(json.dumps(obj), encoding='utf-8')
            result = read_replay_files(*paths).report()
            self.assertEqual(result['artifact_sha256']['input'], hashlib.sha256(paths[0].read_bytes()).hexdigest())
            for raw in (b'{"query":"a","query":"b"}', b'{"x":NaN}',
                        b'[' * 2000 + b']' * 2000, b' ' * (4*1024*1024+1)):
                paths[0].write_bytes(raw)
                with self.assertRaises(ValueError): read_replay_files(*paths)

    def test_cli_actual_entry_writes_new_report_and_rejects_overwrite(self):
        import contextlib
        import io
        from jevlocal.integration.cli import main
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/name for name in ('input.json', 'primary.json', 'sidecar.json')]
            for path, obj in zip(paths, fixture()): path.write_text(json.dumps(obj), encoding='utf-8')
            out = Path(td)/'report.json'
            argv = ['replay-context', '--input', str(paths[0]), '--primary', str(paths[1]),
                    '--shadow', str(paths[2]), '--output', str(out)]
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(main(argv), 0)
            before = out.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(main(argv), 2)
            self.assertEqual(out.read_bytes(), before)
            paths[0].write_bytes(b'[' * 2000 + b']' * 2000)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout): self.assertEqual(main(argv), 2)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'rejected')


if __name__ == '__main__':
    unittest.main()
