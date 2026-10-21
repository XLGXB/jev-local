"""Focused closed-set and parent/worker boundary tests; no real model inference."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration import json_render_consumer as consumer


def ui(title='Public summary'):
    return {'schema_version': 1, 'title': title,
            'layout': {'kind': 'stack', 'columns': 1, 'gap': 'normal'},
            'components': [{'id': 'headline', 'type': 'heading',
                            'props': {'text': 'Summary', 'level': 1}}]}


def request():
    return {'schema_version': 1, 'request_id': 'public-001',
            'intent': 'Show a concise public summary', 'data_classification': 'public',
            'variants': [{'id': 'summary', 'rationale': 'Concise summary', 'ui': ui()},
                         {'id': 'details', 'rationale': 'Detailed public report',
                          'ui': ui('Public details')}]}


class Predictor:
    def __init__(self, selected='summary', mutate=None):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.selected = selected
        self.mutate = mutate
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        keys = list(payload['questions']['ui_choice']['criteria'])
        probs = {key: (0.8 if key == self.selected else 0.1) for key in keys}
        result = {'runtime_fingerprint': self.runtime.fingerprint,
                  'inference_total_ms': 4.0,
                  'answers': {'ui_choice': {'selected_key': self.selected,
                                           'probabilities': probs, 'p_selected': 0.8,
                                           'calibrated': False, 'temperature': 1.0}}}
        if self.mutate:
            self.mutate(result)
        return result


def spec(checkpoint):
    return {'mode': 'native', 'backend': 'laya', 'checkpoint': str(checkpoint),
            'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}


class FakeProc:
    pid = 1234
    def __init__(self, behavior='ok'):
        self.behavior = behavior
        self.killed = False
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.behavior == 'timeout' and not self.killed:
            raise subprocess.TimeoutExpired('worker', timeout)
        if self.behavior == 'oserror' and not self.killed:
            raise OSError('synthetic wait error')
        return -9 if self.killed else 0

    def kill(self):
        self.killed = True


class ConsumerTests(unittest.TestCase):
    def test_one_choice_full_state_and_immutable_input(self):
        original = request()
        frozen = copy.deepcopy(original)
        predictor = Predictor()
        result = consumer.select_ui(original, predictor)
        self.assertEqual(original, frozen)
        self.assertEqual(len(predictor.calls), 1)
        payload = predictor.calls[0]
        self.assertEqual(payload['state']['intent'], original['intent'])
        self.assertEqual(payload['state']['encoding_version'], 1)
        self.assertNotIn('schema_version', payload['state'])  # fixed by encoding v1, restored exactly
        self.assertEqual(consumer.unpack_state(payload['state'], original['request_id']),
                         consumer.validate_selection_request(original))
        self.assertNotEqual(payload['state']['variants'], original['variants'])
        self.assertEqual(set(payload['questions']['ui_choice']['criteria']),
                         {'summary', 'details', 'defer'})
        self.assertEqual(payload['questions']['ui_choice']['type'], 'choice')
        self.assertEqual(result['ui'], consumer.json_render.validate_ui(original['variants'][0]['ui']))
        self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['model_calls'], 1)

    def test_lossless_six_component_catalog_and_order(self):
        full_ui = {'schema_version': 1, 'title': '公开概览 · Ω',
                   'layout': {'kind': 'grid', 'columns': 3, 'gap': 'wide'},
                   'components': [
                       {'id': 'first', 'type': 'heading',
                        'props': {'text': '标题 Ω', 'level': 2}},
                       {'id': 'second', 'type': 'text',
                        'props': {'text': '正文 Ω', 'tone': 'muted'}},
                       {'id': 'third', 'type': 'metric',
                        'props': {'label': '数量', 'value': '28', 'unit': '项'}},
                       {'id': 'fourth', 'type': 'list',
                        'props': {'title': '步骤', 'items': ['甲', '乙', 'Ω'], 'ordered': True}},
                       {'id': 'fifth', 'type': 'notice',
                        'props': {'text': '请留意', 'tone': 'warning'}},
                       {'id': 'sixth', 'type': 'button',
                        'props': {'label': '查看', 'action_id': 'inert-1'}}]}
        item = request()
        item['variants'] = [
            {'id': 'multi', 'rationale': '全组件样本', 'ui': full_ui},
            {'id': 'small', 'rationale': '简要样本', 'ui': ui('第二候选')}]
        safe = consumer.validate_selection_request(item)
        state = consumer.pack_state(item)
        decoded = consumer.unpack_state(state, item['request_id'])
        self.assertEqual(consumer._bytes(decoded), consumer._bytes(safe))
        self.assertEqual([v['id'] for v in decoded['variants']], ['multi', 'small'])
        self.assertEqual([c['id'] for c in decoded['variants'][0]['ui']['components']],
                         ['first', 'second', 'third', 'fourth', 'fifth', 'sixth'])
        self.assertEqual(state['variants'][0][2][2][3][2], ['步骤', ['甲', '乙', 'Ω'], True])
        original = copy.deepcopy(item)
        consumer.select_ui(item, Predictor('multi'))
        self.assertEqual(item, original)
        instructions = consumer._envelope(safe)['questions']['ui_choice']['instructions']
        for legend in ('heading=[text,level]', 'text=[text,tone]',
                       'metric=[label,value,unit]', 'list=[title,items,ordered]',
                       'notice=[text,tone]', 'button=[label,action_id]'):
            self.assertIn(legend, instructions)

    def test_frozen_public_request_roundtrip_without_mutation(self):
        path = Path(__file__).resolve().parents[2] / 'runtime' / 'json-render-01' / 'public-request.json'
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '7595885656146a500a201a5d4eca76f9594dcc8ff820924eb4cacec7d756de97')
        original = json.loads(raw)
        state = consumer.pack_state(original)
        self.assertEqual(consumer.unpack_state(state, original['request_id']),
                         consumer.validate_selection_request(original))
        self.assertEqual(path.read_bytes(), raw)

    def test_bad_encoded_shapes_rejected(self):
        valid = consumer.pack_state(request())
        changes = (
            lambda s: s.update(encoding_version=2),
            lambda s: s.update(schema_version=2),
            lambda s: s.update(unknown=True),
            lambda s: s['variants'][0].append('extra'),
            lambda s: s['variants'][0][2].pop(),
            lambda s: s['variants'][0][2][1].append('extra'),
            lambda s: s['variants'][0][2][2][0].append('extra'),
            lambda s: s['variants'][0][2][2][0][2].pop(),
            lambda s: s['variants'][0][2][2][0].__setitem__(1, 'unknown'),
            lambda s: s['variants'][0][2][2][0][2].__setitem__(1, '2'),
        )
        for change in changes:
            state = copy.deepcopy(valid)
            change(state)
            with self.subTest(change=change), self.assertRaises(ValueError):
                consumer.unpack_state(state, 'public-001')

    def test_defer_has_no_preview(self):
        result = consumer.select_ui(request(), Predictor('defer'))
        self.assertEqual(result['status'], 'deferred')
        self.assertEqual(result['selected_id'], 'defer')
        self.assertNotIn('ui', result)
        self.assertNotIn('html', result)

    def test_rejects_all_before_predictor_call(self):
        bad = []
        for change in (
                lambda r: r.update(extra=True),
                lambda r: r.update(data_classification='private'),
                lambda r: r.update(intent='Bearer synthetic-secret'),
                lambda r: r.update(intent='A\x00B'),
                lambda r: r.update(intent='x' * 2001),
                lambda r: r['variants'][0].update(id='defer'),
                lambda r: r['variants'][1].update(id='summary'),
                lambda r: r['variants'][0].update(extra=True),
                lambda r: r['variants'][0].update(rationale='x' * 501),
                lambda r: r['variants'][0]['ui'].update(extra=True),
                lambda r: r['variants'][0]['ui']['components'][0]['props'].update(text='token=secret'),
                lambda r: r['variants'][0]['ui']['components'][0]['props'].update(level=float('nan')),
                lambda r: r.update(variants=r['variants'] * 5),
        ):
            item = request()
            change(item)
            bad.append(item)
        for item in bad:
            with self.subTest(item=str(item)[:80]):
                predictor = Predictor()
                with self.assertRaises(ValueError):
                    consumer.select_ui(item, predictor)
                self.assertEqual(predictor.calls, [])
        huge = request()
        huge['variants'][0]['ui']['components'] = [
            {'id': f'c{i}', 'type': 'text', 'props': {'text': 'x' * 2000, 'tone': 'default'}}
            for i in range(24)]
        with self.assertRaises(ValueError):
            consumer.validate_selection_request(huge)

    def test_bad_probability_identity_and_calibration_fail_closed(self):
        changes = (
            lambda r: r.update(runtime_fingerprint='b' * 64),
            lambda r: r['answers']['ui_choice']['probabilities'].update(summary=float('nan')),
            lambda r: r['answers']['ui_choice']['probabilities'].update(summary=0.7),
            lambda r: r['answers']['ui_choice']['probabilities'].update(other=0.0),
            lambda r: r['answers']['ui_choice'].update(calibrated=True),
            lambda r: r['answers']['ui_choice'].update(temperature=2.0),
            lambda r: r['answers']['ui_choice'].update(selected_key='missing'),
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    consumer.select_ui(request(), Predictor(mutate=change))

    def test_predictor_cannot_mutate_envelope_or_runtime_identity(self):
        class MutatesPayload(Predictor):
            def predict(self, payload):
                result = super().predict(payload)
                payload['questions']['ui_choice']['criteria']['forged'] = 'new option'
                payload['state']['variants'].clear()
                return result
        result = consumer.select_ui(request(), MutatesPayload())
        self.assertEqual(result['selected_id'], 'summary')

        class MutatesRuntime(Predictor):
            def predict(self, payload):
                result = super().predict(payload)
                self.runtime.fingerprint = 'b' * 64
                return result
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            consumer.select_ui(request(), MutatesRuntime())

    def test_strict_json_duplicate_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            consumer._parse_json(b'{"x":1,"x":2}', 100)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            consumer._parse_json(b'{"x":NaN}', 100)

    def test_complete_request_utf8_budget_and_ascii_id(self):
        item = request()
        item['variants'] = []
        for index in range(3):
            variant_ui = ui()
            variant_ui['components'] = [
                {'id': f'c{i}', 'type': 'text',
                 'props': {'text': '字' * 460, 'tone': 'default'}}
                for i in range(16)]
            item['variants'].append({'id': f'v{index}', 'rationale': 'A public layout',
                                     'ui': variant_ui})
        with self.assertRaisesRegex(ValueError, '64 KiB'):
            consumer.validate_selection_request(item)
        item = request()
        item['variants'][0]['id'] = 'é'
        with self.assertRaises(ValueError):
            consumer.validate_selection_request(item)

    def test_forged_child_output_rejected(self):
        item = consumer.validate_selection_request(request())
        with tempfile.TemporaryDirectory() as td:
            runtime = spec(Path(td))
            result = consumer.select_ui(item, Predictor())
            result['network_policy'] = 'deny_socket_connect'
            result['runtime_spec_sha256'] = consumer._sha(runtime)
            consumer._verify_result(result, item, runtime, consumer._source_sha())
            for key, value in (('request_sha256', 'b' * 64),
                               ('catalog_fingerprint', 'b' * 64),
                               ('consumer_sha256', 'b' * 64),
                               ('runtime_spec_sha256', 'b' * 64),
                               ('html', '<script>alert(1)</script>')):
                forged = copy.deepcopy(result)
                forged[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    consumer._verify_result(forged, item, runtime, consumer._source_sha())

    def _paths(self, td):
        root = Path(td)
        checkpoint = root / 'checkpoint'
        checkpoint.mkdir()
        inp, runtime, out = root / 'input.json', root / 'runtime.json', root / 'output'
        inp.write_text(json.dumps(request()), encoding='utf-8')
        runtime.write_text(json.dumps(spec(checkpoint)), encoding='utf-8')
        return inp, runtime, out

    def test_cli_prevalidates_before_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            inp, runtime, out = self._paths(td)
            bad = request()
            bad['intent'] = 'password=secret'
            inp.write_text(json.dumps(bad), encoding='utf-8')
            with patch.object(consumer.subprocess, 'Popen') as popen:
                with self.assertRaises(ValueError):
                    consumer.run_cli(str(inp), str(runtime), str(out))
                popen.assert_not_called()
            self.assertFalse(out.exists())
            inp.write_text(json.dumps(request()), encoding='utf-8')
            out.mkdir()
            with self.assertRaises(ValueError):
                consumer.run_cli(str(inp), str(runtime), str(out))

    def test_worker_refuses_existing_result_and_bounded_file_read(self):
        with tempfile.TemporaryDirectory() as td:
            inp, runtime, out = self._paths(td)
            out.mkdir()
            result = out / 'result.json'
            result.write_text('unchanged', encoding='utf-8')
            with self.assertRaises(ValueError):
                consumer._worker(inp, runtime, result)
            self.assertEqual(result.read_text(encoding='utf-8'), 'unchanged')
            inp.write_bytes(b'x' * (64 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, 'budget'):
                consumer.run_cli(str(inp), str(runtime), str(out / 'new'))
            self.assertFalse((out / 'new').exists())

    def test_cli_timeout_and_post_popen_oserror_kill_reap(self):
        for behavior in ('timeout', 'oserror'):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as td:
                inp, runtime, out = self._paths(td)
                process = FakeProc(behavior)
                with patch.object(consumer.subprocess, 'Popen', return_value=process):
                    self.assertEqual(consumer.run_cli(str(inp), str(runtime), str(out), 0.01), 2)
                receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
                self.assertTrue(process.killed)
                self.assertTrue(receipt['reaped'])
                self.assertEqual(process.waits[-1], 20)
                self.assertEqual(receipt['status'], 'failed')
                self.assertFalse((out / 'preview.html').exists())

    def test_cli_success_parent_revalidates_and_defer_no_preview(self):
        with tempfile.TemporaryDirectory() as td:
            inp, runtime, out = self._paths(td)
            process = FakeProc()
            spawned = {}
            def fake_popen(argv, **_kwargs):
                spawned['argv'] = argv
                spawned['env'] = _kwargs['env']
                result = consumer.select_ui(request(), Predictor('defer'))
                result['network_policy'] = 'deny_socket_connect'
                result['runtime_spec_sha256'] = consumer._sha(json.loads(runtime.read_text()))
                Path(argv[-1]).write_bytes(consumer._bytes(result))
                return process
            with patch.object(consumer.subprocess, 'Popen', side_effect=fake_popen):
                self.assertEqual(consumer.run_cli(str(inp), str(runtime), str(out)), 0)
            self.assertEqual(spawned['argv'][1:4], ['-I', '-B', '-c'])
            self.assertIn(repr(str(Path(consumer.__file__).resolve().parents[2])), spawned['argv'][4])
            self.assertNotIn('PYTHONPATH', spawned['env'])
            self.assertFalse((out / 'preview.html').exists())
            receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['selected_id'], 'defer')
            self.assertTrue(receipt['reaped'])


if __name__ == '__main__':
    unittest.main()
