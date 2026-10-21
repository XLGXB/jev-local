import copy
import hashlib
import unittest

from jevlocal.integration.claim_support import verify_claim_support


def request():
    return {'schema_version': 1, 'request_id': 'r1', 'claim': 'Exact claim',
            'sources': [{'id': 's1', 'source_ref': 'Caller label', 'start_line': 7,
                         'text': ' quoted\nexcerpt '}]}


def answer(item, label='supports'):
    probs = dict.fromkeys(('supports', 'contradicts', 'mixed', 'insufficient'), 0.0)
    probs[label] = 1.0
    return {'decisions': {item['source_id']: {'label': label, 'probabilities': probs,
                                                  'p_selected': 1.0}},
            'route': {'kind': 'fake'}}


class ClaimSupportTests(unittest.TestCase):
    def test_source_binding_and_exact_quote(self):
        seen = []
        result = verify_claim_support(request(), judge_factory=lambda: lambda items:
                                      (seen.extend(copy.deepcopy(items)) or answer(items[0])))
        item = seen[0]
        self.assertEqual(set(item), {'request_id', 'source_id', 'claim', 'source_ref',
                                     'start_line', 'end_line', 'text', 'text_sha256'})
        self.assertEqual(item['end_line'], 8)
        self.assertEqual(item['text_sha256'], hashlib.sha256(item['text'].encode()).hexdigest())
        citation = result['source_results'][0]['citation']
        self.assertEqual(citation['text'], ' quoted\nexcerpt ')
        self.assertEqual(result['status'], 'supported')

    def test_labels_conflict_and_judge_absence(self):
        for label, expected in [('supports', 'supported'), ('contradicts', 'contradicted'),
                                ('mixed', 'conflicting'), ('insufficient', 'unknown')]:
            result = verify_claim_support(request(), judge_factory=lambda: lambda items:
                                          answer(items[0], label))
            self.assertEqual(result['status'], expected)
        result = verify_claim_support(request())
        self.assertEqual(result['source_results'][0]['label'], 'unknown')
        self.assertEqual(result['reason'], 'semantic_unavailable')

    def test_empty_and_invalid_do_not_create_factory(self):
        calls = []
        empty = request(); empty['sources'] = []
        self.assertEqual(verify_claim_support(empty, judge_factory=lambda: calls.append(1))['reason'],
                         'no_sources')
        bad = request(); bad['sources'][0]['extra'] = True
        with self.assertRaises(ValueError):
            verify_claim_support(bad, judge_factory=lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_invalid_judge_reply_rejected(self):
        with self.assertRaises(ValueError):
            verify_claim_support(request(), judge_factory=lambda: lambda items: {
                'decisions': {}, 'route': {'kind': 'fake'}})

    def test_input_validation_matrix_precedes_factory(self):
        changes = [
            lambda d: d.update(schema_version=True),
            lambda d: d['sources'][0].update(start_line=True),
            lambda d: d['sources'][0].update(start_line=1_000_000_001),
            lambda d: d.update(request_id='bad id'),
            lambda d: d['sources'][0].update(extra=1),
            lambda d: d.update(sources=[{'id': 's1', 'source_ref': 'x',
                                         'start_line': 1, 'text': 'x'},
                                        {'id': 's1', 'source_ref': 'y',
                                         'start_line': 2, 'text': 'y'}]),
            lambda d: d.update(claim=' '),
            lambda d: d.update(claim='x' * 8193),
            lambda d: d['sources'][0].update(text='x' * 8193),
            lambda d: d['sources'][0].update(source_ref='x' * 2049),
            lambda d: d['sources'][0].update(text='\ud800'),
            lambda d: d['sources'][0].update(start_line=1_000_000_000,
                                              text='line1\nline2'),
            lambda d: d.update(sources=[{'id': f's{i}', 'source_ref': 'r',
                                         'start_line': 1, 'text': 'x'} for i in range(16)]),
            lambda d: d.update(extra=True),
        ]
        for change in changes:
            data = request(); change(data); calls = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_claim_support(data, judge_factory=lambda: calls.append(1))
            self.assertEqual(calls, [])

    def test_input_byte_limit_and_plain_json(self):
        data = request(); data['claim'] = 'x' * 8192
        data['sources'][0]['source_ref'] = 'r' * 2048
        data['sources'][0]['text'] = 'y' * 8192
        data['other'] = 'z' * (131072)
        with self.assertRaises(ValueError):
            verify_claim_support(data)
        data = request(); data['sources'] = (data['sources'][0],)
        with self.assertRaises(ValueError):
            verify_claim_support(data)

    def test_judge_snapshot_once_order_conflict_and_caller_isolation(self):
        data = request()
        data['sources'].append({'id': 's2', 'source_ref': 'ref2', 'start_line': 20,
                                'text': 'second'})
        before = copy.deepcopy(data); calls = []

        def factory():
            calls.append('factory')
            def judge(items):
                calls.append('judge')
                self.assertEqual([i['source_id'] for i in items], ['s1', 's2'])
                items[0]['text'] = 'tampered'
                return {'decisions': {items[0]['source_id']: {
                    'label': 'supports', 'probabilities': {'supports': 1.0,
                    'contradicts': 0.0, 'mixed': 0.0, 'insufficient': 0.0},
                    'p_selected': 1.0}, items[1]['source_id']: {
                    'label': 'contradicts', 'probabilities': {'supports': 0.0,
                    'contradicts': 1.0, 'mixed': 0.0, 'insufficient': 0.0},
                    'p_selected': 1.0}}, 'route': {'kind': 'fake'}}
            return judge

        result = verify_claim_support(data, judge_factory=factory)
        self.assertEqual(calls, ['factory', 'judge'])
        self.assertEqual(data, before)
        self.assertEqual([r['source_id'] for r in result['source_results']], ['s1', 's2'])
        self.assertEqual(result['source_results'][0]['citation']['text'], ' quoted\nexcerpt ')
        self.assertEqual(result['status'], 'conflicting')

    def test_judge_reply_matrix_strictly_rejects(self):
        base = answer({'source_id': 's1'})
        alterations = [
            lambda d: d['decisions']['s1']['probabilities'].update(supports=True),
            lambda d: d['decisions']['s1']['probabilities'].update(supports=float('nan')),
            lambda d: d['decisions']['s1']['probabilities'].update(supports=float('inf')),
            lambda d: d['decisions']['s1']['probabilities'].update(supports=.4),
            lambda d: d['decisions']['s1'].update(p_selected=.5),
            lambda d: d['decisions']['s1']['probabilities'].update(unknown=0.0),
            lambda d: d['decisions']['s1'].update(extra=1),
            lambda d: d['route'].update(extra=set()),
        ]
        for alter in alterations:
            reply = copy.deepcopy(base); alter(reply)
            with self.subTest(alter=alter), self.assertRaises(ValueError):
                verify_claim_support(request(), judge_factory=lambda: lambda _: reply)

    def test_report_budget_rejected(self):
        data = request()
        data['sources'][0]['source_ref'] = 'r' * 2048
        data['sources'][0]['text'] = 'x' * 8192
        with self.assertRaises(ValueError):
            verify_claim_support(data, judge_factory=lambda: lambda items: {
                'decisions': {items[0]['source_id']: {'label': 'supports',
                    'probabilities': {'supports': 1.0, 'contradicts': 0.0,
                    'mixed': 0.0, 'insufficient': 0.0}, 'p_selected': 1.0}},
                'route': {'kind': 'fake', 'metadata': 'm' * 255000}})

    def test_reply_snapshot_rejects_subclass_keys_and_preserves_cause(self):
        reply = answer({'source_id': 's1'})
        reply['decisions'] = {type('SourceKey', (str,), {} )('s1'):
                              reply['decisions']['s1']}
        with self.assertRaises(ValueError) as caught:
            verify_claim_support(request(), judge_factory=lambda: lambda _: reply)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(str(caught.exception), 'semantic judge failed')


if __name__ == '__main__':
    unittest.main()
