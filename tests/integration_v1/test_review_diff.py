"""F04 actual, caller-supplied snapshot diff and non-authorizing review floors."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jevlocal.integration.review_diff import review_diff


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewDiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.before = self.ws / 'baseline.txt'
        self.before.write_text('one\ntwo\nthree\n', encoding='utf-8')
        self.current = self.ws / 'src' / 'a.py'
        self.current.parent.mkdir()
        self.current.write_text('one\ntwo changed\nthree\n', encoding='utf-8')
        self.contract_file = self.ws / 'contract.txt'
        self.contract_file.write_text('Review auth and storage.\n', encoding='utf-8')
        self.calls = []

    def request(self, contracts=None):
        return {'schema_version': 1, 'workspace': str(self.ws),
                'request_id': 'r1', 'changes': [
                    {'path': 'src/a.py',
                     'before': {'path': str(self.before), 'sha256': sha(self.before)},
                     'after': {'path': str(self.current), 'sha256': sha(self.current)}}],
                'contracts': [] if contracts is None else contracts}

    def contract(self, category='security', mandatory_review=False, reviewer='human'):
        return {'id': 'c1', 'path': str(self.contract_file),
                'sha256': sha(self.contract_file), 'applies_to': ['src/a.py'],
                'category': category, 'mandatory_review': mandatory_review,
                'reviewer': reviewer}

    def factory(self, score=0, categories=None):
        categories = categories or dict.fromkeys(
            ('security', 'permissions', 'interface', 'persistence'), False)

        def make():
            self.calls.append('factory')

            def judge(items):
                self.calls.append(copy.deepcopy(items))
                return {'decisions': {item['id']: {'risk_score': score,
                         'categories': categories} for item in items}, 'route': {}}
            return judge
        return make

    def test_real_lines_hashes_item_exactness_and_no_mutation(self):
        request = self.request([self.contract(category='general', reviewer='main_model')])
        input_copy = copy.deepcopy(request)
        before_bytes = self.before.read_bytes()
        after_bytes = self.current.read_bytes()
        result = review_diff(request, judge_factory=self.factory())
        self.assertEqual(request, input_copy)
        self.assertEqual(self.before.read_bytes(), before_bytes)
        self.assertEqual(self.current.read_bytes(), after_bytes)
        self.assertEqual(result['status'], 'reviewed')
        h = result['changes'][0]['hunks'][0]
        self.assertEqual(h['status'], 'advisory')
        self.assertEqual(h['risk_score'], 0)
        self.assertTrue(h['review_required'])
        self.assertTrue(h['main_model_review_required'])
        self.assertEqual(h['hunk']['before_start'], 1)
        self.assertEqual(h['hunk']['after_start'], 1)
        self.assertEqual([(line['kind'], line['before_line'], line['after_line'])
                          for line in h['hunk']['lines']],
                         [('context', 1, 1), ('remove', 2, None),
                          ('add', None, 2), ('context', 3, 3)])
        self.assertEqual(h['hunk']['lines'][1]['text'],
                         before_bytes.decode('utf-8').splitlines(keepends=True)[1])
        self.assertEqual(h['hunk']['lines'][2]['text'],
                         after_bytes.decode('utf-8').splitlines(keepends=True)[1])
        item = self.calls[1][0]
        self.assertEqual(set(item), {'id', 'file_path', 'change_sha256',
                                     'hunk_sha256', 'hunk', 'contracts'})
        self.assertEqual(item['hunk'], h['hunk'])
        self.assertEqual(item['contracts'][0]['text'],
                         self.contract_file.read_bytes().decode('utf-8'))
        for key in ('execution_authorized', 'action_authorized', 'merge_authorized',
                    'review_complete'):
            self.assertIs(result[key], False)
        self.assertEqual(result['semantic_quality'], 'unqualified')

    def test_four_sensitive_floors_survive_low_model_score(self):
        for category in ('security', 'permissions', 'interface', 'persistence'):
            with self.subTest(category=category):
                self.calls.clear()
                result = review_diff(self.request([self.contract(category=category)]),
                                     judge_factory=self.factory())
                h = result['changes'][0]['hunks'][0]
                self.assertEqual(h['risk_score'], 0)
                self.assertTrue(h['review_required'])
                self.assertTrue(h['main_model_review_required'])
                self.assertTrue(h['human_review_required'])
                self.assertEqual(h['obligations'][0]['category'], category)
                self.assertEqual(len(self.calls), 2)

    def test_missing_contract_unknown_and_lazy_factory(self):
        result = review_diff(self.request(), judge_factory=self.factory())
        h = result['changes'][0]['hunks'][0]
        self.assertEqual(h['status'], 'unknown')
        self.assertIsNone(h['risk_score'])
        self.assertIsNone(h['categories'])
        self.assertIn('missing_applicable_contract', h['reasons'])
        self.assertEqual(self.calls, [])

    def test_explicit_general_mandatory_obligation_survives_low_score(self):
        contract = self.contract(category='general', mandatory_review=True)
        result = review_diff(self.request([contract]), judge_factory=self.factory())
        h = result['changes'][0]['hunks'][0]
        self.assertEqual(h['risk_score'], 0)
        self.assertTrue(h['human_review_required'])
        self.assertEqual(h['obligations'][0]['id'], 'c1')
        self.assertTrue(h['obligations'][0]['mandatory_review'])

    def test_high_score_adds_escalation_to_general_main_model_contract(self):
        contract = self.contract(category='general', reviewer='main_model')
        h = review_diff(self.request([contract]),
                        judge_factory=self.factory(score=2))['changes'][0]['hunks'][0]
        self.assertTrue(h['human_review_required'])
        self.assertEqual(h['obligations'], [])
        self.assertIn('semantic_human_escalation', h['reasons'])

    def test_creation_deletion_and_exact_zero_side_positions(self):
        create = self.request([self.contract()])
        create['changes'][0]['before'] = None
        created = review_diff(create)['changes'][0]['hunks'][0]['hunk']
        self.assertEqual((created['before_start'], created['before_count']), (0, 0))
        self.assertEqual((created['after_start'], created['after_count']), (1, 3))
        delete = self.request([self.contract()])
        delete['changes'][0]['after'] = None
        self.current.unlink()
        deleted = review_diff(delete)['changes'][0]['hunks'][0]['hunk']
        self.assertEqual((deleted['before_start'], deleted['before_count']), (1, 3))
        self.assertEqual((deleted['after_start'], deleted['after_count']), (0, 0))

    def test_deleted_nested_parent_is_valid_absent_target(self):
        delete = self.request([self.contract()])
        delete['changes'][0]['after'] = None
        self.current.unlink()
        self.current.parent.rmdir()
        h = review_diff(delete)['changes'][0]['hunks'][0]
        self.assertEqual((h['hunk']['after_start'], h['hunk']['after_count']), (0, 0))

    def test_windows_case_alias_of_actual_after_path(self):
        alias = self.request([self.contract()])
        alias['changes'][0]['after']['path'] = str(self.current).replace(
            'src\\a.py', 'SRC\\A.PY')
        self.assertEqual(review_diff(alias)['changes'][0]['hunks'][0]['status'],
                         'unknown')

    def test_empty_file_create_delete_are_unknown_without_model(self):
        self.current.write_bytes(b'')
        request = self.request([self.contract()])
        request['changes'][0]['before'] = None
        h = review_diff(request, judge_factory=self.factory())['changes'][0]['hunks'][0]
        self.assertEqual(h['hunk']['lines'], [])
        self.assertEqual(h['status'], 'unknown')
        self.assertIn('empty_file_structural_change', h['reasons'])
        self.assertEqual(self.calls, [])
        self.before.write_bytes(b'')
        deletion = self.request([self.contract()])
        deletion['changes'][0]['after'] = None
        self.current.unlink()
        h = review_diff(deletion, judge_factory=self.factory())['changes'][0]['hunks'][0]
        self.assertEqual(h['hunk']['lines'], [])
        self.assertEqual(h['status'], 'unknown')
        self.assertIn('empty_file_structural_change', h['reasons'])
        self.assertEqual(self.calls, [])

    def test_tampered_current_contract_and_path_rejected_before_factory(self):
        variants = []
        bad = self.request([self.contract()]); bad['changes'][0]['after']['sha256'] = '0' * 64; variants.append(bad)
        bad = self.request([self.contract()]); bad['contracts'][0]['sha256'] = '0' * 64; variants.append(bad)
        bad = self.request([self.contract()]); bad['changes'][0]['path'] = '../src/a.py'; variants.append(bad)
        bad = self.request([self.contract()]); bad['changes'][0]['after']['path'] = str(self.before); variants.append(bad)
        bad = self.request([self.contract()]); bad['contracts'][0]['applies_to'] = ['src/other.py']; variants.append(bad)
        bad = self.request([self.contract()]); bad['changes'][0]['path'] = 'src/CON.txt'; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                self.calls.clear()
                with self.assertRaises(ValueError):
                    review_diff(bad, judge_factory=self.factory())
                self.assertEqual(self.calls, [])

    def test_semantic_reply_exact_ids_and_types_or_error(self):
        request = self.request([self.contract()])
        for decision in ({}, {'foreign': {'risk_score': 0, 'categories': {}}},
                         {'risk_score': True, 'categories': dict.fromkeys(
                             ('security', 'permissions', 'interface', 'persistence'), False)},
                         {'risk_score': 0, 'categories': {'security': False}}):
            def factory():
                def judge(items):
                    return {'decisions': ({items[0]['id']: decision}
                                          if 'risk_score' in decision else decision),
                            'route': {}}
                return judge
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                review_diff(request, judge_factory=factory)

    def test_input_and_semantic_item_budgets_precede_factory(self):
        self.contract_file.write_text('x' * 32769, encoding='utf-8')
        with self.assertRaises(ValueError):
            review_diff(self.request([self.contract()]), judge_factory=self.factory())
        self.assertEqual(self.calls, [])

    def test_oversize_source_rejected_before_factory(self):
        self.current.write_bytes(b'x' * (256 * 1024 + 1))
        with self.assertRaises(ValueError):
            review_diff(self.request([self.contract()]), judge_factory=self.factory())
        self.assertEqual(self.calls, [])

    def test_blank_contract_rejected_before_factory(self):
        for text in ('', '  \t\r\n  '):
            self.contract_file.write_text(text, encoding='utf-8')
            self.calls.clear()
            with self.subTest(text=text), self.assertRaises(ValueError):
                review_diff(self.request([self.contract()]),
                            judge_factory=self.factory())
            self.assertEqual(self.calls, [])

    def test_judge_exception_is_rejected_without_fallback(self):
        def factory():
            def judge(_items):
                raise RuntimeError('native unavailable')
            return judge
        with self.assertRaisesRegex(ValueError, 'semantic judge failed'):
            review_diff(self.request([self.contract()]), judge_factory=factory)


if __name__ == '__main__':
    unittest.main()
