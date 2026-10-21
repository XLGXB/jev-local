"""CPU-only contract tests for bounded closed-catalog routing advice."""
import copy
import hashlib
import json
import unittest

from jevlocal.integration.model_routing import advise_model_route


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def model(model_id, *, efforts=('medium',), context=1000, tools=('search',),
          cost=2, latency=100):
    return {'model_id': model_id, 'reasoning_efforts': list(efforts),
            'description': model_id + ' description',
            'context_window_tokens': context, 'tools': None if tools is None else list(tools),
            'estimated_cost': cost, 'estimated_latency_ms': latency}


def request():
    return {'schema_version': 1, 'request_id': 'route:1',
            'current_primary': {'model_id': 'primary', 'reasoning_effort': 'medium'},
            'task': {'summary': 'Review an API change.', 'difficulty': 'moderate',
                     'risk': 'medium', 'context_tokens': 500,
                     'required_tools': ['search']},
            'budget': {'max_estimated_cost': 3, 'max_latency_ms': 200,
                       'cost_unit': 'per-task-units'},
            'catalog': {'source': 'caller fixture',
                        'observed_at': '2026-09-26T12:34:56Z',
                        'models': [model('primary'), model('alternate', cost=1)]},
            'profiles': [
                {'id': 'p', 'model_id': 'primary', 'reasoning_effort': 'medium'},
                {'id': 'a', 'model_id': 'alternate', 'reasoning_effort': 'medium'}]}


class ModelRoutingTests(unittest.TestCase):
    def test_advisory_bound_item_owned_result_and_non_authority(self):
        data = request()
        original = copy.deepcopy(data)
        seen = []

        def factory():
            seen.append('factory')

            def judge(item):
                seen.append(copy.deepcopy(item))
                item['task']['summary'] = 'mutated'
                item['candidates'][0]['description'] = 'mutated'
                return {'selected_profile_id': 'a',
                        'route': {'kind': 'fake', 'execution_authorized': True}}
            return judge

        result = advise_model_route(data, judge_factory=factory)
        self.assertEqual(data, original)
        self.assertEqual(seen[0], 'factory')
        item = seen[1]
        self.assertEqual(set(item), {'request_id', 'request_sha256', 'catalog_sha256',
                                     'current_primary', 'task', 'budget', 'candidates'})
        self.assertEqual(item['request_sha256'], hashlib.sha256(canonical(data)).hexdigest())
        self.assertEqual(item['catalog_sha256'], hashlib.sha256(
            canonical(data['catalog'])).hexdigest())
        self.assertEqual(item['budget'], data['budget'])
        self.assertEqual(result['status'], 'advisory')
        self.assertEqual(result['selected_profile_id'], 'a')
        self.assertEqual(result['recommendation'],
                         {'model_id': 'alternate', 'reasoning_effort': 'medium'})
        self.assertEqual(result['task'], data['task'])
        self.assertEqual(result['budget'], data['budget'])
        self.assertEqual(result['current_primary'], data['current_primary'])
        self.assertEqual(result['catalog_source'], 'caller fixture')
        self.assertEqual(result['catalog_observed_at'], '2026-09-26T12:34:56Z')
        self.assertEqual(result['catalog_provenance'],
                         'caller_supplied_not_host_authenticated')
        self.assertEqual(result['catalog_freshness'], 'not_verified')
        self.assertEqual(result['semantic_quality'], 'unqualified')
        self.assertIs(result['calibrated'], False)
        self.assertIs(result['main_model_review_required'], True)
        for key in ('execution_authorized', 'model_switch_authorized',
                    'provider_change_authorized', 'current_primary_changed'):
            self.assertIs(result[key], False)
        self.assertIn('semantic_closed_set_advice', result['reasons'])
        self.assertIn('task_difficulty_moderate', result['reasons'])
        self.assertIn('task_risk_medium', result['reasons'])

    def test_metadata_only_unknown_without_factory(self):
        result = advise_model_route(request())
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['recommendation'])
        self.assertIsNone(result['selected_profile_id'])
        self.assertEqual(result['route'], {})
        self.assertIn('semantic_unavailable', result['reasons'])

    def test_primary_catalog_pair_need_not_be_advice_candidate(self):
        data = request()
        data['profiles'] = [data['profiles'][1]]
        seen = []

        def factory():
            def judge(item):
                seen.append(item)
                return {'selected_profile_id': 'a', 'route': {'kind': 'fake'}}
            return judge

        result = advise_model_route(data, judge_factory=factory)
        self.assertEqual(result['status'], 'advisory')
        self.assertEqual(result['current_primary'],
                         {'model_id': 'primary', 'reasoning_effort': 'medium'})
        self.assertEqual([candidate['id'] for candidate in seen[0]['candidates']], ['a'])
        self.assertEqual(result['recommendation'],
                         {'model_id': 'alternate', 'reasoning_effort': 'medium'})

    def test_primary_pair_unsupported_by_catalog_rejected_before_factory(self):
        for primary in ({'model_id': 'primary', 'reasoning_effort': 'high'},
                        {'model_id': 'absent', 'reasoning_effort': 'medium'}):
            data = request()
            data['current_primary'] = primary
            data['profiles'] = [data['profiles'][1]]
            calls = []
            with self.subTest(primary=primary), self.assertRaises(ValueError):
                advise_model_route(data, judge_factory=lambda: calls.append('factory'))
            self.assertEqual(calls, [])

    def test_known_constraints_exclude_and_no_factory(self):
        data = request()
        data['task']['context_tokens'] = 1500
        calls = []
        result = advise_model_route(data, judge_factory=lambda: calls.append('factory'))
        self.assertEqual(calls, [])
        self.assertIn('no_compatible_profile', result['reasons'])
        self.assertEqual([p['exclusion_reasons'] for p in result['profiles']],
                         [['context_capacity_insufficient']] * 2)

    def test_known_tool_cost_latency_exclusions(self):
        data = request()
        data['catalog']['models'][0]['tools'] = []
        data['catalog']['models'][1]['estimated_cost'] = 4
        data['catalog']['models'][1]['estimated_latency_ms'] = 201
        result = advise_model_route(data)
        self.assertEqual(result['profiles'][0]['exclusion_reasons'],
                         ['required_tool_unsupported'])
        self.assertEqual(result['profiles'][1]['exclusion_reasons'],
                         ['estimated_cost_over_budget', 'estimated_latency_over_budget'])

    def test_unknown_metadata_explicit_and_not_excluded(self):
        data = request()
        m = data['catalog']['models'][1]
        for field in ('context_window_tokens', 'tools', 'estimated_cost',
                      'estimated_latency_ms'):
            m[field] = None
        seen = []

        def factory():
            def judge(item):
                seen.append(item)
                return {'selected_profile_id': 'a', 'route': {'kind': 'fake'}}
            return judge

        result = advise_model_route(data, judge_factory=factory)
        unknowns = ['context_capacity_unknown', 'tool_support_unknown',
                    'cost_unknown', 'latency_unknown']
        self.assertEqual(result['profiles'][1]['unknowns'], unknowns)
        self.assertTrue(result['profiles'][1]['eligible'])
        self.assertEqual(seen[0]['candidates'][1]['unknowns'], unknowns)
        self.assertIn('metadata_unknown', result['reasons'])
        data['task']['context_tokens'] = None
        data['task']['required_tools'] = []
        data['budget']['max_estimated_cost'] = None
        data['budget']['max_latency_ms'] = None
        self.assertEqual(advise_model_route(data)['profiles'][1]['unknowns'], [])

    def test_abstention_is_unknown(self):
        result = advise_model_route(request(), judge_factory=lambda: lambda _: {
            'selected_profile_id': None, 'route': {'kind': 'fake'}})
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['recommendation'])
        self.assertIn('semantic_abstained', result['reasons'])

    def test_invalid_shapes_pairs_and_strict_types_precede_factory(self):
        changes = [
            lambda d: d.update(extra=True),
            lambda d: d.update(schema_version=True),
            lambda d: d['task'].update(context_tokens=True),
            lambda d: d['budget'].update(max_latency_ms=True),
            lambda d: d['budget'].update(max_estimated_cost=True),
            lambda d: d['budget'].update(max_estimated_cost=float('nan')),
            lambda d: d['budget'].update(max_estimated_cost=float('inf')),
            lambda d: d['budget'].update(max_estimated_cost=10**400),
            lambda d: d['budget'].update(cost_unit=''),
            lambda d: d['task'].update(required_tools=['search', 'search']),
            lambda d: d['catalog']['models'][0].update(
                reasoning_efforts=['medium', 'medium']),
            lambda d: d['catalog']['models'][0].update(
                context_window_tokens=True),
            lambda d: d['catalog']['models'][0].update(estimated_cost=float('nan')),
            lambda d: d['catalog']['models'][0].update(tools=['search', 'search']),
            lambda d: d['catalog']['models'][0].update(estimated_latency_ms=False),
            lambda d: d['catalog']['models'].append(copy.deepcopy(d['catalog']['models'][0])),
            lambda d: d['profiles'].append(copy.deepcopy(d['profiles'][0])),
            lambda d: d['profiles'][1].update(model_id='primary'),
            lambda d: d['profiles'][1].update(reasoning_effort='ultra'),
            lambda d: d['profiles'][0].update(id='__abstain__'),
            lambda d: d['current_primary'].update(reasoning_effort='ultra'),
            lambda d: d['catalog'].update(observed_at='yesterday'),
            lambda d: d['catalog'].update(observed_at='2026-02-30T00:00:00Z'),
            lambda d: d['task'].update(summary='x' * 8193),
            lambda d: d['profiles'].append(('tuple',)),
        ]
        for change in changes:
            data = request()
            change(data)
            calls = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                advise_model_route(data, judge_factory=lambda: calls.append('factory'))
            self.assertEqual(calls, [])

    def test_input_budget_and_non_json_value(self):
        data = request()
        data['task']['summary'] = 'x' * 8192
        data['catalog']['source'] = 's' * 512
        data['catalog']['models'] = [model('m' + str(n), tools=('a' * 64,))
                                     for n in range(32)]
        data['current_primary']['model_id'] = 'm0'
        data['profiles'] = [{'id': 'p', 'model_id': 'm0',
                             'reasoning_effort': 'medium'}]
        data['task']['required_tools'] = []
        # Repeated bounded descriptions make the canonical request exceed 128 KiB.
        for m in data['catalog']['models']:
            m['description'] = 'x' * 2048
        data['task']['summary'] = 'x' * 8192
        self.assertLess(len(canonical(data)), 128 * 1024)
        data['task']['required_tools'] = ['t' * 64 for _ in range(32)]
        data['catalog']['models'][0]['tools'] = ['u' * 64 for _ in range(32)]
        # Exact 128 KiB bound is also exercised by unconstrained input shapes.
        data['extra'] = 'z' * (128 * 1024)
        with self.assertRaises(ValueError):
            advise_model_route(data)

    def test_profile_count_reserves_abstain_protocol_slot(self):
        data = request()
        data['catalog']['models'] = [model('m' + str(n)) for n in range(15)]
        data['profiles'] = [{'id': 'p' + str(n), 'model_id': 'm' + str(n),
                             'reasoning_effort': 'medium'} for n in range(15)]
        data['current_primary']['model_id'] = 'm0'
        self.assertEqual(len(advise_model_route(data)['profiles']), 15)
        data['catalog']['models'].append(model('m15'))
        data['profiles'].append({'id': 'p15', 'model_id': 'm15',
                                 'reasoning_effort': 'medium'})
        calls = []
        with self.assertRaises(ValueError):
            advise_model_route(data, judge_factory=lambda: calls.append('factory'))
        self.assertEqual(calls, [])
        data = request()
        data['task']['required_tools'] = ('search',)
        with self.assertRaises(ValueError):
            advise_model_route(data)

    def test_reply_rejects_out_of_set_and_unbounded_metadata(self):
        replies = [
            {'selected_profile_id': 'other', 'route': {'kind': 'fake'}},
            {'selected_profile_id': '__abstain__', 'route': {'kind': 'fake'}},
            {'selected_profile_id': True, 'route': {'kind': 'fake'}},
            {'selected_profile_id': 'p', 'route': {}},
            {'selected_profile_id': 'p', 'route': {'x': float('nan')}},
            {'selected_profile_id': 'p', 'route': {'x': 'x' * (256 * 1024)}},
            {'selected_profile_id': 'p', 'route': {'x': set()}},
            {'selected_profile_id': 'p', 'route': {'x': 1}, 'extra': True},
        ]
        for reply in replies:
            with self.subTest(reply=type(reply['route'])), self.assertRaises(ValueError):
                advise_model_route(request(), judge_factory=lambda: lambda _: reply)

        data = request()
        data['catalog']['models'][1]['context_window_tokens'] = 100
        with self.assertRaises(ValueError):
            advise_model_route(data, judge_factory=lambda: lambda _: {
                'selected_profile_id': 'a', 'route': {'kind': 'fake'}})

    def test_factory_failure_once_without_private_error_echo_or_fallback(self):
        calls = []

        def factory():
            calls.append('factory')
            raise RuntimeError('private token: secret')

        with self.assertRaises(ValueError) as caught:
            advise_model_route(request(), judge_factory=factory)
        self.assertEqual(calls, ['factory'])
        self.assertEqual(str(caught.exception), 'semantic judge failed')
        self.assertNotIn('secret', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
