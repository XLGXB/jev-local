"""CPU-only filesystem fixtures for F06 repository navigation."""
from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from jevlocal.integration import repository_navigation as navigation
from jevlocal.integration.repository_navigation import navigate_repository


class RepositoryNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'docs').mkdir()
        (self.root / 'docs' / 'note.txt').write_text('public fixture', encoding='utf-8')
        (self.root / '.env').write_text('not to read', encoding='utf-8')
        self.request = {
            'schema_version': 1, 'request_id': 'f06-fixture',
            'repository_root': self.root.as_posix(), 'current_directory': '.',
            'query': 'Find note', 'candidate_names': None, 'max_hops': 2,
        }

    def test_two_hops_and_owned_snapshot(self):
        calls = []

        def factory():
            def judge(prepared):
                calls.append(prepared)
                wanted = 'docs' if len(calls) == 1 else 'note.txt'
                selected = next(item for item in prepared['candidates']
                                if item['name'] == wanted)
                return {'selected_candidate_id': selected['id'],
                        'route': {'fixture': True}}
            return judge

        report = navigate_repository(self.request, judge_factory=factory)
        self.assertEqual(report['status'], 'file_selected')
        self.assertEqual(report['selected_file'], 'docs/note.txt')
        self.assertEqual(len(report['hops']), 2)
        self.assertEqual([call['hop_index'] for call in calls], [1, 2])
        self.assertEqual(set(calls[0]), {'request_id', 'hop_index', 'query',
                                        'current_directory', 'candidates',
                                        'snapshot_sha256'})
        self.assertEqual(set(calls[0]['candidates'][0]),
                         {'id', 'name', 'relative_path', 'kind'})
        self.assertNotIn('.env', [item['name'] for item in report['hops'][0]['candidates']])
        self.assertEqual(report['hops'][0]['excluded_count'], 1)
        self.assertFalse(report['execution_authorized'])
        self.assertFalse(report['file_write_authorized'])

    def test_no_judge_and_capacity_need_explicit_subset(self):
        result = navigate_repository(self.request)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'semantic_unavailable')
        self.assertEqual(len(result['hops']), 1)
        for number in range(16):
            (self.root / f'item{number}.txt').touch()
        result = navigate_repository(self.request, judge_factory=lambda: self.fail('judge called'))
        self.assertEqual(result['status'], 'requires_candidate_subset')
        self.assertEqual(result['reason'], 'candidate_capacity_exceeded')
        explicit = dict(self.request, candidate_names=['docs'])
        result = navigate_repository(explicit)
        self.assertEqual(result['hops'][0]['scope'], 'explicit_subset')
        self.assertEqual([item['name'] for item in result['hops'][0]['candidates']], ['docs'])

    def test_abstain_depth_empty_and_reject_bad_selection(self):
        abstain = navigate_repository(self.request, judge_factory=lambda: (
            lambda item: {'selected_candidate_id': None, 'route': {'fixture': True}}))
        self.assertEqual(abstain['status'], 'unknown')
        self.assertEqual(abstain['reason'], 'semantic_abstained')
        shallow = dict(self.request, max_hops=1, candidate_names=['docs'])
        depth = navigate_repository(shallow, judge_factory=lambda: (
            lambda item: {'selected_candidate_id': item['candidates'][0]['id'],
                          'route': {'fixture': True}}))
        self.assertEqual(depth['status'], 'depth_limit')
        self.assertEqual(depth['next_directory'], 'docs')
        empty_dir = self.root / 'empty'
        empty_dir.mkdir()
        empty = navigate_repository(dict(self.request, current_directory='empty'))
        self.assertEqual(empty['status'], 'empty_directory')
        with self.assertRaises(ValueError):
            navigate_repository(self.request, judge_factory=lambda: (
                lambda item: {'selected_candidate_id': 'outside',
                              'route': {'fixture': True}}))

    def test_stale_snapshot_and_invalid_explicit_names(self):
        def mutate():
            def judge(item):
                (self.root / 'docs' / 'new.txt').touch()
                return {'selected_candidate_id': item['candidates'][0]['id'],
                        'route': {'fixture': True}}
            return judge
        with self.assertRaisesRegex(ValueError, 'path evidence changed'):
            navigate_repository(dict(self.request, candidate_names=['docs']),
                                judge_factory=mutate)
        for name in ('.env', '../docs', 'CON', 'token.txt', 'bad\\path'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                navigate_repository(dict(self.request, candidate_names=[name]))

    def test_never_opens_candidate_body_and_rejects_root_escape(self):
        with patch.object(Path, 'open', side_effect=AssertionError('file body opened')):
            report = navigate_repository(self.request)
        self.assertEqual(report['status'], 'unknown')
        for current in ('../outside', 'docs/../docs', '/outside', '.env'):
            with self.subTest(current=current), self.assertRaises(ValueError):
                navigate_repository(dict(self.request, current_directory=current))
        with self.assertRaises(ValueError):
            navigate_repository(dict(self.request, candidate_names=['missing']))

    def test_link_exclusion(self):
        link = self.root / 'linked'
        try:
            link.symlink_to(self.root / 'docs', target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f'symlink unavailable: {exc}')
        observed = navigate_repository(self.request)
        self.assertNotIn('linked', [item['name'] for item in observed['hops'][0]['candidates']])
        with self.assertRaises(ValueError):
            navigate_repository(dict(self.request, candidate_names=['linked']))

    def test_scan_limit(self):
        for number in range(260):
            (self.root / f'bulk{number:03}.txt').touch()
        report = navigate_repository(self.request, judge_factory=lambda: self.fail('judge called'))
        self.assertEqual(report['status'], 'requires_candidate_subset')
        self.assertEqual(report['reason'], 'discovery_scan_limit_exceeded')
        self.assertEqual(report['hops'][0]['scanned_count'], 256)
        selected = navigate_repository(dict(self.request, candidate_names=['docs']))
        self.assertEqual(selected['hops'][0]['scope'], 'explicit_subset')

    def test_root_component_preflight_before_filesystem(self):
        for component in ('CON', 'bad.', 'bad ', '.env.', 'credentials.', 'bad\x7f'):
            with self.subTest(component=component):
                unsafe = dict(self.request,
                              repository_root=(self.root / component).as_posix())
                with patch.object(navigation, '_local_path',
                                  side_effect=AssertionError('filesystem touched')):
                    with self.assertRaises(ValueError):
                        navigate_repository(unsafe)

    def test_plain_json_preflight_before_serialization(self):
        nonplain = dict(self.request, candidate_names=('docs',))
        with patch.object(navigation, '_serialized',
                          side_effect=AssertionError('serialized nonplain')):
            with self.assertRaises(ValueError):
                navigate_repository(nonplain)

    def test_prepared_budget_and_snapshot_before_judge_creation(self):
        source = navigation._serialized
        def tiny_prepared(value, label, limit):
            if label == 'prepared navigation':
                return source(value, label, 1)
            return source(value, label, limit)
        with patch.object(navigation, '_serialized', side_effect=tiny_prepared):
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                navigate_repository(self.request,
                                    judge_factory=lambda: self.fail('judge created'))
        with patch.object(navigation, '_stable',
                          side_effect=ValueError('path evidence changed')):
            with self.assertRaisesRegex(ValueError, 'path evidence changed'):
                navigate_repository(self.request,
                                    judge_factory=lambda: self.fail('judge created'))

    def test_name_and_current_directory_length_before_filesystem(self):
        with patch.object(navigation, '_local_path',
                          side_effect=AssertionError('filesystem touched')):
            with self.assertRaises(ValueError):
                navigate_repository(dict(self.request,
                                         candidate_names=['n' * 256]))
            with self.assertRaises(ValueError):
                navigate_repository(dict(self.request,
                                         current_directory='d' * 4097))

    def test_relative_path_length_rejects_without_candidate_stat(self):
        root = Path('D:/fixture')
        current = root.joinpath(*(['a' * 255] * 16))
        with patch.object(navigation, '_checked_info',
                          side_effect=AssertionError('candidate stat touched')):
            with self.assertRaisesRegex(ValueError, 'relative_path length'):
                navigation._candidate(root, current, 'z')

    @unittest.skipUnless(os.name == 'nt', 'Windows lease dispatch')
    def test_windows_lease_failure_does_not_fall_back_to_path_observation(self):
        class RejectedLease:
            def __init__(self, root):
                self.root = root

            def __enter__(self):
                raise ValueError('native lease rejected')

            def __exit__(self, *unused):
                pass

        with patch.object(navigation, '_new_windows_lease', RejectedLease), \
             patch.object(navigation, '_local_path',
                          side_effect=AssertionError('path fallback')), \
             patch.object(navigation.os, 'scandir',
                          side_effect=AssertionError('path enumeration')):
            with self.assertRaisesRegex(ValueError, 'native lease rejected'):
                navigate_repository(self.request)

    @unittest.skipUnless(os.name == 'nt', 'Windows lease dispatch')
    def test_windows_one_lease_closes_on_semantic_exception(self):
        leases = []
        identity = {'device': 1, 'inode': 2, 'size': 0,
                    'mtime_ns': 3, 'ctime_ns': 4}

        class FakeLease:
            def __init__(self, root):
                self.root = root
                self.closed = False
                leases.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                self.closed = True

            def chain_identities(self, current):
                return [identity]

            def iter_names(self, current):
                return iter(['docs'])

            def child_metadata(self, current, name):
                return {'kind': 'directory', 'identity': identity}

        with patch.object(navigation, '_new_windows_lease', FakeLease), \
             patch.object(navigation, '_local_path',
                          side_effect=AssertionError('path fallback')), \
             patch.object(navigation, '_check_components',
                          side_effect=AssertionError('path observation')), \
             patch.object(navigation.os, 'scandir',
                          side_effect=AssertionError('path enumeration')):
            with self.assertRaisesRegex(ValueError, 'semantic judge failed'):
                navigate_repository(self.request, judge_factory=lambda: (
                    lambda item: (_ for _ in ()).throw(RuntimeError('fixture'))))
        self.assertEqual(len(leases), 1)
        self.assertTrue(leases[0].closed)

    @unittest.skipUnless(os.name == 'nt', 'Windows native lease')
    def test_windows_real_lease_never_uses_path_snapshot(self):
        with patch.object(navigation, '_local_path',
                          side_effect=AssertionError('path admission')), \
             patch.object(navigation, '_check_components',
                          side_effect=AssertionError('path ancestor scan')), \
             patch.object(navigation, '_checked_info',
                          side_effect=AssertionError('path candidate stat')), \
             patch.object(navigation.os, 'scandir',
                          side_effect=AssertionError('path enumeration')):
            report = navigate_repository(self.request)
        self.assertEqual(report['path_concurrency_guarantee'],
                         'windows_handle_relative_no_reparse')
        self.assertEqual(report['status'], 'unknown')
        self.assertEqual([item['name'] for item in report['hops'][0]['candidates']],
                         ['docs'])


if __name__ == '__main__':
    unittest.main()
