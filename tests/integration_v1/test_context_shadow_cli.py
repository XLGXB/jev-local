from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main
from jevlocal.integration.context_tools import rank_evidence


class PassiveCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.json'
        self.spec = self.root / 'runtime.json'
        self.sidecar = self.root / 'sidecar.json'
        self.primary = self.root / 'primary.json'
        self.arguments = {'query': 'timeout password=test-secret', 'candidates': [
            {'id': 'b', 'source': 'other.txt', 'text': 'unrelated\r\n'},
            {'id': 'a', 'source': 'log.txt', 'start_line': 7, 'text': 'timeout\r\npassword=test-secret\r\n'}],
            'max_chars': 5000}
        self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
        self.spec.write_text(json.dumps({'mode': 'native', 'backend': 'laya', 'checkpoint': str(self.root),
            'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}), encoding='utf-8')
        self.out, self.err = io.StringIO(), io.StringIO()
        self.started = []
        self.finished = []
        self.closed = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, extra=(), *, unavailable=False, finish=None, start_error=False, cleanup_failed=False):
        test = self
        class Job:
            def finish(self, primary, *, store_dir=None):
                test.assertTrue(test.out.getvalue(), 'primary must be published before finish waits')
                test.finished.append(copy.deepcopy(primary))
                if finish:
                    return finish(primary)
                return {'schema_version': 1, 'status': 'unavailable' if unavailable else 'ok',
                        'shadow': True, 'training_allowed': False, 'execution_authorized': False}
            def close(self):
                test.closed += 1
                if cleanup_failed:
                    return {'closed': False, 'reaped': False, 'temporary_cleaned': False}
        def start(query, candidates, spec, *, timeout_s):
            test.assertFalse(test.out.getvalue(), 'start must precede primary publication')
            test.started.append(copy.deepcopy((query, candidates, spec, timeout_s)))
            if start_error:
                raise OSError('private-startup-error-detail')
            return Job()
        fake = SimpleNamespace(start_shadow=start,
                               unavailable_shadow=lambda *a, **kw: Job())
        with patch.dict(sys.modules, {'jevlocal.integration.context_shadow': fake}), redirect_stdout(self.out), redirect_stderr(self.err):
            try:
                return main(['context', 'rank_evidence', '--input', str(self.input), *extra])
            except SystemExit as exc:
                return exc.code

    def flags(self):
        return ['--shadow-runtime', str(self.spec), '--shadow-output', str(self.sidecar)]

    def test_default_unchanged_and_observer_not_started(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(json.loads(self.out.getvalue()), rank_evidence(**self.arguments))
        self.assertEqual(self.started, [])
        self.assertEqual(self.closed, 0)

    def test_primary_unchanged_redacted_before_start_and_published_before_wait(self):
        self.assertEqual(self.run_cli([*self.flags(), '--output', str(self.primary)]), 0)
        expected = rank_evidence(**self.arguments)
        self.assertEqual(json.loads(self.out.getvalue()), expected)
        self.assertEqual(json.loads(self.primary.read_text(encoding='utf-8')), expected)
        self.assertNotIn('test-secret', json.dumps(self.started))
        self.assertEqual(self.finished, [expected])
        self.assertEqual(self.closed, 1)
        self.assertFalse(json.loads(self.sidecar.read_text(encoding='utf-8'))['training_allowed'])

    def test_unavailable_does_not_change_primary_or_exit_code(self):
        self.assertEqual(self.run_cli(self.flags(), unavailable=True), 0)
        self.assertEqual(json.loads(self.out.getvalue()), rank_evidence(**self.arguments))
        self.assertEqual(json.loads(self.sidecar.read_text())['status'], 'unavailable')
        self.assertEqual(self.closed, 1)

    def test_unexpected_start_error_keeps_primary_and_no_private_message(self):
        self.assertEqual(self.run_cli(self.flags(), unavailable=True, start_error=True), 0)
        self.assertEqual(json.loads(self.out.getvalue()), rank_evidence(**self.arguments))
        self.assertEqual(json.loads(self.sidecar.read_text())['status'], 'unavailable')
        self.assertNotIn('private-startup-error-detail', self.out.getvalue() + self.err.getvalue() + self.sidecar.read_text())
        self.assertEqual(self.closed, 1)

    def test_unreaped_cleanup_does_not_claim_success_or_replace_primary(self):
        self.assertEqual(self.run_cli(self.flags(), unavailable=True, cleanup_failed=True), 2)
        self.assertEqual(json.loads(self.out.getvalue()), rank_evidence(**self.arguments))
        self.assertEqual(json.loads(self.err.getvalue())['status'], 'shadow_cleanup_failed')

    def test_primary_raises_after_start_and_still_closes(self):
        with patch('jevlocal.integration.context_tools._rank_prepared_evidence', side_effect=OSError('private-primary-detail')):
            self.assertEqual(self.run_cli(self.flags()), 2)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.closed, 1)
        self.assertEqual(self.finished, [])
        self.assertFalse(self.sidecar.exists())
        self.assertNotIn('private-primary-detail', self.out.getvalue() + self.err.getvalue())

    def test_partial_staging_write_is_never_published(self):
        from jevlocal.integration.cli import _write_new_atomic
        class BrokenStream:
            def __init__(self, real):
                self.real = real
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.real.close()
            def write(self, data):
                self.real.write(data[:4]); self.real.flush()
                raise OSError('disk failure')
        import os
        original = os.fdopen
        with patch('jevlocal.integration.cli.os.fdopen', side_effect=lambda fd, mode: BrokenStream(original(fd, mode))):
            with self.assertRaises(OSError):
                _write_new_atomic(self.sidecar, {'status': 'ok'})
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(list(self.root.glob('.jevlocal-shadow-*.tmp')), [])

    def test_store_bytes_match_default_and_no_training_labels(self):
        baseline_store, shadow_store = self.root / 'baseline-store', self.root / 'shadow-store'
        expected = rank_evidence(**self.arguments, store_dir=baseline_store)
        self.arguments['store_dir'] = str(shadow_store)
        self.input.write_text(json.dumps(self.arguments), encoding='utf-8')
        self.assertEqual(self.run_cli(self.flags()), 0)
        self.assertEqual(json.loads(self.out.getvalue()), expected)
        self.assertEqual({p.name: p.read_bytes() for p in baseline_store.iterdir()},
                         {p.name: p.read_bytes() for p in shadow_store.iterdir()})
        self.assertNotIn('verified_labels', self.sidecar.read_text())

    def test_primary_failure_closes_observer_without_sidecar_success(self):
        self.primary.write_text('preserved', encoding='utf-8')
        self.assertEqual(self.run_cli([*self.flags(), '--output', str(self.primary)]), 2)
        self.assertEqual(self.primary.read_text(), 'preserved')
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(self.finished, [])
        self.assertEqual(self.closed, len(self.started))

    def test_missing_misused_flags_and_collision_fail_before_start(self):
        cases = [
            ['--shadow-runtime', str(self.spec)],
            ['--shadow-output', str(self.sidecar)],
            ['--shadow-timeout', '1'],
            [*self.flags(), '--native-runtime', str(self.spec)],
            ['--shadow-runtime', str(self.spec), '--shadow-output', str(self.input)],
            [*self.flags(), '--output', str(self.sidecar)],
            [*self.flags(), '--shadow-timeout', '0'],
            [*self.flags(), '--shadow-timeout', 'nan'],
            [*self.flags(), '--shadow-timeout', '61'],
        ]
        for flags in cases:
            with self.subTest(flags=flags):
                self.out.seek(0); self.out.truncate(); self.err.seek(0); self.err.truncate()
                self.assertEqual(self.run_cli(flags), 2)
                self.assertFalse(self.started)
                self.assertFalse(self.sidecar.exists())

    def test_existing_sidecar_is_not_overwritten_or_started(self):
        self.sidecar.write_text('existing', encoding='utf-8')
        self.assertEqual(self.run_cli(self.flags()), 2)
        self.assertFalse(self.started)
        self.assertEqual(self.sidecar.read_text(), 'existing')

    def test_sidecar_write_failure_keeps_one_primary_stdout(self):
        def collision(primary):
            self.sidecar.write_text('concurrent-owner', encoding='utf-8')
            return {'status': 'unavailable'}
        self.assertEqual(self.run_cli(self.flags(), finish=collision), 2)
        self.assertEqual(json.loads(self.out.getvalue()), rank_evidence(**self.arguments))
        self.assertEqual(self.sidecar.read_text(), 'concurrent-owner')
        self.assertEqual(json.loads(self.err.getvalue())['status'], 'shadow_receipt_not_written')
        self.assertEqual(self.closed, 1)

    def test_other_tool_rejects_before_observer(self):
        with redirect_stdout(self.out), redirect_stderr(self.err):
            code = main(['context', 'filter_tool_output', '--input', str(self.input), *self.flags()])
        self.assertEqual(code, 2)
        self.assertFalse(self.sidecar.exists())


if __name__ == '__main__':
    unittest.main()
