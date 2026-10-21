from contextlib import redirect_stdout, redirect_stderr
import asyncio
import copy
import io
import json
import os
import sys
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main
from jevlocal.integration.context_tools import ModelRouteError, restore_lines
from jevlocal.integration.native_context import NativeContextScorer, build_native_scorer, invoke_context


class FixturePredictor:
    def __init__(self, mutate=None):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.requests = []
        self.mutate = mutate

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        p = 0.8 if request['state']['candidate']['id'] == 'useful' else 0.2
        result = {'runtime_fingerprint': self.runtime.fingerprint, 'inference_total_ms': 0.0,
            'answers': {'relevance': {'selected_key': 'true' if p > 0.5 else 'false',
                'probabilities': {'false': 1 - p, 'true': p}, 'p_selected': max(p, 1 - p),
                'calibrated': False, 'temperature': 1.0}}}
        if self.mutate:
            self.mutate(result)
        return result


def rank_args():
    return {'query': 'fix the timeout', 'candidates': [
        {'id': 'other', 'source': 'fixture', 'text': 'unrelated\r\nsecond line\r\n'},
        {'id': 'useful', 'source': 'fixture', 'text': 'fix the timeout\r\nthird line\r\n'}],
        'max_chars': 4000}


def skills_args():
    return {'task': 'review code', 'enabled_skills': [
        {'id': 'useful', 'description': 'review code'},
        {'id': 'other', 'description': 'read PDFs'}], 'max_chars': 2000}


class NativeContextTests(unittest.TestCase):
    def test_independent_noul_scores_and_exact_sanitized_restore(self):
        predictor = FixturePredictor()
        scorer = NativeContextScorer(predictor)
        with tempfile.TemporaryDirectory() as directory:
            args = rank_args()
            args['store_dir'] = directory
            original = copy.deepcopy(args)
            result = invoke_context('rank_evidence', args, scorer)
            self.assertEqual(args, original)
            self.assertEqual(result['results'][0]['id'], 'useful')
            self.assertEqual(result['results'][0]['score'], 0.8)
            self.assertEqual(result['route']['kind'], 'native_local')
            self.assertFalse(result['route']['calibrated'])
            self.assertTrue(result['route']['shadow'])
            self.assertFalse(result['execution_authorized'])
            self.assertEqual(len(predictor.requests), 2)
            for request in predictor.requests:
                self.assertEqual(set(request['state']), {'query', 'candidate'})
                self.assertEqual(request['questions']['relevance']['type'], 'noul')
            for row in result['results']:
                original_text = next(c['text'] for c in args['candidates'] if c['id'] == row['id'])
                self.assertEqual(restore_lines(directory, row['sha256'], 1, 2), original_text)

    def test_secrets_removed_before_real_callback_boundary(self):
        predictor = FixturePredictor()
        args = rank_args()
        args['query'] = 'Bearer synthetic-query-secret'
        args['candidates'][0]['text'] = 'password=synthetic-secret\nknown-private-value\n'
        args['known_secrets'] = ['known-private-value']
        result = invoke_context('rank_evidence', args, NativeContextScorer(predictor))
        serialized = json.dumps(predictor.requests) + json.dumps(result)
        for value in ('synthetic-query-secret', 'synthetic-secret', 'known-private-value'):
            self.assertNotIn(value, serialized)
        self.assertIn('REDACTED', serialized)

    def test_complete_normalized_identity_bound_uncalibrated_output_required(self):
        changes = [
            lambda r: r.update(runtime_fingerprint='b' * 64),
            lambda r: r['answers'].clear(),
            lambda r: r.update(unexpected='private'),
            lambda r: r['answers']['relevance'].update(calibrated=True),
            lambda r: r['answers']['relevance'].update(temperature=2.0),
            lambda r: r['answers']['relevance']['probabilities'].update(true=float('nan')),
            lambda r: r['answers']['relevance']['probabilities'].update(extra=0.0),
            lambda r: r['answers']['relevance']['probabilities'].update(true=0.0),
            lambda r: r['answers']['relevance'].update(selected_key='missing'),
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                args = rank_args()
                args['store_dir'] = directory
                with self.assertRaises(ModelRouteError):
                    invoke_context('rank_evidence', args, NativeContextScorer(FixturePredictor(change)))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_provenance_budget_reserved_before_persistence(self):
        scorer = NativeContextScorer(FixturePredictor())
        args = rank_args()
        with tempfile.TemporaryDirectory() as directory:
            args['store_dir'] = directory
            full = invoke_context('rank_evidence', args, scorer)
            required = len(json.dumps(full, ensure_ascii=False, separators=(',', ':')))
        with tempfile.TemporaryDirectory() as directory:
            args['store_dir'] = directory
            args['max_chars'] = required - 1
            rejected = invoke_context('rank_evidence', args, scorer)
            self.assertEqual(rejected['status'], 'requires_larger_budget')
            self.assertEqual(rejected['required_chars'], required)
            self.assertEqual(list(Path(directory).iterdir()), [])
            args['max_chars'] = required
            accepted = invoke_context('rank_evidence', args, scorer)
            self.assertEqual(accepted['status'], 'ok')
            self.assertEqual(len(json.dumps(accepted, ensure_ascii=False, separators=(',', ':'))), required)

    def test_full_request_validation_before_any_inference(self):
        predictor = FixturePredictor()
        args = rank_args()
        args['candidates'][1]['text'] = 'x' * 140000
        with self.assertRaises(ModelRouteError):
            invoke_context('rank_evidence', args, NativeContextScorer(predictor))
        self.assertEqual(predictor.requests, [])

    def test_skill_closed_set_and_empty_set(self):
        predictor = FixturePredictor()
        scorer = NativeContextScorer(predictor)
        result = invoke_context('suggest_skills', skills_args(), scorer)
        self.assertEqual([r['id'] for r in result['suggestions']], ['useful', 'other'])
        predictor.requests.clear()
        args = skills_args()
        args['enabled_skills'] = []
        self.assertEqual(invoke_context('suggest_skills', args, scorer)['suggestions'], [])
        self.assertEqual(predictor.requests, [])

    def test_no_call_when_provenance_exceeds_tiny_budget(self):
        predictor = FixturePredictor()
        args = skills_args()
        args['max_chars'] = 35
        result = invoke_context('suggest_skills', args, NativeContextScorer(predictor))
        self.assertEqual(result, {'status': 'requires_larger_budget'})
        self.assertEqual(predictor.requests, [])

    def test_factory_rejects_remote_extra_fields_and_nonoffline_start(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = dict(mode='native', backend='laya', checkpoint=directory,
                        device='cpu', amp='none', weights_dtype='fp32')
            for change in ({'checkpoint': 'remote/model'}, {'backend': 'cloud'}, {'mode': 'ort'},
                           {'weights_dtype': 'bf16'}, {'provider': 'cloud'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    build_native_scorer({**spec, **change})
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                build_native_scorer(spec)

    def test_factory_rejects_network_paths_linked_ancestors_and_cached_online_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint'
            checkpoint.mkdir()
            spec = dict(mode='native', backend='laya', checkpoint=str(checkpoint),
                        device='cpu', amp='none', weights_dtype='fp32')
            with self.assertRaises(ValueError):
                build_native_scorer({**spec, 'checkpoint': '//server/share/checkpoint'})
            with patch.object(Path, 'is_symlink', lambda p: p == checkpoint.parent), \
                    patch.object(Path, 'is_dir', side_effect=AssertionError('must not follow linked ancestor')), \
                    self.assertRaises(ValueError):
                build_native_scorer(spec)
            flags = {k: '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')}
            with patch.dict(os.environ, flags), patch.dict(sys.modules, {
                    'huggingface_hub.constants': SimpleNamespace(HF_HUB_OFFLINE=False)}):
                with self.assertRaisesRegex(ValueError, 'imported without offline'):
                    build_native_scorer(spec)
            with patch.dict(os.environ, flags), patch.dict(sys.modules, {
                    'huggingface_hub.constants': SimpleNamespace(HF_HUB_OFFLINE=True),
                    'transformers.utils.hub': SimpleNamespace(_is_offline_mode=False)}):
                with self.assertRaisesRegex(ValueError, 'imported without offline'):
                    build_native_scorer(spec)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows junction fixture')
    def test_factory_rejects_real_windows_junction_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target, link = root / 'owned-target', root / 'owned-link'
            (target / 'checkpoint').mkdir(parents=True)
            script = root / 'make-junction.ps1'
            script.write_text('param([string]$Link, [string]$Target)\n'
                'New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null\n', encoding='utf-8')
            subprocess.run(['pwsh.exe', '-NoLogo', '-NoProfile', '-File', str(script),
                            str(link), str(target)], check=True, capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=20)
            try:
                with self.assertRaisesRegex(ValueError, 'absolute local directory'):
                    build_native_scorer(dict(mode='native', backend='laya', checkpoint=str(link / 'checkpoint'),
                                            device='cpu', amp='none', weights_dtype='fp32'))
            finally:
                # rmdir unlinks only this test-owned junction, not its target.
                link.rmdir()

    def test_cli_explicit_native_and_failure_never_echoes_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source, config, output = (path / x for x in ('input.json', 'config.json', 'out.json'))
            source.write_text(json.dumps(rank_args()), encoding='utf-8')
            config.write_text('{}', encoding='utf-8')
            command = ['context', 'rank_evidence', '--input', str(source),
                       '--native-runtime', str(config), '--output', str(output)]
            stream = io.StringIO()
            with patch('jevlocal.integration.native_context.build_native_scorer',
                       return_value=NativeContextScorer(FixturePredictor())), redirect_stdout(stream):
                self.assertEqual(main(command), 0)
            self.assertEqual(json.loads(stream.getvalue())['route']['kind'], 'native_local')
            before = output.read_bytes()
            def broken(_):
                raise RuntimeError('private-failure-payload')
            with patch('jevlocal.integration.native_context.build_native_scorer',
                       return_value=NativeContextScorer(FixturePredictor(broken))), redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(command), 2)
            self.assertNotIn('private-failure-payload', stream.getvalue())
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(json.loads(source.read_text(encoding='utf-8')), rank_args())

    def test_mcp_native_call_and_no_request_selected_runtime(self):
        from jevlocal.integration.mcp_server import create_server
        with tempfile.TemporaryDirectory() as directory:
            with patch('jevlocal.integration.native_context.build_native_scorer',
                       return_value=NativeContextScorer(FixturePredictor())):
                server = create_server(directory, native_runtime={})
            tools = asyncio.run(server.list_tools())
            for tool in tools:
                self.assertNotIn('native_runtime', tool.inputSchema['properties'])
                self.assertNotIn('checkpoint', tool.inputSchema['properties'])
                self.assertNotIn('score_callback', tool.inputSchema['properties'])
            result = asyncio.run(server.call_tool('suggest_skills', skills_args()))
            structured = json.loads(result[0].text)
            self.assertEqual(structured['route']['kind'], 'native_local')
            self.assertFalse(structured['execution_authorized'])

    def test_native_selection_on_unsupported_tool_rejects_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps({'command': 'NEVER_EXECUTE', 'exit_code': 0,
                'output': 'synthetic', 'goal': 'test', 'constraints': [], 'max_chars': 1000}), encoding='utf-8')
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                status = main(['context', 'filter_tool_output', '--input', str(source),
                               '--native-runtime', str(Path(directory) / 'not-read.json')])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(output.getvalue())['status'], 'rejected')
            self.assertIn('NEVER_EXECUTE', source.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
