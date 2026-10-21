"""Model-free SemDecide advisory route and parent/worker boundary tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from jevlocal.integration import semdecide_cli as cli


ROUTE = 'state_json_advisory_v1'
FP = 'a' * 64


def envelope(rid='r1', kind='score'):
    criteria = (['Low.', 'High.'] if kind == 'score' else
                {'a': 'A.', 'b': 'B.'} if kind == 'choice' else
                {'false': 'No.', 'true': 'Yes.'})
    return {'request_id': rid, 'observation_id': 'o1', 'state': {'text': 'public'},
            'questions': {'q': {'type': kind, 'instructions': 'Decide.',
                                'criteria': criteria}}}


def config():
    return {'route': ROUTE, 'expected_fingerprint': FP,
            'calibration': {'sha256': 'b' * 64}}


class Fake:
    def __init__(self):
        self.runtime = SimpleNamespace(fingerprint=FP)
        self.calls = []

    def predict(self, request):
        self.calls.append(copy.deepcopy(request))
        answers = {}
        options = cli.validate_envelope({'request_id': 'r1', 'observation_id': 'o1',
                                         **request})
        for qid, keys in options.items():
            kind = request['questions'][qid]['type']
            answers[qid] = {'selected_key': keys[0],
                            'probabilities': {key: float(key == keys[0]) for key in keys},
                            'p_selected': 1.0, 'calibrated': kind == 'score',
                            'temperature': .41738125681877136 if kind == 'score' else 1.0}
        return {'runtime_fingerprint': self.runtime.fingerprint, 'answers': answers,
                'inference_total_ms': 1.0}


def mock_adapter(monkeypatch):
    def identity(predictor, route):
        if predictor.runtime.fingerprint != route['expected_fingerprint']:
            raise ValueError('drift')
        return route['expected_fingerprint']

    def validate_prediction(value, prediction, route):
        if prediction['runtime_fingerprint'] != route['expected_fingerprint']:
            raise ValueError('fingerprint')
        for qid, answer in prediction['answers'].items():
            score = value['questions'][qid]['type'] == 'score'
            if answer['calibrated'] is not score or answer['temperature'] != (
                    .41738125681877136 if score else 1.0):
                raise ValueError('per-kind calibration')
        reply = {'request_id': value['request_id'],
                 'observation_id': value['observation_id'], **prediction}
        cli.validate_reply(value, reply)
        return reply

    monkeypatch.setattr(cli.state_json_advisory, 'identity', identity)
    monkeypatch.setattr(cli.state_json_advisory, 'validate_prediction', validate_prediction)
    monkeypatch.setattr(cli.state_json_advisory, 'metadata', lambda route: {
        'schema_version': 1, 'kind': ROUTE,
        'runtime_fingerprint': route['expected_fingerprint'],
        'calibration_sha256': route['calibration']['sha256'],
        'advisory_only': True, 'execution_authorized': False,
        'qualification_status': 'not_qualified'})


@pytest.mark.parametrize('fmt', ['json', 'jsonl'])
def test_mixed_types_once_and_bound_reply(monkeypatch, fmt):
    mock_adapter(monkeypatch)
    values = [envelope('score', 'score'), envelope('choice', 'choice'),
              envelope('noul', 'noul')]
    data = (json.dumps(values).encode() if fmt == 'json' else
            b'\n'.join(json.dumps(value).encode() for value in values) + b'\n')
    parsed = cli._parse_input(data, fmt, 64)
    fake = Fake()
    rows = cli._process_batch(parsed, lambda: fake, None, (), route_config=config())
    assert [row['status'] for row in rows] == ['ok'] * 3
    assert len(fake.calls) == 3
    expected, _ = cli._prevalidate(parsed, None, ())
    assert cli._verify_worker_rows(rows, None, (), parsed, expected, config()) == rows
    public = cli._public_rows(rows, None, (), config())
    assert all(set(row['reply']) == {'request_id', 'observation_id',
                                    'runtime_fingerprint', 'answers',
                                    'inference_total_ms'} for row in public)
    assert all(row['route']['execution_authorized'] is False for row in public)
    assert all('input' not in row for row in public)


def test_parent_rejects_pinned_fingerprint_and_per_kind_tampering(monkeypatch):
    mock_adapter(monkeypatch)
    value = envelope()
    rows = cli._process_batch([value], lambda: Fake(), None, (), route_config=config())
    original = copy.deepcopy(rows)
    rows[0]['reply']['runtime_fingerprint'] = 'c' * 64
    with pytest.raises(ValueError, match='pinned route'):
        cli._verify_worker_rows(rows, None, (), [value], [None], config())
    rows = original
    rows[0]['reply']['answers']['q']['temperature'] = 1.0
    with pytest.raises(ValueError, match='calibration'):
        cli._verify_worker_rows(rows, None, (), [value], [None], config())


def test_worker_drift_quarantine_no_retry_or_exception_text(monkeypatch):
    mock_adapter(monkeypatch)
    fake = Fake()
    original = fake.predict

    def drift(request):
        result = original(request)
        fake.runtime.fingerprint = 'c' * 64
        return result

    fake.predict = drift
    rows = cli._process_batch([envelope('a'), envelope('b')], lambda: fake,
                              None, (), route_config=config())
    assert [row['error'] for row in rows] == [{'code': 'route_invalid'}] * 2
    assert len(fake.calls) == 1
    assert 'c' * 64 not in json.dumps(rows)


def test_malformed_prediction_is_row_error_without_exception_text(monkeypatch):
    mock_adapter(monkeypatch)
    fake = Fake()

    def malformed(_request):
        fake.calls.append('one')
        return {'unexpected': 'private model error detail'}

    fake.predict = malformed
    rows = cli._process_batch([envelope()], lambda: fake, None, (), route_config=config())
    assert rows[0]['error'] == {'code': 'invalid_prediction'}
    assert len(fake.calls) == 1
    assert 'private model error detail' not in json.dumps(rows)


def test_advisory_filter_is_only_stdout_projection(monkeypatch, capfd):
    mock_adapter(monkeypatch)
    value = envelope('source', 'noul')
    original = copy.deepcopy(value)
    rows = cli._process_batch([value], lambda: Fake(), 'q', ('true',),
                              route_config=config())
    assert rows[0]['input'] == original
    assert cli._public_rows(rows, 'q', ('true',), config()) == []
    assert json.loads(capfd.readouterr().err)['event'] == 'filtered_out'
    kept = cli._public_rows(rows, 'q', ('false',), config())
    assert kept[0]['input'] == original
    assert kept[0]['route']['qualification_status'] == 'not_qualified'


def test_explicit_selection_and_legacy_default(monkeypatch):
    spec = {'mode': 'native', 'backend': 'kev', 'checkpoint': 'P',
            'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
    monkeypatch.setattr(cli, 'validate_native_spec', lambda value: copy.deepcopy(value))
    monkeypatch.setattr(cli.state_json_advisory, 'validate_config', lambda value: copy.deepcopy(value))
    assert cli._validate_runtime_config(spec) == spec
    assert cli._runtime_backend(spec) == 'kev'
    assert cli._validate_runtime_config(config()) == config()
    assert cli._runtime_backend({**config(), 'spec': {'backend': 'kev'}}) == 'kev'
    for bad in ('wrong', None, 1, ''):
        with pytest.raises(ValueError):
            cli._validate_runtime_config({**config(), 'route': bad})


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job required')
def test_real_adapter_metadata_no_model_for_invalid_rows(capfd, tmp_path):
    """The pinned real config is validated in both processes; invalid input loads no model."""
    config_path = Path(__file__).resolve().parents[2] / 'runtime' / (
        'kev-state-json-semdecide-01/consumer-config.json')
    from jevlocal.integration.state_json_advisory import validate_config, metadata
    route = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    assert metadata(route) == {
        'schema_version': 1, 'kind': ROUTE,
        'runtime_fingerprint': route['expected_fingerprint'],
        'calibration_sha256': route['calibration']['sha256'],
        'advisory_only': True, 'execution_authorized': False,
        'qualification_status': 'not_qualified'}
    source = tmp_path / 'invalid.jsonl'
    source.write_text(json.dumps({**envelope('invalid'), 'questions': {}}), encoding='utf-8')
    observed = []

    def popen(argv, **kwargs):
        manifest = json.loads(Path(argv[-5]).read_text(encoding='utf-8'))
        assert manifest['runtime'] == route
        observed.append(True)
        return subprocess.Popen(argv, **kwargs)

    assert cli.run_cli(str(config_path), str(source), timeout=10, popen=popen) == 1
    assert observed == [True]
    row = json.loads(capfd.readouterr().out)
    assert row['error'] == {'code': 'invalid_envelope'}
    assert row['route'] == metadata(route)


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job required')
@pytest.mark.parametrize('failure', ['worker_failed', 'timeout'])
def test_valid_advisory_fatal_has_metadata_without_model(capfd, failure):
    config_path = Path(__file__).resolve().parents[2] / 'runtime' / (
        'kev-state-json-semdecide-01/consumer-config.json')
    from jevlocal.integration.state_json_advisory import validate_config, metadata
    route = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    if failure == 'worker_failed':
        def popen(*_args, **_kwargs):
            raise OSError('private injected worker failure')
    else:
        def popen(argv, **kwargs):
            # No snapshot is published; the admitted owned process merely waits.
            return subprocess.Popen([sys._base_executable, '-I', '-S', '-B', '-c',
                                     'import time;time.sleep(60)'], **kwargs)
    result = cli.run_cli(str(config_path), timeout=1 if failure == 'timeout' else 5,
                         popen=popen)
    assert result == 2
    row = json.loads(capfd.readouterr().out)
    assert row['error'] == {'code': failure}
    assert row['route'] == metadata(route)
    assert 'private injected worker failure' not in json.dumps(row)


def test_legacy_fatal_has_no_route(monkeypatch, capfd):
    monkeypatch.setattr(cli.state_json_advisory, 'metadata', lambda *_: pytest.fail('called'))
    assert cli._fatal('worker_failed', 'jsonl') == 2
    assert 'route' not in json.loads(capfd.readouterr().out)


@pytest.mark.skipif(os.name != 'nt', reason='Windows interpreter isolation required')
@pytest.mark.parametrize('mutation', ['route', 'fingerprint', 'sidecar_hash'])
def test_bad_route_or_pin_fails_before_worker(capfd, tmp_path, mutation):
    source = Path(__file__).resolve().parents[2] / 'runtime' / (
        'kev-state-json-semdecide-01/consumer-config.json')
    route = json.loads(source.read_text(encoding='utf-8'))
    if mutation == 'route':
        route['route'] = 'unknown'
    elif mutation == 'fingerprint':
        route['expected_fingerprint'] = '0' * 64
    else:
        route['calibration']['sha256'] = '0' * 64
    path = tmp_path / 'bad-route.json'
    path.write_text(json.dumps(route), encoding='utf-8')

    def no_worker(*_args, **_kwargs):
        pytest.fail('worker spawned before config admission')

    assert cli.run_cli(str(path), popen=no_worker) == 2
    row = json.loads(capfd.readouterr().out)
    assert row['error'] == {'code': 'invalid_runtime_config'}
    assert 'route' not in row


def test_real_adapter_identity_prediction_and_metadata_with_fake_runtime():
    """Real adapter policy, but no native load or forward in this synthetic test."""
    from jevlocal.integration import state_json_advisory as adapter
    config_path = Path(__file__).resolve().parents[2] / 'runtime' / (
        'kev-state-json-semdecide-01/consumer-config.json')
    route = adapter.validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    runtime = SimpleNamespace(spec=copy.deepcopy(route['spec']),
                              identity=copy.deepcopy(route['expected_identity']),
                              fingerprint=route['expected_fingerprint'])
    predictor = adapter.StateJsonAdvisoryPredictor(runtime, {}, route)
    fake = Fake()
    fake.runtime.fingerprint = route['expected_fingerprint']
    predictor.predict = fake.predict
    values = [envelope('score', 'score'), envelope('choice', 'choice'),
              envelope('noul', 'noul')]
    rows = cli._process_batch(values, lambda: predictor, None, (), route_config=route)
    assert [row['status'] for row in rows] == ['ok'] * 3
    assert len(fake.calls) == 3
    assert cli._verify_worker_rows(rows, None, (), values, [None] * 3, route) == rows
    public = cli._public_rows(rows, None, (), route)
    assert all(row['route'] == adapter.metadata(route) for row in public)
