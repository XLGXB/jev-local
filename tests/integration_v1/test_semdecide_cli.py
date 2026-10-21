"""Focused, model-free SemDecide protocol and lifecycle tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from jevlocal.integration import semdecide_cli as cli


def envelope(rid='r1', kind='noul'):
    criteria = {'false': 'No.', 'true': 'Yes.'} if kind == 'noul' else (
        {'a': 'A.', 'b': 'B.'} if kind == 'choice' else ['Low.', 'High.'])
    return {'request_id': rid, 'observation_id': 'o1', 'state': {'text': 'public'},
            'questions': {'q': {'type': kind, 'instructions': 'Decide.', 'criteria': criteria}}}


class Fake:
    def __init__(self):
        self.runtime = SimpleNamespace(mode='native', backend='laya', fingerprint='a' * 64)
        self.calibration = None
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        keys = cli.validate_envelope({'request_id': 'r1', 'observation_id': 'o1', **request})['q']
        return {'runtime_fingerprint': self.runtime.fingerprint, 'inference_total_ms': 1,
                'answers': {'q': {'selected_key': keys[0],
                                  'probabilities': {k: float(i == 0) for i, k in enumerate(keys)},
                                  'p_selected': 1.0, 'calibrated': False, 'temperature': 1.0}}}


@pytest.mark.parametrize('kind', ['noul', 'choice', 'score'])
def test_valid_types_once(kind):
    fake = Fake()
    rows = cli._process_batch([envelope(kind=kind)], lambda: fake, None, ())
    assert rows[0]['status'] == 'ok'
    assert len(fake.calls) == 1


def test_mixed_schema_and_duplicate_reject_before_model():
    good = envelope('r1')
    bad = envelope('r2'); bad['questions']['q']['label'] = True
    rows = cli._process_batch([good, bad, envelope('r1')], lambda: pytest.fail('model loaded'), None, ())
    assert [r['status'] for r in rows] == ['error'] * 3
    assert [r['error']['code'] for r in rows] == ['duplicate_request_id', 'invalid_envelope', 'duplicate_request_id']


def test_jsonl_bad_line_continues():
    data = b'\n'.join([json.dumps(envelope('a')).encode(), b'{broken', json.dumps(envelope('b')).encode()]) + b'\n'
    rows = cli._parse_input(data, 'jsonl', 64)
    assert len(rows) == 3
    fake = Fake()
    output = cli._process_batch(rows, lambda: fake, None, ())
    assert [r['status'] for r in output] == ['ok', 'error', 'ok']
    assert len(fake.calls) == 2


def test_jsonl_lone_surrogate_is_bad_line_only():
    data = b'{"request_id":"\\ud800","observation_id":"o","state":{},"questions":{}}\n' + (
        json.dumps(envelope('next')).encode() + b'\n')
    values = cli._parse_input(data, 'jsonl', 64)
    rows = cli._process_batch(values, lambda: Fake(), None, ())
    assert rows[0]['error']['code'] == 'invalid_json'
    assert rows[1]['status'] == 'ok'


@pytest.mark.parametrize('data', [b'\xef\xbb\xbf{}', b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff'])
def test_strict_json_rejects(data):
    with pytest.raises(ValueError):
        cli._parse_input(data, 'json', 64)


def test_filter_retains_input_and_omits_nonmatches(capsys):
    fake = Fake()
    rows = cli._process_batch([envelope('a'), envelope('b')], lambda: fake, 'q', ('false',))
    checked = cli._verify_worker_rows(rows, 'q', ('false',))
    assert checked == rows
    assert rows[0]['input'] == envelope('a')


def test_filter_public_output_and_utf8_event(capfd):
    value = envelope('请求甲')
    rows = cli._process_batch([value], lambda: Fake(), 'q', ('true',))
    assert cli._public_rows(rows, 'q', ('true',)) == []
    event = json.loads(capfd.readouterr().err)
    assert event == {'event': 'filtered_out', 'record_index': 1,
                     'request_id': '请求甲', 'observation_id': 'o1'}
    kept = cli._process_batch([value], lambda: Fake(), 'q', ('false',))
    assert cli._public_rows(kept, 'q', ('false',))[0]['input'] == value


def test_invalid_cli_argument_is_structured(capsys):
    assert cli.main(['--format', 'json', '--max-records', '0']) == 2
    assert json.loads(capsys.readouterr().out)[0]['error']['code'] == 'invalid_arguments'


def test_prediction_rejects_calibration_and_drift():
    fake = Fake()
    real = fake.predict
    def calibrated(request):
        out = real(request); out['answers']['q']['calibrated'] = True; return out
    fake.predict = calibrated
    rows = cli._process_batch([envelope()], lambda: fake, None, ())
    assert rows[0]['error']['code'] == 'invalid_prediction'
    fake = Fake()
    def drift(request):
        out = Fake.predict(fake, request); fake.runtime.fingerprint = 'b' * 64; return out
    fake.predict = drift
    rows = cli._process_batch([envelope('a'), envelope('b')], lambda: fake, None, ())
    assert [r['error']['code'] for r in rows] == ['route_invalid', 'route_invalid']
    assert len(fake.calls) == 1


def test_tampered_reply_rejected():
    fake = Fake()
    rows = cli._process_batch([envelope()], lambda: fake, None, ())
    rows[0]['reply']['observation_id'] = 'other'
    with pytest.raises(ValueError):
        cli._verify_worker_rows(rows, None, ())


def test_parent_snapshot_binding_and_quarantine():
    source = [envelope('a'), envelope('b')]
    fake = Fake()
    rows = cli._process_batch(copy.deepcopy(source), lambda: fake, None, ())
    expected, _ = cli._prevalidate(source, None, ())
    assert cli._verify_worker_rows(rows, None, (), source, expected) == rows
    rows[0]['input']['state'] = {'text': 'changed'}
    with pytest.raises(ValueError):
        cli._verify_worker_rows(rows, None, (), source, expected)
    rows[0]['input']['state'] = {'text': 'public'}
    rows[0] = cli._error(1, source[0], 'route_invalid')
    with pytest.raises(ValueError):
        cli._verify_worker_rows(rows, None, (), source, expected)


def test_batch_caps_and_empty():
    for data in (b'', b'[]', b'1'):
        with pytest.raises(ValueError):
            cli._parse_input(data, 'json', 64)
    with pytest.raises(ValueError):
        cli._parse_input(b'', 'jsonl', 64)
    with pytest.raises(ValueError):
        cli._parse_input(b'{}\n{}', 'jsonl', 1)
    with pytest.raises(ValueError):
        cli._parse_input(b' ' * (cli.INPUT_CAP + 1), 'jsonl', 64)


def test_ordinary_failure_continues_without_retry():
    fake = Fake()
    original = fake.predict
    def one_failure(request):
        if not fake.calls:
            fake.calls.append(request)
            raise RuntimeError('secret source data never echoed')
        return original(request)
    fake.predict = one_failure
    rows = cli._process_batch([envelope('a'), envelope('b')], lambda: fake, None, ())
    assert [r['status'] for r in rows] == ['error', 'ok']
    assert len(fake.calls) == 2
    assert rows[0]['error'] == {'code': 'invalid_prediction'}


@pytest.mark.skipif(os.name != 'nt', reason='Windows interpreter isolation test')
def test_worker_interpreter_only_venv_distributions():
    """The same isolated interpreter bootstrap resolves active-venv metadata only."""
    import importlib.metadata as metadata
    site_path = str(Path(sys.prefix) / 'Lib' / 'site-packages')
    names = ('torch', 'transformers', 'safetensors')
    expected = {name: metadata.version(name) for name in names}
    code = (f'import sys,json,importlib.metadata as m;sys.prefix={sys.prefix!r};'
            f'sys.exec_prefix={sys.exec_prefix!r};sys.path.insert(0,{site_path!r});'
            f'names={names!r};print(json.dumps({{"prefix":sys.prefix,"path":sys.path,'
            '"dist":{n:[m.version(n),str(m.distribution(n).locate_file(""))] for n in names}}))')
    raw = subprocess.check_output([sys._base_executable, '-I', '-S', '-B', '-X', 'utf8',
                                   '-c', code], text=True, timeout=10,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
    worker = json.loads(raw)
    assert worker['prefix'] == sys.prefix
    assert {name: entry[0] for name, entry in worker['dist'].items()} == expected
    assert all(Path(entry[1]).resolve().is_relative_to(Path(site_path).resolve())
               for entry in worker['dist'].values())
    assert all('site-packages' not in path.lower() or
               Path(path).resolve().is_relative_to(Path(site_path).resolve())
               for path in worker['path'])


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job admission required')
def test_unicode_id_real_subprocess(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({**envelope('请求甲'), 'questions': {}}), encoding='utf-8')
    assert cli.run_cli(str(runtime), str(source), 'jsonl', 64, 5, None, ()) == 1
    output = json.loads(capfd.readouterr().out)
    assert output['request_id'] == '请求甲'


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job admission required')
def test_job_admission_failure_reaps_unadmitted_worker(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    started = []
    def popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        started.append(process)
        return process
    class Refuse(cli._Job):
        def assign(self, process):
            raise OSError('injected assignment failure')
    assert cli.run_cli(str(runtime), job_factory=Refuse, popen=popen) == 2
    assert started and started[0].poll() is not None
    assert json.loads(capfd.readouterr().out)['error']['code'] == 'worker_failed'


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job admission required')
def test_timeout_terminates_owned_tree(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    observed = []
    class ObservedJob(cli._Job):
        def wait_empty(self, deadline):
            super().wait_empty(deadline)
            observed.append('empty')
    def popen(argv, **kwargs):
        # The fixture, like production, cannot spawn descendants before admission.
        tree = ('import pathlib,subprocess,sys,time\n'
                'gate=pathlib.Path(' + repr(argv[-4]) + ')\n'
                'while not gate.exists(): time.sleep(.01)\n'
                'subprocess.Popen([sys.executable,"-I","-S","-c","import time;time.sleep(60)"],'
                'creationflags=subprocess.CREATE_NO_WINDOW)\n'
                'time.sleep(60)')
        return subprocess.Popen([sys._base_executable, '-I', '-S', '-B', '-c', tree], **kwargs)
    assert cli.run_cli(str(runtime), timeout=1, job_factory=ObservedJob, popen=popen) == 2
    assert observed == ['empty']
    assert json.loads(capfd.readouterr().out)['error']['code'] == 'timeout'


@pytest.mark.skipif(os.name != 'nt', reason='Windows atomic snapshot test')
def test_unpublished_snapshot_is_not_read(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    def popen(argv, **kwargs):
        snapshot = Path(argv[-3])
        code = ('import pathlib,time;pathlib.Path(' + repr(str(snapshot.with_suffix('.tmp'))) +
                ').write_bytes(b"{partial");time.sleep(60)')
        return subprocess.Popen([sys._base_executable, '-I', '-S', '-B', '-c', code], **kwargs)
    assert cli.run_cli(str(runtime), timeout=1, popen=popen) == 2
    assert json.loads(capfd.readouterr().out)['error']['code'] == 'timeout'


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job admission required')
def test_real_subprocess_model_free_stdin_timeout(tmp_path, capfd):
    """A pipe left open must be interrupted by the parent deadline."""
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, 'rb', closefd=True) as stream:
            result = cli.run_cli(str(runtime), '-', 'jsonl', 64, 1, None, (), stdin=stream)
        assert result == 2
        captured = capfd.readouterr()
        assert json.loads(captured.out)['error']['code'] == 'timeout'
    finally:
        os.close(write_fd)


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job admission required')
def test_real_subprocess_all_invalid_no_model(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'mode': 'native', 'backend': 'laya',
        'checkpoint': str(checkpoint), 'device': 'cpu', 'amp': 'none',
        'weights_dtype': 'fp32'}), encoding='utf-8')
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'request_id': 'bad', 'observation_id': 'o',
                                  'state': {}, 'questions': {}}) + '\n', encoding='utf-8')
    assert cli.run_cli(str(runtime), str(source), 'jsonl', 64, 5, None, ()) == 1
    output = json.loads(capfd.readouterr().out)
    assert output['error']['code'] == 'invalid_envelope'


@pytest.mark.skipif(os.name != 'nt', reason='Windows isolated environment boundary')
def test_base_environment_rejected_before_popen(monkeypatch, capfd):
    monkeypatch.setattr(cli.sys, 'prefix', cli.sys.base_prefix)
    assert cli.run_cli(None, popen=lambda *a, **k: pytest.fail('process spawned')) == 2
    assert json.loads(capfd.readouterr().out)['error']['code'] == 'isolated_environment_required'


@pytest.mark.skipif(os.name != 'nt', reason='Windows isolated environment boundary')
def test_shared_site_environment_rejected(tmp_path, monkeypatch):
    prefix = tmp_path / 'shared-env'
    (prefix / 'Lib' / 'site-packages').mkdir(parents=True)
    (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding='utf-8')
    monkeypatch.setattr(cli.sys, 'prefix', str(prefix))
    with pytest.raises(ValueError, match='shared site-packages forbidden'):
        cli._isolated_interpreter()


@pytest.mark.skipif(os.name != 'nt', reason='Windows configuration snapshot boundary')
def test_runtime_config_is_snapshotted_before_blocked_input(tmp_path, capfd):
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    runtime = tmp_path / 'runtime.json'
    spec = {'mode': 'native', 'backend': 'laya', 'checkpoint': str(checkpoint),
            'device': 'cpu', 'amp': 'none', 'weights_dtype': 'fp32'}
    runtime.write_text(json.dumps(spec), encoding='utf-8')
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({**envelope('invalid'), 'questions': {}}), encoding='utf-8')
    observed = []
    def popen(argv, **kwargs):
        manifest = json.loads(Path(argv[-5]).read_text(encoding='utf-8'))
        assert manifest['runtime'] == spec
        observed.append(True)
        # This deliberately invalid replacement must never be reread by the worker.
        runtime.write_text('{"changed_after_validation":true}', encoding='utf-8')
        return subprocess.Popen(argv, **kwargs)
    assert cli.run_cli(str(runtime), str(source), timeout=5, popen=popen) == 1
    assert observed == [True]
    assert json.loads(capfd.readouterr().out)['error']['code'] == 'invalid_envelope'


@pytest.mark.skipif(os.name != 'nt', reason='Windows pre-import Job admission boundary')
def test_bootstrap_imports_project_only_after_job_gate(tmp_path):
    src = tmp_path / 'src'
    integration = src / 'jevlocal' / 'integration'
    integration.mkdir(parents=True)
    (src / 'jevlocal' / '__init__.py').write_text('', encoding='utf-8')
    (integration / '__init__.py').write_text('', encoding='utf-8')
    marker = tmp_path / 'imported.txt'
    (integration / 'semdecide_cli.py').write_text(
        'from pathlib import Path\nPath(' + repr(str(marker)) +
        ').write_text("imported",encoding="utf-8")\ndef main(args): return 0\n', encoding='utf-8')
    gate = tmp_path / 'admit.json'
    executable, site = cli._isolated_interpreter()
    code = cli._bootstrap(str(src), sys.prefix, sys.exec_prefix, site, gate)
    job = cli._Job()
    process = None
    try:
        process = subprocess.Popen([executable, '-I', '-S', '-B', '-X', 'utf8', '-c', code],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(process)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=.3)
        assert not marker.exists()
        cli._write_gate(gate, process.pid)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0 and stdout == b'' and stderr == b''
        assert marker.read_text(encoding='utf-8') == 'imported'
    finally:
        deadline = cli.time.monotonic() + 3
        job.terminate()
        if process is not None:
            process.wait(timeout=max(0, deadline-cli.time.monotonic()))
        job.wait_empty(deadline)
        job.close()
