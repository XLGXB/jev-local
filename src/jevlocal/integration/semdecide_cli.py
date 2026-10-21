"""Bounded, isolated, uncalibrated local typed decisions. Not an authority grant."""
from __future__ import annotations

import argparse
import copy
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import time

from decision_deploy.protocol import validate_envelope, validate_reply
from .native_context import validate_native_spec
from . import state_json_advisory


INPUT_CAP = 8 * 1024 * 1024
RESULT_CAP = 24 * 1024 * 1024
OFFLINE = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
           'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD')
SHA_CHARS = frozenset('0123456789abcdef')


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def _constant(_):
    raise ValueError('nonfinite JSON')


def _loads(raw):
    text = raw.decode('utf-8')
    if text.startswith('\ufeff'):
        raise ValueError('UTF-8 BOM forbidden')
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    _dumps(value)  # Reject escaped lone surrogates before retaining a parsed row.
    return value


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _read_bounded(stream, cap):
    data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError('byte budget exceeded')
    return data


def _local_file(raw):
    if type(raw) is not str or raw.startswith(('\\\\', '//')):
        raise ValueError('local absolute path required')
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError('local absolute path required')
    _reject_reparse(path)
    if not path.is_file():
        raise ValueError('local file required')
    return path


def _reject_reparse(path):
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) &
                                          stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('reparse path forbidden')


def _isolated_interpreter():
    """Require the explicitly active venv, not shared/base site-packages."""
    prefix = Path(sys.prefix)
    base = Path(sys.base_prefix)
    if prefix == base or not prefix.is_absolute() or not base.is_absolute():
        raise ValueError('isolated virtual environment required')
    site = prefix / 'Lib' / 'site-packages'
    for path in (prefix, base, site):
        if str(path).startswith(('\\\\', '//')):
            raise ValueError('local virtual environment required')
        _reject_reparse(path)
        if not path.is_dir():
            raise ValueError('virtual environment path missing')
    config_path = _local_file(str(prefix / 'pyvenv.cfg'))
    with config_path.open('rb') as stream:
        text = _read_bounded(stream, 16384).decode('utf-8')
    config = {}
    for line in text.splitlines():
        key, separator, value = line.partition('=')
        if separator:
            key = key.strip().lower()
            if key in config:
                raise ValueError('ambiguous virtual environment configuration')
            config[key] = value.strip()
    if config.get('include-system-site-packages', '').lower() != 'false':
        raise ValueError('shared site-packages forbidden')
    executable = _local_file(str(getattr(sys, '_base_executable', sys.executable)))
    if executable.parent != base:
        raise ValueError('underlying interpreter does not match the active venv')
    return str(executable), str(site)


def _bootstrap(src, executable_prefix, exec_prefix, venv_site, gate):
    # -I -S leaves only the standard library available before Job admission.
    return (
        'import os,sys,json,time\n'
        f'gate={str(gate)!r}\n'
        'deadline=time.monotonic()+20\n'
        'while not os.path.isfile(gate):\n'
        ' if time.monotonic()>=deadline: raise SystemExit(2)\n'
        ' time.sleep(.01)\n'
        'with open(gate,"rb") as stream: raw=stream.read(1025)\n'
        'if len(raw)>1024: raise SystemExit(2)\n'
        'if json.loads(raw)!={"pid":os.getpid(),"admitted":True}: raise SystemExit(2)\n'
        f'sys.prefix={executable_prefix!r}\n'
        f'sys.exec_prefix={exec_prefix!r}\n'
        f'sys.path.insert(0,{venv_site!r})\n'
        f'sys.path.insert(0,{src!r})\n'
        'from jevlocal.integration.semdecide_cli import main\n'
        'raise SystemExit(main(sys.argv[1:]))\n')


def _valid_id(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 128


def _ids(value):
    if type(value) is not dict:
        return None, None
    return tuple(value.get(k) if _valid_id(value.get(k)) else None
                 for k in ('request_id', 'observation_id'))


def _parse_input(data, fmt, max_records):
    if len(data) > INPUT_CAP:
        raise ValueError('input byte budget exceeded')
    if fmt == 'json':
        value = _loads(data)
        if type(value) not in (dict, list):
            raise ValueError('JSON batch must be an envelope or array')
        rows = value if type(value) is list else [value]
        if not 1 <= len(rows) <= max_records:
            raise ValueError('record count exceeded or empty')
        return rows
    if fmt != 'jsonl':
        raise ValueError('invalid format')
    lines = data.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    if not 1 <= len(lines) <= max_records:
        raise ValueError('record count exceeded or empty')
    rows = []
    for line in lines:
        try:
            rows.append(_loads(line))
        except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
            rows.append(_ParseError())
    return rows


class _ParseError:
    pass


def _error(index, value, code):
    rid, oid = _ids(value)
    row = {'record_index': index, 'request_id': rid, 'observation_id': oid,
           'status': 'error', 'error': {'code': code}}
    if not isinstance(value, _ParseError):
        row['input'] = value
    return row


def _validate_runtime_config(config):
    if type(config) is dict and 'route' in config:
        if config['route'] != state_json_advisory.ROUTE:
            raise ValueError('invalid runtime route')
        return state_json_advisory.validate_config(config)
    return validate_native_spec(config)


def _runtime_backend(config):
    return config['spec']['backend'] if 'route' in config else config['backend']


def _native_identity(predictor, expected_backend=None, route_config=None):
    if route_config is not None:
        return state_json_advisory.identity(predictor, route_config)
    runtime = predictor.runtime
    fp = runtime.fingerprint
    if (runtime.mode != 'native' or runtime.backend not in ('laya', 'kev') or
            (expected_backend is not None and runtime.backend != expected_backend) or
            predictor.calibration is not None or type(fp) is not str or
            len(fp) != 64 or any(c not in SHA_CHARS for c in fp)):
        raise ValueError('invalid native route')
    return fp


def _prediction(envelope, prediction, fingerprint, route_config=None):
    if route_config is not None:
        if type(prediction) is not dict or 'runtime_fingerprint' not in prediction:
            raise ValueError('advisory prediction shape')
        if prediction['runtime_fingerprint'] != fingerprint:
            raise _RouteDrift()
        return state_json_advisory.validate_prediction(envelope, prediction, route_config)
    if (type(prediction) is not dict or set(prediction) !=
            {'runtime_fingerprint', 'answers', 'inference_total_ms'}):
        raise ValueError('native prediction shape')
    if prediction['runtime_fingerprint'] != fingerprint:
        raise _RouteDrift()
    answers = prediction['answers']
    if type(answers) is not dict or set(answers) != set(envelope['questions']):
        raise ValueError('answer set changed')
    for answer in answers.values():
        if (type(answer) is not dict or set(answer) !=
                {'selected_key', 'probabilities', 'p_selected', 'calibrated', 'temperature'} or
                answer['calibrated'] is not False or
                type(answer['temperature']) not in (int, float) or
                answer['temperature'] != 1.0):
            raise ValueError('uncalibrated exact answer required')
    reply = {'request_id': envelope['request_id'],
             'observation_id': envelope['observation_id'], **prediction}
    validate_reply(envelope, reply)
    return reply


def _filter_valid(value, question, keys):
    if question is None:
        return True
    options = validate_envelope(value)
    return question in options and all(key in options[question] for key in keys)


def _prevalidate(values, filter_question, keep_keys):
    """Pure protocol pass shared by worker and parent before model admission."""
    rows = [None] * len(values)
    seen = {}
    valid = []
    for i, value in enumerate(values, 1):
        rid, _ = _ids(value)
        if rid is not None:
            seen.setdefault(rid, []).append(i)
    duplicates = {rid for rid, positions in seen.items() if len(positions) > 1}
    for i, value in enumerate(values, 1):
        if isinstance(value, _ParseError):
            rows[i-1] = _error(i, value, 'invalid_json')
            continue
        rid, _ = _ids(value)
        if rid in duplicates:
            rows[i-1] = _error(i, value, 'duplicate_request_id')
            continue
        try:
            validate_envelope(value)
        except (ValueError, TypeError, KeyError):
            rows[i-1] = _error(i, value, 'invalid_envelope')
            continue
        if not _filter_valid(value, filter_question, keep_keys):
            rows[i-1] = _error(i, value, 'invalid_filter')
            continue
        valid.append((i, value))
    return rows, valid


def _process_batch(values, build_predictor, filter_question, keep_keys,
                   expected_backend=None, route_config=None):
    """Prevalidate every row; initialize only if at least one routeable row."""
    rows, valid = _prevalidate(values, filter_question, keep_keys)
    if not valid:
        return rows
    try:
        predictor = build_predictor()
        fingerprint = _native_identity(predictor, expected_backend, route_config)
    except Exception:
        for i, value in valid:
            rows[i-1] = _error(i, value, 'initialization_failed')
        return rows
    drifted = False
    for i, value in valid:
        if drifted:
            rows[i-1] = _error(i, value, 'route_invalid')
            continue
        try:
            if _native_identity(predictor, expected_backend, route_config) != fingerprint:
                raise _RouteDrift()
            request = copy.deepcopy({'state': value['state'], 'questions': value['questions']})
            prediction = predictor.predict(request)
            if _native_identity(predictor, expected_backend, route_config) != fingerprint:
                raise _RouteDrift()
            reply = _prediction(value, prediction, fingerprint, route_config)
            rows[i-1] = {'record_index': i, 'request_id': value['request_id'],
                         'observation_id': value['observation_id'], 'status': 'ok',
                         'input': value, 'reply': reply}
        except _RouteDrift:
            drifted = True
            rows[i-1] = _error(i, value, 'route_invalid')
        except Exception:
            # A changed route is never treated as an ordinary model failure.
            try:
                drifted = _native_identity(predictor, expected_backend, route_config) != fingerprint
            except Exception:
                drifted = True
            rows[i-1] = _error(i, value, 'route_invalid' if drifted else 'invalid_prediction')
    return rows


class _RouteDrift(Exception):
    pass


def _verify_worker_rows(rows, filter_question, keep_keys, expected_values=None,
                        expected_prevalidation=None, route_config=None):
    if (type(rows) is not list or not rows or len(rows) > 256 or
            (expected_values is not None and len(rows) != len(expected_values))):
        raise ValueError('worker rows shape')
    duplicates = {rid for rid in (row.get('request_id') for row in rows if type(row) is dict)
                  if rid is not None and sum(type(r) is dict and r.get('request_id') == rid
                                             for r in rows) > 1}
    success_fingerprint = None
    route_quarantined = False
    for i, row in enumerate(rows, 1):
        if type(row) is not dict or row.get('record_index') != i or type(row['record_index']) is not int:
            raise ValueError('worker row index')
        if expected_values is not None:
            original = expected_values[i-1]
            if isinstance(original, _ParseError):
                if 'input' in row:
                    raise ValueError('worker fabricated parse input')
            elif 'input' not in row or _dumps(row['input']) != _dumps(original):
                raise ValueError('worker input snapshot mismatch')
            expected_error = expected_prevalidation[i-1]
            if expected_error is not None and (row.get('status') != 'error' or
                    row.get('error') != expected_error['error']):
                raise ValueError('worker prevalidation changed')
            if expected_error is None and row.get('status') == 'error' and row.get('error', {}).get('code') in (
                    'invalid_json', 'duplicate_request_id', 'invalid_envelope', 'invalid_filter'):
                raise ValueError('worker fabricated prevalidation error')
        if row.get('status') == 'ok':
            if route_quarantined:
                raise ValueError('worker predicted after route drift')
            if set(row) != {'record_index', 'request_id', 'observation_id', 'status', 'input', 'reply'}:
                raise ValueError('worker success shape')
            envelope = row['input']
            validate_envelope(envelope)
            if not _filter_valid(envelope, filter_question, keep_keys):
                raise ValueError('worker filter mismatch')
            if (row['request_id'], row['observation_id']) != _ids(envelope):
                raise ValueError('worker association')
            if row['request_id'] in duplicates:
                raise ValueError('worker duplicate success')
            reply = row['reply']
            if type(reply) is not dict or set(reply) != {'request_id', 'observation_id',
                    'runtime_fingerprint', 'answers', 'inference_total_ms'}:
                raise ValueError('worker reply shape')
            if (reply['request_id'], reply['observation_id']) != _ids(envelope):
                raise ValueError('worker reply association')
            fp = reply['runtime_fingerprint']
            if type(fp) is not str or len(fp) != 64 or any(c not in SHA_CHARS for c in fp):
                raise ValueError('worker fingerprint')
            if route_config is not None and fp != route_config['expected_fingerprint']:
                raise ValueError('worker fingerprint differs from pinned route')
            if success_fingerprint is not None and fp != success_fingerprint:
                raise ValueError('worker fingerprint changed')
            success_fingerprint = fp
            try:
                _prediction(envelope, {k: reply[k] for k in ('runtime_fingerprint', 'answers',
                                                             'inference_total_ms')}, fp, route_config)
            except _RouteDrift as exc:
                raise ValueError('worker route drift') from exc
        elif row.get('status') == 'error':
            if set(row) not in ({'record_index', 'request_id', 'observation_id', 'status', 'error'},
                                {'record_index', 'request_id', 'observation_id', 'status', 'error', 'input'}):
                raise ValueError('worker error shape')
            error = row['error']
            allowed = {'invalid_json', 'duplicate_request_id', 'invalid_envelope', 'invalid_filter',
                       'initialization_failed', 'invalid_prediction', 'route_invalid'}
            if type(error) is not dict or set(error) != {'code'} or error['code'] not in allowed:
                raise ValueError('worker error code')
            if 'input' in row and (row['request_id'], row['observation_id']) != _ids(row['input']):
                raise ValueError('worker error association')
            if 'input' not in row and error['code'] != 'invalid_json':
                raise ValueError('worker missing input')
            code = error['code']
            if code == 'route_invalid':
                route_quarantined = True
            if code == 'invalid_json':
                if 'input' in row or row['request_id'] is not None or row['observation_id'] is not None:
                    raise ValueError('worker parse error association')
            elif code == 'duplicate_request_id':
                if row['request_id'] is None or row['request_id'] not in duplicates:
                    raise ValueError('worker duplicate error mismatch')
            else:
                if row['request_id'] in duplicates:
                    raise ValueError('worker duplicate priority mismatch')
                valid = True
                try:
                    validate_envelope(row['input'])
                except (ValueError, TypeError, KeyError):
                    valid = False
                if code == 'invalid_envelope' and valid:
                    raise ValueError('worker envelope error mismatch')
                if code not in ('invalid_envelope',) and not valid:
                    raise ValueError('worker unvalidated input')
                if valid:
                    filter_valid = _filter_valid(row['input'], filter_question, keep_keys)
                    if (code == 'invalid_filter') == filter_valid:
                        raise ValueError('worker filter error mismatch')
        else:
            raise ValueError('worker status')
        rid = row['request_id']
        if rid is not None and not _valid_id(rid):
            raise ValueError('worker request ID')
        if row['observation_id'] is not None and not _valid_id(row['observation_id']):
            raise ValueError('worker observation ID')
    return rows


def _deny_socket():
    def denied(*_args, **_kwargs):
        raise OSError('offline SemDecide worker')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def _await_gate(gate_path):
    deadline = time.monotonic() + 20
    while not Path(gate_path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('worker admission not received')
        time.sleep(.01)
    with Path(gate_path).open('rb') as stream:
        gate = _loads(_read_bounded(stream, 1024))
    if gate != {'pid': os.getpid(), 'admitted': True}:
        raise ValueError('worker admission mismatch')


def _worker(manifest_path, gate_path, snapshot_path, model_gate_path, result_path):
    # First gate admits input I/O only after Job assignment. Second gate admits
    # model work only after the parent independently snapshots and validates it.
    _await_gate(gate_path)
    with Path(manifest_path).open('rb') as stream:
        manifest = _loads(_read_bounded(stream, 65536))
    if type(manifest) is not dict or set(manifest) != {'runtime', 'input', 'format',
                                                      'max_records', 'filter_question', 'keep_keys'}:
        return 2
    if any(os.environ.get(flag) != '1' for flag in OFFLINE):
        return 2
    _deny_socket()
    try:
        if manifest['input'] == '-':
            data = _read_bounded(sys.stdin.buffer, INPUT_CAP)
        else:
            with _local_file(manifest['input']).open('rb') as stream:
                data = _read_bounded(stream, INPUT_CAP)
    except Exception:
        _write_result(result_path, {'batch_error': 'invalid_input_batch'})
        return 0
    staged_snapshot = Path(snapshot_path).with_suffix('.tmp')
    with staged_snapshot.open('xb') as stream:
        stream.write(data)
    os.rename(staged_snapshot, snapshot_path)
    _await_gate(model_gate_path)
    try:
        spec = _validate_runtime_config(manifest['runtime'])
        rows = _parse_input(data, manifest['format'], manifest['max_records'])
    except Exception:
        _write_result(result_path, {'batch_error': 'invalid_input_batch'})
        return 0
    route_config = spec if 'route' in spec else None
    if route_config is None:
        from .native_context import build_native_predictor
        build_predictor = lambda: build_native_predictor(spec)
    else:
        build_predictor = lambda: state_json_advisory.build_predictor(spec)
    result = _process_batch(rows, build_predictor, manifest['filter_question'],
                            tuple(manifest['keep_keys']), _runtime_backend(spec), route_config)
    _write_result(result_path, {'backend': _runtime_backend(spec), 'rows': result})
    return 0


def _write_result(result_path, result):
    with Path(result_path).open('xb') as stream:
        payload = _dumps(result)
        if len(payload) > RESULT_CAP:
            raise ValueError('result byte budget exceeded')
        stream.write(payload)


class _LargeInteger(ctypes.Structure):
    _fields_ = [('QuadPart', ctypes.c_int64)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', _LargeInteger), ('PerJobUserTimeLimit', _LargeInteger),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimit), ('IoInfo', _IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [('TotalUserTime', _LargeInteger), ('TotalKernelTime', _LargeInteger),
                ('ThisPeriodTotalUserTime', _LargeInteger),
                ('ThisPeriodTotalKernelTime', _LargeInteger),
                ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]


class _Job:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows Job Object required')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        h = wintypes.HANDLE
        for name, args, result in (
            ('CreateJobObjectW', (ctypes.c_void_p, wintypes.LPCWSTR), h),
            ('SetInformationJobObject', (h, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD), wintypes.BOOL),
            ('AssignProcessToJobObject', (h, h), wintypes.BOOL),
            ('TerminateJobObject', (h, wintypes.UINT), wintypes.BOOL),
            ('QueryInformationJobObject', (h, ctypes.c_int, ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)), wintypes.BOOL),
            ('CloseHandle', (h,), wintypes.BOOL)):
            fn = getattr(self.api, name); fn.argtypes = args; fn.restype = result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, proc):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait_empty(self, deadline):
        while True:
            accounting = _BasicAccounting()
            returned = wintypes.DWORD()
            if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(accounting),
                    ctypes.sizeof(accounting), ctypes.byref(returned)):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.ActiveProcesses == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('owned worker tree did not exit')
            time.sleep(.02)

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def _public_rows(rows, question, keys, route_config=None):
    output = []
    route = state_json_advisory.metadata(route_config) if route_config is not None else None
    for row in rows:
        public = {k: v for k, v in row.items() if k != 'input'}
        if route is not None:
            public['route'] = route
        if row['status'] == 'error':
            output.append(public)
        elif question is None:
            output.append(public)
        elif row['reply']['answers'][question]['selected_key'] in keys:
            public['input'] = row['input']
            output.append(public)
        else:
            sys.stderr.buffer.write(_dumps({'event': 'filtered_out', 'record_index': row['record_index'],
                                            'request_id': row['request_id'],
                                            'observation_id': row['observation_id']}) + b'\n')
    return output


def _emit(rows, fmt):
    if fmt == 'json':
        sys.stdout.buffer.write(_dumps(rows) + b'\n')
    else:
        for row in rows:
            sys.stdout.buffer.write(_dumps(row) + b'\n')
    sys.stdout.buffer.flush()


def _fatal(code, fmt, route_config=None):
    row = {'record_index': 0, 'request_id': None, 'observation_id': None,
           'status': 'error', 'error': {'code': code}}
    if route_config is not None:
        row['route'] = state_json_advisory.metadata(route_config)
    _emit([row], fmt)
    return 2


def _write_gate(path, pid):
    staging = path.with_suffix('.tmp')
    with staging.open('xb') as stream:
        stream.write(_dumps({'pid': pid, 'admitted': True}))
    os.rename(staging, path)


class _InputBatch(Exception):
    pass


def run_cli(runtime_config, input_path='-', fmt='jsonl', max_records=64, timeout=120,
            filter_question=None, keep_keys=(), *, stdin=None, job_factory=_Job,
            popen=subprocess.Popen):
    """One owned Windows worker; no success is emitted until Job cleanup and validation."""
    if os.name != 'nt':
        return _fatal('unsupported_platform', fmt)
    try:
        executable, venv_site = _isolated_interpreter()
    except Exception:
        return _fatal('isolated_environment_required', fmt if fmt in ('json', 'jsonl') else 'jsonl')
    try:
        runtime = _local_file(runtime_config)
        source = '-' if input_path == '-' else str(_local_file(input_path))
        if (fmt not in ('json', 'jsonl') or type(max_records) is not int or
                not 1 <= max_records <= 256 or type(timeout) not in (int, float) or
                not math.isfinite(timeout) or not 1 <= timeout <= 600 or
                (filter_question is None) != (not keep_keys) or
                (filter_question is not None and not _valid_id(filter_question)) or
                type(keep_keys) not in (tuple, list) or len(set(keep_keys)) != len(keep_keys) or
                any(not _valid_id(k) for k in keep_keys)):
            raise ValueError('invalid arguments')
    except Exception:
        return _fatal('invalid_arguments', fmt if fmt in ('json', 'jsonl') else 'jsonl')
    try:
        with runtime.open('rb') as stream:
            spec = _validate_runtime_config(_loads(_read_bounded(stream, 65536)))
    except Exception:
        return _fatal('invalid_runtime_config', fmt)
    started = time.monotonic()
    proc = None
    temp_obj = None
    job = None
    assigned = False
    cleanup_deadline = None
    error = None
    batch_error = None
    rows = None
    expected_values = None
    expected_prevalidation = None
    try:
        temp_obj = tempfile.TemporaryDirectory(prefix='jevlocal-semdecide-')
        temp = temp_obj.name
        root = Path(temp)
        _reject_reparse(root)
        manifest = root / 'manifest.json'
        gate = root / 'admit.json'
        snapshot = root / 'input.snapshot'
        model_gate = root / 'model-admit.json'
        result = root / 'result.json'
        with manifest.open('xb') as stream:
            stream.write(_dumps({'runtime': spec, 'input': source, 'format': fmt,
                'max_records': max_records, 'filter_question': filter_question,
                'keep_keys': list(keep_keys)}))
        job = job_factory()
        env = os.environ.copy()
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        env.update({flag: '1' for flag in OFFLINE})
        src = str(Path(__file__).resolve().parents[2])
        # Windows venv python.exe is a redirector: Popen tracks its short-lived
        # launcher, not the real worker. Launch its underlying interpreter so
        # Job assignment and admission bind the actual process ID.
        bootstrap = _bootstrap(src, sys.prefix, sys.exec_prefix, venv_site, gate)
        proc = popen([executable, '-I', '-S', '-B', '-X', 'utf8', '-c', bootstrap, '--worker',
                      str(manifest), str(gate), str(snapshot), str(model_gate), str(result)],
                     stdin=((stdin if stdin is not None else 0)
                            if source == '-' else subprocess.DEVNULL),
                     stdout=2, stderr=2, env=env,
                     creationflags=subprocess.CREATE_NO_WINDOW,
                     close_fds=True)
        job.assign(proc)
        assigned = True
        _write_gate(gate, proc.pid)
        while not snapshot.exists() and proc.poll() is None:
            if time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            time.sleep(.01)
        if snapshot.exists():
            try:
                with _local_file(str(snapshot)).open('rb') as stream:
                    raw_input = _read_bounded(stream, INPUT_CAP)
                expected_values = _parse_input(raw_input, fmt, max_records)
                expected_prevalidation, _ = _prevalidate(expected_values, filter_question,
                                                         tuple(keep_keys))
            except Exception as exc:
                raise _InputBatch() from exc
            _write_gate(model_gate, proc.pid)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        worker_exit = proc.wait(timeout=remaining)
        if worker_exit != 0:
            raise RuntimeError('worker failed')
        cleanup_deadline = time.monotonic() + 20
        job.terminate()  # No descendant may outlive the invocation.
        job.wait_empty(cleanup_deadline)
        job.close()
        job = None
        with _local_file(str(result)).open('rb') as stream:
            payload = _loads(_read_bounded(stream, RESULT_CAP))
        if type(payload) is dict and set(payload) == {'batch_error'} and payload['batch_error'] in (
                'invalid_runtime_config', 'invalid_input_batch'):
            batch_error = payload['batch_error']
        else:
            if (type(payload) is not dict or set(payload) != {'backend', 'rows'} or
                    payload['backend'] != _runtime_backend(spec) or expected_values is None):
                raise ValueError('worker result route or snapshot mismatch')
            rows = _verify_worker_rows(payload['rows'], filter_question, tuple(keep_keys),
                                       expected_values, expected_prevalidation,
                                       spec if 'route' in spec else None)
            if 'route' in spec:
                state_json_advisory.validate_config(spec)
    except subprocess.TimeoutExpired:
        error = 'timeout'
    except _InputBatch:
        batch_error = 'invalid_input_batch'
    except Exception:
        error = 'worker_failed'
    finally:
        cleanup_failed = False
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + 20
        if job is not None and assigned:
            try:
                job.terminate()
            except Exception:
                cleanup_failed = True
        if proc is not None and proc.poll() is None:
            try:
                if not assigned:
                    proc.kill()  # Only the owned, not-yet-admitted child.
            except Exception:
                cleanup_failed = True
            try:
                proc.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
            except Exception:
                cleanup_failed = True
        if job is not None:
            try:
                if assigned:
                    job.wait_empty(cleanup_deadline)
            except Exception:
                cleanup_failed = True
            try:
                job.close()
            except Exception:
                cleanup_failed = True
        if temp_obj is not None:
            try:
                temp_obj.cleanup()
            except Exception:
                cleanup_failed = True
        if cleanup_failed:
            error = 'cleanup_failed'
    if error is not None or rows is None:
        return _fatal(error or batch_error or 'worker_failed', fmt,
                      spec if 'route' in spec else None)
    public = _public_rows(rows, filter_question, tuple(keep_keys),
                          spec if 'route' in spec else None)
    _emit(public, fmt)
    count = sum(r['status'] == 'error' for r in rows)
    sys.stderr.buffer.write(_dumps({'event': 'summary', 'records': len(rows), 'errors': count,
                                    'emitted': len(public)}) + b'\n')
    return 1 if count else 0


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = _Parser(description='Isolated local typed decisions; never authorization')
    parser.add_argument('--runtime-config')
    parser.add_argument('--input', default='-')
    parser.add_argument('--format', choices=('json', 'jsonl'), default='jsonl')
    parser.add_argument('--max-records', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--filter-question')
    parser.add_argument('--keep-key', action='append', default=[])
    parser.add_argument('--worker', nargs=5, help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
    except ValueError:
        words = list(sys.argv[1:] if argv is None else argv)
        fmt = 'json' if '--format=json' in words or any(
            word == '--format' and i + 1 < len(words) and words[i+1] == 'json'
            for i, word in enumerate(words)) else 'jsonl'
        return _fatal('invalid_arguments', fmt)
    if args.worker:
        try:
            return _worker(*args.worker)
        except Exception:
            print('worker_failed', file=sys.stderr)
            return 2
    return run_cli(args.runtime_config, args.input, args.format, args.max_records,
                   args.timeout, args.filter_question, tuple(args.keep_key))


if __name__ == '__main__':
    raise SystemExit(main())
