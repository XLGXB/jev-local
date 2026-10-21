"""Explicit, bounded MCP passive sessions; saved records are not authority.

Only the background finalizer owns a native job. A caller can signal cancellation,
but cannot race its process reaping or temporary-directory cleanup.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import threading
import uuid

from . import context_tools
from .context_shadow import (_strict_pairs, _reject_constant, start_shadow,
                             unavailable_shadow)
from .scoped_invocation import (prepare_invocation, replay_invocation,
                                validate_binding, validate_invocation)

_LIMIT = 4 * 1024 * 1024


def _safe_path(path: Path) -> None:
    """Reject any existing symlink/junction component before path access."""
    if not path.is_absolute() or str(path).startswith(('\\\\', '//')):
        raise ValueError('owned path must be absolute and local')
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if (component.is_symlink() or
                getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('owned path must not traverse a reparse point')


def _json_bytes(value: object) -> bytes:
    try:
        # The v2 child receipt binds the lexical primary's ordered JSON bytes;
        # sorting keys here would change that digest on readback.
        data = json.dumps(value, ensure_ascii=False, sort_keys=False,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        raise ValueError('invalid bounded record') from None
    if len(data) > _LIMIT:
        raise ValueError('record exceeds size limit')
    return data


def _write_once(path: Path, value: object) -> None:
    _safe_path(path)
    data = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    staging = path.with_name('.' + uuid.uuid4().hex + '.tmp')
    _safe_path(staging)
    try:
        descriptor = os.open(staging, flags, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A complete, fsynced inode becomes visible all at once. Hard linking
        # is exclusive on Windows and POSIX; unlike replace, it cannot overwrite.
        os.link(staging, path)
    except FileExistsError:
        raise ValueError('record already exists') from None
    finally:
        staging.unlink(missing_ok=True)


def _read(path: Path) -> object:
    _safe_path(path)
    with path.open('rb') as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError('record exceeds size limit')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_pairs,
                          parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid saved record') from None


def _invocation_id(value: str) -> str:
    # Reuse the v2 UUID grammar rather than accepting a path-like handle.
    validate_binding({'invocation_id': value, 'manifest_sha256': '0' * 64,
                      'rank_input_sha256': '0' * 64})
    return value


class PassiveSession:
    def __init__(self, runtime: dict, records_dir: str | Path):
        from .native_context import validate_native_spec
        self._runtime = validate_native_spec(copy.deepcopy(runtime))
        self._root = Path(records_dir)
        _safe_path(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        _safe_path(self._root)
        if not self._root.is_dir():
            raise ValueError('owned records directory required')
        self._lock = threading.Lock()
        self._active_id: str | None = None
        self._active_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._quarantined = False
        self._closed = False

    def _directory(self, invocation_id: str) -> Path:
        directory = self._root / _invocation_id(invocation_id)
        _safe_path(directory)
        return directory

    def start(self, query: str, candidates: list[dict], target: dict,
              bindings: dict, *, max_chars: int = 65536,
              rank_max_chars: int = 12000) -> dict:
        arguments = dict(query=query, candidates=candidates, target=target,
                         bindings=bindings, max_chars=max_chars,
                         rank_max_chars=rank_max_chars)
        record = prepare_invocation(arguments)
        if record.get('status') == 'requires_larger_budget':
            return {'status': 'requires_larger_budget', 'invocation_id': None,
                    'primary': record, 'observer': 'not_started'}
        invocation_id = record['binding']['invocation_id']
        directory = self._directory(invocation_id)
        with self._lock:
            if self._closed:
                raise ValueError('passive session is closed')
            # A failed publication remains visible as an incomplete record.
            directory.mkdir(exist_ok=False)
        _safe_path(directory)
        # The full sanitized manifest is exclusively durable before any observer
        # is even scheduled. Its input is derivable from snapshot + numeric budget.
        _write_once(directory / 'invocation.json',
                    {'record': record, 'scope_max_chars': max_chars})
        derived = record['rank_input']
        if derived is None:
            primary = {'status': 'no_matches', 'execution_authorized': False,
                       'training_allowed': False}
        else:
            primary = context_tools._rank_prepared_evidence(
                derived['query'], derived['candidates'],
                max_chars=derived['max_chars'])
        _write_once(directory / 'primary.json', primary)
        if derived is None:
            return {'status': 'complete', 'invocation_id': invocation_id,
                    'primary': primary, 'observer': 'not_started'}
        with self._lock:
            if self._closed:
                capacity = True
            else:
                capacity = self._active_id is not None or self._quarantined
            if not capacity:
                event = threading.Event()
                worker = threading.Thread(
                    target=self._finalize, args=(invocation_id, record, primary, event),
                    name='jevlocal-passive-finalizer', daemon=True)
                self._active_id = invocation_id
                self._active_thread = worker
                self._cancel_event = event
                worker.start()
        if capacity:
            # A capacity receipt is owned by this fresh invocation and does not
            # launch another native process.
            job = unavailable_shadow(derived['query'], derived['candidates'],
                                     self._runtime, reason='capacity',
                                     binding=record['binding'])
            sidecar = job.finish(primary)
            _write_once(directory / 'sidecar.json', sidecar)
            return {'status': 'complete', 'invocation_id': invocation_id,
                    'primary': primary, 'observer': 'capacity'}
        return {'status': 'pending', 'invocation_id': invocation_id,
                'primary': primary, 'observer': 'started'}

    def _finalize(self, invocation_id: str, record: dict, primary: dict,
                  cancel_event: threading.Event) -> None:
        job = None
        quarantine = False
        try:
            derived = record['rank_input']
            if cancel_event.is_set():
                job = unavailable_shadow(derived['query'], derived['candidates'],
                                         self._runtime, reason='cancelled',
                                         binding=record['binding'])
            else:
                job = start_shadow(derived['query'], derived['candidates'], self._runtime,
                                   timeout_s=60, binding=record['binding'],
                                   manifest=record['manifest'], rank_input=derived)
            sidecar = job.finish(primary, cancel_event=cancel_event)
            quarantine = not sidecar['lifecycle']['closed'] or (
                sidecar['lifecycle']['pid'] is not None and
                not sidecar['lifecycle']['reaped'])
            with self._lock:
                # Publication and cancellation share this lock. A cancellation
                # that won before publication cannot leave an observed receipt;
                # after publication active state is cleared before unlock.
                if (cancel_event.is_set() and not quarantine and
                        sidecar['availability']['reason'] != 'cleanup_failed'):
                    sidecar = copy.deepcopy(sidecar)
                    sidecar['availability'] = {
                        'status': 'unavailable', 'observed_count': 0,
                        'candidate_count': len(derived['candidates']),
                        'reason': 'cancelled'}
                    sidecar['native_receipts'] = []
                    sidecar['shadow_ranking'] = []
                    sidecar['shadow_top'] = None
                _write_once(self._directory(invocation_id) / 'sidecar.json', sidecar)
                self._quarantined |= quarantine
                self._active_id = None
                self._active_thread = None
                self._cancel_event = None
        except Exception:
            # A failed finish must not strand a child. The same finalizer owns
            # close; no caller may race it. Do not persist private exception text.
            if job is not None:
                try:
                    cleanup = job.close()
                    quarantine = not cleanup['closed'] or (
                        not cleanup['reaped'] and getattr(job, '_pid', None) is not None)
                except Exception:
                    quarantine = True
        finally:
            with self._lock:
                if self._active_id == invocation_id:
                    self._quarantined |= quarantine
                    self._active_id = None
                    self._active_thread = None
                    self._cancel_event = None

    def _load(self, invocation_id: str) -> tuple[dict, dict | None]:
        """Read one immutable snapshot once and validate its terminal binding."""
        directory = self._directory(invocation_id)
        try:
            saved = _read(directory / 'invocation.json')
        except FileNotFoundError:
            if not directory.is_dir():
                raise ValueError('unknown invocation ID') from None
            return ({'status': 'interrupted', 'invocation_id': invocation_id,
                     'primary': None, 'sidecar': None}, None)
        if type(saved) is not dict or set(saved) != {'record', 'scope_max_chars'} or \
                type(saved['scope_max_chars']) is not int:
            raise ValueError('invalid saved invocation')
        record = saved['record']
        if type(record) is not dict or type(record.get('binding')) is not dict or \
                record['binding'].get('invocation_id') != invocation_id:
            raise ValueError('invalid saved invocation')
        try:
            snapshot = record['manifest']['snapshot']
            arguments = {'query': snapshot['query'], 'candidates': snapshot['candidates'],
                         'target': snapshot['target'], 'bindings': snapshot['bindings'],
                         'max_chars': saved['scope_max_chars'],
                         'rank_max_chars': snapshot['rank_max_chars']}
        except (KeyError, TypeError):
            raise ValueError('invalid saved invocation') from None
        validate_invocation(arguments, record)
        try:
            primary = _read(directory / 'primary.json')
        except FileNotFoundError:
            return ({'status': 'interrupted', 'invocation_id': invocation_id,
                     'primary': None, 'sidecar': None}, None)
        with self._lock:
            try:
                sidecar = _read(directory / 'sidecar.json')
            except FileNotFoundError:
                sidecar = None
            active = self._active_id == invocation_id
        if record['rank_input'] is None or sidecar is not None:
            replay = replay_invocation(arguments, record, primary, sidecar)
            status = 'complete'
        else:
            derived = record['rank_input']
            expected = context_tools._rank_prepared_evidence(
                derived['query'], derived['candidates'],
                max_chars=derived['max_chars'])
            if json.dumps(primary, sort_keys=True, allow_nan=False) != json.dumps(
                    expected, sort_keys=True, allow_nan=False):
                raise ValueError('saved lexical primary mismatch')
            replay = None
            status = 'pending' if active else 'interrupted'
        return ({'status': status, 'invocation_id': invocation_id,
                 'primary': primary, 'sidecar': sidecar}, replay)

    def collect(self, invocation_id: str) -> dict:
        return self._load(invocation_id)[0]

    def cancel(self, invocation_id: str) -> dict:
        self._directory(invocation_id)  # Validate before any state access.
        with self._lock:
            if self._active_id == invocation_id and self._cancel_event is not None:
                self._cancel_event.set()
                return {'status': 'cancel_requested', 'invocation_id': invocation_id}
        result = self.collect(invocation_id)
        return {'status': result['status'], 'invocation_id': invocation_id}

    def replay(self, invocation_id: str) -> dict:
        result, replay = self._load(invocation_id)
        if result['status'] != 'complete':
            return {'status': result['status'], 'invocation_id': invocation_id,
                    'model_calls': 0, 'actions_replayed': 0}
        return replay

    def close(self, *, timeout_s: float = 1.8) -> dict:
        with self._lock:
            self._closed = True
            worker = self._active_thread
            event = self._cancel_event
            if event is not None:
                event.set()
        if worker is not None:
            worker.join(timeout=timeout_s)
        with self._lock:
            clean = not self._quarantined
        return {'status': 'closed' if clean and (worker is None or not worker.is_alive())
                else 'interrupted'}
