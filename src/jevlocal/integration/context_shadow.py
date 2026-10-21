"""Explicit, finite, process-isolated passive context ranking observation.

The lexical caller owns its result. This module never routes an action, trains,
or treats native relevance estimates as independently verified truth.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


_LIMIT = 1024 * 1024
_FLAGS = dict(shadow=True, execution_authorized=False,
              training_allowed=False, cloud_data_used=False)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _snapshot_hash(query, safe_candidates):
    return _digest({'query': query, 'candidates': safe_candidates})


def _envelope(query, candidate, binding=None):
    # Use precisely the current active native context Noul rubric, never a
    # Choice score normalized over a changing candidate chunk.
    from .native_context import _CRITERIA, _INSTRUCTIONS
    state = {'query': query, 'candidate': {
                    key: candidate[key] for key in ('id', 'source', 'text')}}
    request_id = 'local-context-shadow'
    if binding is not None:
        from .scoped_invocation import validate_binding
        from .evidence_scope import _digest as canonical_digest
        validate_binding(binding)
        # Attribution belongs to the envelope, not the semantic model input.
        # LocalShadow passes only state/questions to the predictor. Bind the
        # complete invocation in its request ID and therefore request_digest.
        request_id = 'local-context-shadow-v2-' + canonical_digest(binding)
    return dict(request_id=request_id, observation_id=candidate['id'],
                state=state,
                questions={'relevance': {'type': 'noul',
                    'instructions': _INSTRUCTIONS, 'criteria': dict(_CRITERIA)}})


def verify_primary(query, safe_candidates, primary_result, *, store_dir=None):
    """Check frozen lexical citations mechanically; semantic labels stay unknown."""
    from . import context_tools
    if not isinstance(primary_result, dict) or primary_result.get('status') != 'ok':
        return {'status': 'not_available', 'reasons': ['primary_not_ok']}
    reasons = []
    if (primary_result.get('method') != 'lexical' or
            primary_result.get('execution_authorized') is not False or
            not isinstance(primary_result.get('results'), list)):
        return {'status': 'failed', 'reasons': ['primary_shape']}
    try:
        expected = {c['id']: c for c in safe_candidates}
        if len(expected) != len(safe_candidates):
            return {'status': 'failed', 'reasons': ['snapshot_ids']}
        rows = primary_result['results']
        ids = [r.get('id') if isinstance(r, dict) else None for r in rows]
        if len(ids) != len(expected) or set(ids) != set(expected):
            return {'status': 'failed', 'reasons': ['result_ids']}
        for row in rows:
            cid = row['id']
            candidate = expected[cid]
            prefix = 'candidate:' + cid + ':'
            if row.get('source') != candidate['source']:
                reasons.append(prefix + 'source')
            if row.get('sha256') != _text_hash(candidate['text']):
                reasons.append(prefix + 'content_hash')
            if row.get('stored') is not (store_dir is not None):
                reasons.append(prefix + 'stored_flag')
            lines = candidate['text'].splitlines(keepends=True)
            first = row.get('restore_start_line')
            last = row.get('restore_end_line')
            start = row.get('start_line')
            end = row.get('end_line')
            if (type(first) is not int or type(last) is not int or first != last or
                    first < 1 or first > len(lines) or type(start) is not int or
                    type(end) is not int or start != end or
                    start != candidate['start_line'] + first - 1):
                reasons.append(prefix + 'line_range')
                continue
            if row.get('quote') != lines[first - 1]:
                reasons.append(prefix + 'quote')
            if store_dir is not None:
                try:
                    restored = context_tools.restore_lines(
                        store_dir, _text_hash(candidate['text']), first, last)
                    if restored != lines[first - 1]:
                        reasons.append(prefix + 'stored_quote')
                except (OSError, UnicodeError, ValueError):
                    reasons.append(prefix + 'store_integrity')
    except (KeyError, TypeError, ValueError):
        reasons.append('primary_shape')
    return {'status': 'failed' if reasons else 'verified', 'reasons': sorted(set(reasons)),
            'semantic_relevance': 'unknown'}


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError('nonfinite JSON value')


def _validate_output(payload, query, safe_candidates, backend, *, binding=None):
    """Treat every child byte as untrusted, including its native receipts."""
    from decision_deploy.protocol import validate_reply
    from .shadow import request_digest
    if binding is not None:
        from .scoped_invocation import validate_binding
        validate_binding(binding)
    version = 2 if binding is not None else 1
    fields = {'version', 'snapshot_sha256', 'receipts'} | ({'binding'} if binding is not None else set())
    if (not isinstance(payload, dict) or set(payload) != fields or
            type(payload['version']) is not int or payload['version'] != version or
            (binding is not None and payload['binding'] != binding) or
            payload['snapshot_sha256'] != _snapshot_hash(query, safe_candidates) or
            not isinstance(payload['receipts'], list) or
            len(payload['receipts']) != len(safe_candidates)):
        raise ValueError('shadow output binding')
    expected = {c['id']: c for c in safe_candidates}
    if len(expected) != len(safe_candidates):
        raise ValueError('duplicate snapshot ID')
    found = set()
    fingerprint_seen = None
    for item in payload['receipts']:
        if not isinstance(item, dict) or set(item) != {'id', 'record'}:
            raise ValueError('shadow receipt shape')
        cid, record = item['id'], item['record']
        if not isinstance(cid, str) or cid not in expected or cid in found or not isinstance(record, dict):
            raise ValueError('shadow receipt ID')
        found.add(cid)
        envelope = _envelope(query, expected[cid], binding)
        base = {'schema_version', 'request_id', 'observation_id', 'request_sha256',
                'candidate_keys', 'backend', 'runtime_fingerprint', 'status', 'answers',
                'shadow', 'execution_authorized', 'training_allowed', 'cloud_data_used'}
        optional = {'error_type', 'unavailable_reason', 'shadow_elapsed_ms'}
        if not base <= set(record) or set(record) - base - optional:
            raise ValueError('shadow receipt fields')
        if (type(record['schema_version']) is not int or record['schema_version'] != 1 or
                record['request_id'] != envelope['request_id'] or
                record['observation_id'] != cid or record['request_sha256'] != request_digest(envelope) or
                record['candidate_keys'] != {'relevance': ['false', 'true']} or
                record['backend'] != backend or backend not in ('laya', 'kev') or
                any(record.get(key) is not value for key, value in _FLAGS.items())):
            raise ValueError('shadow receipt binding')
        if 'shadow_elapsed_ms' in record and (type(record['shadow_elapsed_ms']) not in (int, float) or
                not math.isfinite(record['shadow_elapsed_ms']) or record['shadow_elapsed_ms'] < 0):
            raise ValueError('shadow elapsed')
        fingerprint = record['runtime_fingerprint']
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
                any(char not in '0123456789abcdef' for char in fingerprint)):
            raise ValueError('shadow runtime fingerprint')
        if fingerprint_seen is None:
            fingerprint_seen = fingerprint
        elif fingerprint_seen != fingerprint:
            raise ValueError('shadow runtime changed')
        if record['status'] == 'observed':
            if (set(record) & {'error_type', 'unavailable_reason'} or
                    'shadow_elapsed_ms' not in record or not isinstance(record['answers'], dict)):
                raise ValueError('observed shadow shape')
            validate_reply(envelope, {'request_id': envelope['request_id'],
                                      'observation_id': cid,
                                      'runtime_fingerprint': fingerprint,
                                      'answers': record['answers']})
            answer = record['answers']['relevance']
            if answer['calibrated'] is not False or answer.get('temperature') != 1.0:
                raise ValueError('shadow calibration')
        elif record['status'] == 'unavailable':
            if record['answers'] != {}:
                raise ValueError('unavailable shadow answers')
            if 'error_type' in record and (not isinstance(record['error_type'], str) or
                    not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', record['error_type'])):
                raise ValueError('unavailable error shape')
            if 'unavailable_reason' in record and record['unavailable_reason'] not in (
                    'closed', 'saturated', 'queue_expired', 'inference_failed',
                    'timeout', 'cancelled'):
                raise ValueError('unavailable reason shape')
        else:
            raise ValueError('shadow status')
    if found != set(expected):
        raise ValueError('missing shadow receipts')
    return payload


class _Job:
    def __init__(self, query, safe_candidates, backend, started, timeout_s,
                 temp_path=None, process=None, unavailable_reason=None, binding=None):
        self._query = query
        self._safe = safe_candidates
        self._binding = copy.deepcopy(binding)
        self._backend = backend
        self._deadline = started + timeout_s
        self._temp_path = temp_path
        self._process = process
        self._reason = unavailable_reason
        self._closed = False
        self._finished = None
        self._pid = getattr(process, 'pid', None)
        self._exit_code = None
        self._reaped = False
        self._timed_out = False
        self._temporary_cleaned = temp_path is None

    def _stop(self):
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            self._exit_code = proc.wait(timeout=1.0)
            self._reaped = True
            self._process = None
        except (OSError, subprocess.TimeoutExpired):
            self._reaped = False

    def close(self):
        if self._closed:
            return self._close_status()
        if self._reason is None and self._finished is None:
            self._reason = 'cancelled'
        self._stop()
        if self._process is None and not self._temporary_cleaned and self._temp_path is not None:
            try:
                root = self._temp_path.resolve()
                # Only the root created by this job is removed; no parent walk.
                if root == self._temp_path and root.is_dir():
                    shutil.rmtree(root)
                self._temporary_cleaned = not self._temp_path.exists()
            except OSError:
                self._temporary_cleaned = False
        self._closed = self._process is None and self._temporary_cleaned
        return self._close_status()

    def _close_status(self):
        return {'closed': self._closed, 'reaped': self._reaped,
                'temporary_cleaned': self._temporary_cleaned}

    def finish(self, primary_result, *, store_dir=None, cancel_event=None):
        if self._finished is not None:
            return copy.deepcopy(self._finished)
        feedback = verify_primary(self._query, self._safe, primary_result,
                                  store_dir=store_dir)
        try:
            primary_hash = _digest(primary_result)
        except (TypeError, ValueError, OverflowError, RecursionError):
            primary_hash = None
        receipt = None
        reason = self._reason
        try:
            proc = self._process
            if proc is not None and reason is None:
                remaining = self._deadline - time.monotonic()
                if cancel_event is not None and cancel_event.is_set():
                    reason = 'cancelled'
                elif remaining <= 0:
                    reason = 'timeout'
                    self._timed_out = True
                else:
                    try:
                        if cancel_event is None:
                            code = proc.wait(timeout=remaining)
                        else:
                            # Only this finish owner touches the process. Other
                            # threads signal intent; none may close the job.
                            while True:
                                if cancel_event.is_set():
                                    reason = 'cancelled'
                                    break
                                remaining = self._deadline - time.monotonic()
                                if remaining <= 0:
                                    reason = 'timeout'
                                    self._timed_out = True
                                    break
                                try:
                                    code = proc.wait(timeout=min(remaining, 0.05))
                                    break
                                except subprocess.TimeoutExpired:
                                    continue
                    except subprocess.TimeoutExpired:
                        reason = 'timeout'
                        self._timed_out = True
                    except OSError:
                        reason = 'worker_failed'
                    else:
                        if reason is None:
                            self._exit_code = code
                            if cancel_event is not None and cancel_event.is_set():
                                reason = 'cancelled'
                            elif time.monotonic() > self._deadline:
                                reason = 'timeout'
                                self._timed_out = True
                            elif code != 0:
                                reason = 'worker_failed'
                if cancel_event is not None and cancel_event.is_set():
                    reason = 'cancelled'
                if reason is None:
                    output_path = self._temp_path / 'result.json'
                    try:
                        if output_path.stat().st_size > _LIMIT:
                            raise ValueError('oversized')
                        with output_path.open('rb') as stream:
                            raw = stream.read(_LIMIT + 1)
                        if len(raw) > _LIMIT:
                            raise ValueError('oversized')
                        decoded = json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_pairs,
                                             parse_constant=_reject_constant)
                        receipt = _validate_output(decoded, self._query, self._safe, self._backend,
                                                   binding=self._binding)
                    except (OSError, UnicodeError, ValueError, TypeError):
                        reason = 'invalid_output'
        finally:
            cleanup = self.close()
            if not cleanup['closed'] or (self._pid is not None and not cleanup['reaped']):
                receipt = None
                reason = 'cleanup_failed'
        if cancel_event is not None and cancel_event.is_set() and reason != 'cleanup_failed':
            receipt = None
            reason = 'cancelled'
        observed = []
        if receipt is not None:
            observed = [item['record'] for item in receipt['receipts']
                        if item['record']['status'] == 'observed']
            if not observed:
                reason = 'native_unavailable'
        shadow_ranking = sorted(({'id': record['observation_id'],
                           'probability_true': record['answers']['relevance']['probabilities']['true']}
                          for record in observed),
                         key=lambda row: (-row['probability_true'], row['id']))
        host_ranking = []
        primary_bindings = []
        safe_ids = {candidate['id'] for candidate in self._safe}
        if isinstance(primary_result, dict) and primary_result.get('status') == 'ok':
            rows = primary_result.get('results')
            if isinstance(rows, list):
                host_ranking = [{'id': row['id'], 'score': row['score']}
                                for row in rows if isinstance(row, dict) and
                                isinstance(row.get('id'), str) and row['id'] in safe_ids and
                                type(row.get('score')) in (int, float) and
                                math.isfinite(row['score'])]
                primary_bindings = [
                    {'id': row['id'], 'quote_sha256': _text_hash(row['quote']),
                     'start_line': row['start_line'], 'end_line': row['end_line'],
                     'restore_start_line': row['restore_start_line'],
                     'restore_end_line': row['restore_end_line']}
                    for row in rows if isinstance(row, dict) and
                    isinstance(row.get('id'), str) and row['id'] in safe_ids and
                    isinstance(row.get('quote'), str) and
                    all(type(row.get(key)) is int for key in (
                        'start_line', 'end_line', 'restore_start_line', 'restore_end_line'))]
        sidecar = {'version': 1, 'adapter': 'jevlocal-context-shadow-v1',
                   'snapshot_sha256': _snapshot_hash(self._query, self._safe),
                   'primary_result_sha256': primary_hash,
                   'primary_bindings': primary_bindings,
                   'candidates': [{'id': c['id'], 'source_sha256': _text_hash(c['source']),
                                   'text_sha256': _text_hash(c['text']),
                                   'start_line': c['start_line']} for c in self._safe],
                   'availability': {'status': 'observed' if observed else 'unavailable',
                                    'observed_count': len(observed),
                                    'candidate_count': len(self._safe),
                                    'reason': None if observed else reason},
                   'native_receipts': copy.deepcopy(receipt['receipts']) if receipt else [],
                   'host_ranking': host_ranking,
                   'host_top': host_ranking[0]['id'] if host_ranking else None,
                   'shadow_ranking': shadow_ranking,
                   'shadow_top': shadow_ranking[0]['id'] if shadow_ranking else None,
                   'lifecycle': {'pid': self._pid, 'exit_code': self._exit_code,
                                 'timed_out': self._timed_out, 'reaped': self._reaped,
                                 'closed': self._closed,
                                 'temporary_cleaned': self._temporary_cleaned},
                   'mechanical_feedback': feedback, **_FLAGS}
        if self._binding is not None:
            sidecar['version'] = 2
            sidecar['adapter'] = 'jevlocal-context-shadow-v2'
            sidecar['binding'] = copy.deepcopy(self._binding)
        self._finished = sidecar
        return copy.deepcopy(sidecar)


def start_shadow(query, safe_candidates, spec, *, timeout_s=60.0, binding=None,
                 manifest=None, rank_input=None):
    """Start one hidden observer before lexical computation, without blocking for inference."""
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
        raise ValueError('finite shadow timeout in (0, 60] required')
    if binding is not None:
        from .scoped_invocation import validate_snapshot
        validate_snapshot(binding, manifest, rank_input, query, safe_candidates)
    elif manifest is not None or rank_input is not None:
        raise ValueError('unbound scoped snapshot')
    started = time.monotonic()
    query = copy.deepcopy(query)
    safe_candidates = copy.deepcopy(safe_candidates)
    spec = copy.deepcopy(spec)
    backend = spec.get('backend') if isinstance(spec, dict) else None
    job = _Job(query, safe_candidates, backend, started, float(timeout_s), binding=binding)
    try:
        payload = {'version': 2 if binding is not None else 1, 'query': query,
                   'safe_candidates': safe_candidates, 'spec': spec}
        if binding is not None:
            payload['binding'] = copy.deepcopy(binding)
            payload['manifest'] = copy.deepcopy(manifest)
            payload['rank_input'] = copy.deepcopy(rank_input)
        payload = _json_bytes(payload)
        if len(payload) > _LIMIT:
            job._reason = 'capacity'
            return job
        from .native_context import validate_native_spec
        validate_native_spec(spec)
        root = Path(tempfile.mkdtemp(prefix='jevlocal-context-shadow-')).resolve()
        job._temp_path = root
        job._temporary_cleaned = False
        path = root / 'payload.json'
        with path.open('xb') as stream:
            stream.write(payload)
        if time.monotonic() >= job._deadline:
            job._reason = 'timeout'
            job._timed_out = True
            job.close()
            return job
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        job._process = subprocess.Popen(
            [sys.executable, '-m', 'jevlocal.integration.context_shadow', '--worker',
             str(path), '--result', str(root / 'result.json')],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=flags)
        job._pid = getattr(job._process, 'pid', None)
    except (OSError, TypeError, ValueError, OverflowError, RecursionError):
        job._reason = 'startup_failed'
        job.close()
    return job


def unavailable_shadow(query, safe_candidates, spec, *, reason='startup_failed', binding=None):
    """Represent an unexpected optional observer failure without a second route."""
    if reason not in ('startup_failed', 'capacity', 'worker_failed', 'cancelled'):
        raise ValueError('unsupported observer failure reason')
    frozen_query = copy.deepcopy(query)
    frozen_safe = copy.deepcopy(safe_candidates)
    backend = spec.get('backend') if isinstance(spec, dict) else None
    return _Job(frozen_query, frozen_safe, backend, time.monotonic(), 0,
                unavailable_reason=reason, binding=binding)


def _child(payload_path, result_path):
    # Process-only flags precede all native dependency imports. Deny network
    # socket connections even if a downstream library tries to ignore offline.
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'):
        os.environ[key] = '1'
    import socket
    def denied(*_args, **_kwargs):
        raise OSError('offline shadow')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    from .native_context import build_native_predictor
    from .shadow_host import ShadowHost
    from .shadow import request_digest
    if Path(payload_path).stat().st_size > _LIMIT:
        raise ValueError('payload capacity')
    with Path(payload_path).open('rb') as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError('payload capacity')
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_pairs,
                      parse_constant=_reject_constant)
    if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] not in (1, 2):
        raise ValueError('payload shape')
    binding = data.get('binding') if data['version'] == 2 else None
    fields = {'version', 'query', 'safe_candidates', 'spec'} | (
        {'binding', 'manifest', 'rank_input'} if binding is not None else set())
    if set(data) != fields or (data['version'] == 2 and not isinstance(binding, dict)):
        raise ValueError('payload shape')
    if binding is not None:
        from .scoped_invocation import validate_snapshot
        validate_snapshot(binding, data['manifest'], data['rank_input'],
                          data['query'], data['safe_candidates'])
    query, candidates = data['query'], data['safe_candidates']
    if not isinstance(query, str) or not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
        raise ValueError('snapshot shape')
    predictor = build_native_predictor(data['spec'])
    if predictor.calibration is not None:
        raise ValueError('shadow requires uncalibrated predictor')
    host = ShadowHost(predictor, max_pending=1)
    receipts = []
    try:
        for candidate in candidates:
            envelope = _envelope(query, candidate, binding)
            ticket = host.begin(envelope, timeout_s=60.0)
            record = host.collect(ticket).receipt()
            receipts.append({'id': candidate['id'], 'record': record})
    finally:
        host.close(timeout_s=0)
    output_value = {'version': data['version'], 'snapshot_sha256': _snapshot_hash(query, candidates),
                    'receipts': receipts}
    if binding is not None:
        output_value['binding'] = binding
    output = _json_bytes(output_value)
    if len(output) > _LIMIT:
        raise ValueError('result capacity')
    with Path(result_path).open('xb') as stream:
        stream.write(output)


def _main():
    if len(sys.argv) != 5 or sys.argv[1] != '--worker' or sys.argv[3] != '--result':
        return 2
    try:
        _child(sys.argv[2], sys.argv[4])
        return 0
    except Exception:
        return 1


if __name__ == '__main__':
    raise SystemExit(_main())
