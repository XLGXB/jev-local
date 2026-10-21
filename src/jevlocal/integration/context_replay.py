"""Read-only replay of explicit passive context observations, not action replay.

Artifact consistency is not authenticated history or independent semantic truth.
Original store paths are never followed implicitly. No native model, cloud API,
training converter, action executor, or JSON-specified verifier is invoked.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import context_tools as context
from .context_shadow import (_FLAGS, _digest, _envelope, _json_bytes,
    _reject_constant, _snapshot_hash, _strict_pairs, _text_hash, _validate_output,
    verify_primary)
from .shadow import ShadowObservation, assess_feedback, request_digest

_LIMIT = 4 * 1024 * 1024
_FIELDS = {'version', 'adapter', 'snapshot_sha256', 'primary_result_sha256',
    'primary_bindings', 'candidates', 'availability', 'native_receipts',
    'host_ranking', 'host_top', 'shadow_ranking', 'shadow_top', 'lifecycle',
    'mechanical_feedback', *_FLAGS}
_UNAVAILABLE = {'startup_failed', 'capacity', 'worker_failed', 'timeout',
    'invalid_output', 'cleanup_failed', 'cancelled'}


def _equal(actual, expected, reason):
    # JSON type-aware equality: True must not stand in for 1. Object formatting
    # may differ; ordered arrays and the original snapshot digest may not.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False):
        raise ValueError(reason)


def _bounded_copy(value):
    try:
        raw = _json_bytes(value)
        if len(raw) > _LIMIT or not isinstance(value, dict):
            raise ValueError('bounded object artifact required')
        return copy.deepcopy(value)
    except (RecursionError, OverflowError):
        raise ValueError('artifact nesting exceeds replay limit') from None


def _lifecycle(value, has_receipts):
    keys = {'pid', 'exit_code', 'timed_out', 'reaped', 'closed', 'temporary_cleaned'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('lifecycle shape')
    for key in keys - {'pid', 'exit_code'}:
        if type(value[key]) is not bool:
            raise ValueError('lifecycle flag')
    pid, code = value['pid'], value['exit_code']
    if (pid is not None and (type(pid) is not int or pid <= 0) or
            code is not None and type(code) is not int):
        raise ValueError('lifecycle process identity')
    if ((pid is None and (code is not None or value['reaped'])) or
            (value['reaped'] and code is None) or
            (value['closed'] and (not value['temporary_cleaned'] or
                                 pid is not None and not value['reaped']))):
        raise ValueError('lifecycle contradiction')
    if has_receipts and (pid is None or code != 0 or value['timed_out'] or
                        not all(value[k] for k in ('closed', 'reaped', 'temporary_cleaned'))):
        raise ValueError('native receipts lack a completed observer')


def _reported_mechanics(value, primary_ok, candidate_ids):
    # Historical store state cannot be reconstructed from a claimed status.
    # Validate shape only; current checks are reported separately below.
    if not isinstance(value, dict):
        raise ValueError('mechanical feedback shape')
    if not primary_ok:
        _equal(value, {'status': 'not_available', 'reasons': ['primary_not_ok']},
               'mechanical feedback unavailable')
        return
    reasons = {'primary_shape', 'snapshot_ids', 'result_ids'} | {
        'candidate:' + cid + ':' + suffix for cid in candidate_ids for suffix in (
            'source', 'content_hash', 'stored_flag', 'line_range', 'quote',
            'stored_quote', 'store_integrity')}
    if (set(value) != {'status', 'reasons', 'semantic_relevance'} or
            value['semantic_relevance'] != 'unknown' or
            value['status'] not in ('verified', 'failed') or
            not isinstance(value['reasons'], list) or len(value['reasons']) > 1024 or
            any(not isinstance(x, str) or x not in reasons for x in value['reasons']) or
            bool(value['reasons']) != (value['status'] == 'failed')):
        raise ValueError('mechanical feedback cannot supply semantic truth')


class ContextReplay:
    """Frozen validated replay; construct through load_replay/read_replay_files.

    report() excludes source text. assess() returns a separate feedback record;
    it never promotes that record to a training sample or changes the report.
    Verifiers are trusted in-process host code, not file-selected plugins.
    """
    def __init__(self, report, observations):
        self._report = copy.deepcopy(report)
        self._observations = observations

    def report(self):
        return copy.deepcopy(self._report)

    def assess(self, candidate_id, evidence, *, verifier, verifier_id):
        if not isinstance(candidate_id, str) or candidate_id not in self._observations:
            raise ValueError('no recorded native observation for candidate')
        result = assess_feedback(self._observations[candidate_id], evidence,
                                 verifier=verifier, verifier_id=verifier_id)
        result['replay_binding'] = {key: copy.deepcopy(self._report[key]) for key in (
            'snapshot_sha256', 'primary_result_sha256', 'sidecar_sha256',
            'artifact_sha256', 'origin_authenticated', 'event_time_known')}
        return result


def load_replay(arguments, primary, sidecar, *, store_dir=None, binding=None):
    """Validate a v1 observation and re-run only its pure lexical computation.

    store_dir is an explicit read capability, separate from the path recorded
    in arguments. A missing/changed explicitly requested store rejects replay.
    Without it, candidate bytes/citations can still be checked in memory, but
    no claim about current store availability is made. Artifacts <=4 MiB each.
    """
    arguments, primary, sidecar = map(_bounded_copy, (arguments, primary, sidecar))
    if (set(arguments) - {'query', 'candidates', 'max_chars', 'known_secrets', 'store_dir'} or
            not {'query', 'candidates', 'max_chars'} <= set(arguments)):
        raise ValueError('rank request data only')
    recorded_store = arguments.get('store_dir')
    if recorded_store is not None and (not isinstance(recorded_store, str) or
                                      not recorded_store.strip() or len(recorded_store) > 32768):
        raise ValueError('recorded store path shape')
    query, safe = context._prepare_rank_evidence(**arguments)
    if binding is not None:
        from .scoped_invocation import validate_binding
        validate_binding(binding)
    version = 2 if binding is not None else 1
    fields = _FIELDS | ({'binding'} if binding is not None else set())
    if (set(sidecar) != fields or type(sidecar['version']) is not int or
            sidecar['version'] != version or
            sidecar['adapter'] != f'jevlocal-context-shadow-v{version}' or
            (binding is not None and sidecar['binding'] != binding) or
            any(sidecar[k] is not v for k, v in _FLAGS.items())):
        raise ValueError('passive sidecar binding required')
    snapshot_hash = _snapshot_hash(query, safe)
    if (sidecar['snapshot_sha256'] != snapshot_hash or
            sidecar['primary_result_sha256'] != _digest(primary)):
        raise ValueError('artifact binding mismatch')
    _equal(sidecar['candidates'], [{'id': c['id'], 'source_sha256': _text_hash(c['source']),
        'text_sha256': _text_hash(c['text']), 'start_line': c['start_line']} for c in safe],
        'candidate binding mismatch')

    # Never call the persistence branch. Restore the recorded stored *flag*
    # before applying the original budget, since true/false differ in length.
    expected = context._rank_prepared_evidence(query, safe, max_chars=10**9)
    for row in expected['results']:
        row['stored'] = recorded_store is not None
    required = context._size(expected)
    if required > arguments['max_chars']:
        expected = context._too_large(required, arguments['max_chars'])
    _equal(primary, expected, 'lexical primary does not reproduce')
    primary_ok = primary.get('status') == 'ok'
    rows = primary.get('results', [])
    host = [{'id': row['id'], 'score': row['score']} for row in rows]
    _equal(sidecar['host_ranking'], host, 'host ranking mismatch')
    _equal(sidecar['host_top'], host[0]['id'] if host else None, 'host top mismatch')
    _equal(sidecar['primary_bindings'], [
        {'id': row['id'], 'quote_sha256': _text_hash(row['quote']),
         **{key: row[key] for key in ('start_line', 'end_line', 'restore_start_line', 'restore_end_line')}}
        for row in rows], 'citation binding mismatch')
    _reported_mechanics(sidecar['mechanical_feedback'], primary_ok, [c['id'] for c in safe])
    store_check = 'not_requested'
    if store_dir is not None:
        if recorded_store is None or not primary_ok:
            raise ValueError('no recorded store citations to recheck')
        current = verify_primary(query, safe, primary, store_dir=store_dir)
        if current['status'] != 'verified':
            raise ValueError('current store failed independent integrity check')
        store_check = 'verified'

    receipts = sidecar['native_receipts']
    if not isinstance(receipts, list):
        raise ValueError('native receipts list required')
    _lifecycle(sidecar['lifecycle'], bool(receipts))
    if receipts:
        first = receipts[0]
        if not isinstance(first, dict) or not isinstance(first.get('record'), dict):
            raise ValueError('native receipt shape')
        payload = {'version': version, 'snapshot_sha256': snapshot_hash, 'receipts': receipts}
        if binding is not None:
            payload['binding'] = binding
        _validate_output(payload, query, safe, first['record'].get('backend'), binding=binding)
        # Explicitly exclude bool despite Python's True == 1.0 equivalence.
        for item in receipts:
            if item['record']['status'] == 'observed':
                temperature = item['record']['answers']['relevance']['temperature']
                if type(temperature) not in (int, float) or temperature != 1.0:
                    raise ValueError('uncalibrated numeric temperature required')
    observed = [x['record'] for x in receipts if x['record']['status'] == 'observed']
    reason = None
    if not observed:
        reason = 'native_unavailable' if receipts else None
        if not receipts and isinstance(sidecar['availability'], dict):
            reason = sidecar['availability'].get('reason')
    if not receipts and reason not in _UNAVAILABLE:
        raise ValueError('missing observation reason')
    _equal(sidecar['availability'], {'status': 'observed' if observed else 'unavailable',
        'observed_count': len(observed), 'candidate_count': len(safe), 'reason': reason},
        'availability denominator mismatch')
    if reason == 'timeout' and not sidecar['lifecycle']['timed_out']:
        raise ValueError('timeout lifecycle mismatch')
    if reason == 'cleanup_failed' and sidecar['lifecycle']['closed']:
        raise ValueError('cleanup lifecycle mismatch')
    ranked = sorted(({'id': x['observation_id'],
                      'probability_true': x['answers']['relevance']['probabilities']['true']}
                     for x in observed), key=lambda x: (-x['probability_true'], x['id']))
    _equal(sidecar['shadow_ranking'], ranked, 'shadow ranking mismatch')
    _equal(sidecar['shadow_top'], ranked[0]['id'] if ranked else None, 'shadow top mismatch')

    candidates = {c['id']: c for c in safe}
    observations = {item['id']: ShadowObservation(_envelope(query, candidates[item['id']], binding),
                                                  item['record']) for item in receipts}
    report = {'version': version, 'status': 'consistent', 'replayed_at': datetime.now(timezone.utc).isoformat(),
        'snapshot_sha256': snapshot_hash, 'primary_result_sha256': _digest(primary),
        'sidecar_sha256': _digest(sidecar), 'artifact_sha256': {},
        'primary_reproduced': True, 'citation_content_verified': primary_ok,
        'store_check': store_check, 'recorded_mechanical_feedback': sidecar['mechanical_feedback'],
        'availability': sidecar['availability'], 'semantic_labels': 'unknown',
        'origin_authenticated': False, 'event_time_known': False,
        'actions_replayed': 0, 'model_calls': 0,
        'candidate_bindings': [{'id': c['id'], 'request_sha256': request_digest(_envelope(query, c, binding)),
                                'text_sha256': _text_hash(c['text'])} for c in safe], **_FLAGS}
    if binding is not None:
        report['binding'] = copy.deepcopy(binding)
    return ContextReplay(report, observations)


def read_replay_files(input_path, primary_path, shadow_path, *, store_dir=None):
    """Read exactly the three supplied artifacts; no discovery or implicit paths."""
    values, hashes = [], {}
    for name, path in zip(('input', 'primary', 'shadow'), (input_path, primary_path, shadow_path)):
        with Path(path).open('rb') as stream:
            raw = stream.read(_LIMIT + 1)
        if len(raw) > _LIMIT:
            raise ValueError('artifact exceeds replay limit')
        try:
            values.append(json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_strict_pairs,
                                     parse_constant=_reject_constant))
        except RecursionError:
            raise ValueError('artifact nesting exceeds replay limit') from None
        hashes[name] = hashlib.sha256(raw).hexdigest()
    replay = load_replay(*values, store_dir=store_dir)
    replay._report['artifact_sha256'] = hashes
    return replay
