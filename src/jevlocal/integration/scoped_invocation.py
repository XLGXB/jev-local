"""Explicit local scoped ranking; no host registration or authority transfer."""
from __future__ import annotations

import copy
import re
import uuid

from .evidence_scope import _digest, _json, scope_evidence, verify_scope_evidence


def validate_binding(binding):
    if type(binding) is not dict or set(binding) != {
            'invocation_id', 'manifest_sha256', 'rank_input_sha256'}:
        raise ValueError('scoped binding shape')
    identity = binding['invocation_id']
    if (type(identity) is not str or not re.fullmatch(
            r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', identity)):
        raise ValueError('scoped invocation ID')
    for key in ('manifest_sha256', 'rank_input_sha256'):
        value = binding[key]
        if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('scoped binding digest')


def validate_snapshot(binding, manifest, rank_input, query, candidates):
    """Check the exact sanitized child input against the full published manifest."""
    validate_binding(binding)
    if (type(manifest) is not dict or type(rank_input) is not dict or
            manifest.get('status') != 'prepared' or
            _digest(manifest) != binding['manifest_sha256'] or
            _digest(rank_input) != binding['rank_input_sha256'] or
            _digest(manifest.get('rank_input')) != _digest(rank_input) or
            rank_input.get('query') != query or
            rank_input.get('candidates') != candidates):
        raise ValueError('scoped child snapshot mismatch')


def prepare_invocation(arguments):
    """Freeze complete sanitized manifest and actual rank input in one record."""
    if type(arguments) is not dict:
        raise ValueError('scope arguments required')
    manifest = scope_evidence(**arguments)
    if manifest['status'] == 'requires_larger_budget':
        return manifest
    rank_input = copy.deepcopy(manifest['rank_input'])
    verify_scope_evidence(arguments, manifest, rank_input)
    # The invocation ID is generated here, never taken from the scope input.
    binding = {'invocation_id': str(uuid.uuid4()),
               'manifest_sha256': _digest(manifest),
               'rank_input_sha256': _digest(rank_input)}
    validate_binding(binding)
    record = {'version': 2, 'adapter': 'jevlocal-scoped-invocation-v2',
            'binding': binding, 'manifest': manifest, 'rank_input': rank_input,
            'execution_authorized': False, 'training_allowed': False,
            'origin_authenticated': False}
    if len(_json(record).encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('scoped invocation record exceeds replay limit')
    return record


def validate_invocation(arguments, record):
    if (type(record) is not dict or set(record) != {
            'version', 'adapter', 'binding', 'manifest', 'rank_input',
            'execution_authorized', 'training_allowed', 'origin_authenticated'} or
            type(record['version']) is not int or record['version'] != 2 or
            record['adapter'] != 'jevlocal-scoped-invocation-v2' or
            any(record[key] is not False for key in (
                'execution_authorized', 'training_allowed', 'origin_authenticated'))):
        raise ValueError('scoped invocation record shape')
    binding = record['binding']
    validate_binding(binding)
    verify_scope_evidence(arguments, record['manifest'], record['rank_input'])
    if (binding['manifest_sha256'] != _digest(record['manifest']) or
            binding['rank_input_sha256'] != _digest(record['rank_input'])):
        raise ValueError('scoped invocation binding mismatch')
    return binding


def replay_invocation(arguments, record, primary, sidecar=None):
    """Pure v2 consistency check; no model, action, path discovery or authority."""
    binding = validate_invocation(arguments, record)
    derived = record['rank_input']
    if derived is None:
        expected = {'status': 'no_matches', 'execution_authorized': False,
                    'training_allowed': False}
        if sidecar is not None or _digest(primary) != _digest(expected):
            raise ValueError('no_matches artifact mismatch')
        replay = None
    else:
        if sidecar is None:
            raise ValueError('scoped observer sidecar required')
        from .context_replay import load_replay
        replay = load_replay(derived, primary, sidecar, binding=binding).report()
    return {'version': 2, 'status': 'consistent', 'binding': copy.deepcopy(binding),
            'scope': {'counts': copy.deepcopy(record['manifest']['counts']),
                      'status': record['manifest']['status']},
            'replay': replay, 'model_calls': 0, 'actions_replayed': 0,
            'execution_authorized': False, 'training_allowed': False,
            'origin_authenticated': False}
