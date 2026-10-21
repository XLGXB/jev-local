"""Bounded, caller-catalog model routing advice; never changes host authority."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone


_EFFORTS = frozenset(('none', 'minimal', 'low', 'medium', 'high', 'xhigh',
                      'max', 'ultra'))
_DIFFICULTIES = frozenset(('routine', 'moderate', 'complex', 'unknown'))
_RISKS = frozenset(('low', 'medium', 'high', 'unknown'))
_ID = re.compile(r'[A-Za-z0-9_.:-]{1,64}\Z')
_MAX_INPUT = 128 * 1024
_MAX_OUTPUT = 256 * 1024
_REQUEST_KEYS = {'schema_version', 'request_id', 'current_primary', 'task',
                 'budget', 'catalog', 'profiles'}
_MODEL_KEYS = {'model_id', 'reasoning_efforts', 'description',
               'context_window_tokens', 'tools', 'estimated_cost',
               'estimated_latency_ms'}


def _json_bytes(value, label, limit):
    """Bound and own a plain JSON value, without exposing its content on error."""
    try:
        _plain_json(value)
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(f'invalid {label} JSON') from None
    if len(raw) > limit:
        raise ValueError(f'{label} too large')
    return raw


def _plain_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return
    if type(value) is list:
        for item in value:
            _plain_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('non-string JSON key')
            _plain_json(item)
        return
    raise ValueError('non-JSON value')


def _shape(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f'invalid {label} shape')


def _text(value, label, limit, *, nonempty=True):
    if (type(value) is not str or len(value) > limit or
            (nonempty and not value.strip())):
        raise ValueError(f'invalid {label}')


def _id(value, label):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f'invalid {label}')


def _positive_int(value, label, *, maximum=None):
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f'invalid {label}')


def _cost(value, label):
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f'invalid {label}')
    try:
        if not math.isfinite(value):
            raise ValueError(f'invalid {label}')
    except OverflowError:
        raise ValueError(f'invalid {label}') from None


def _optional_positive(value, label, *, maximum=None):
    if value is not None:
        _positive_int(value, label, maximum=maximum)


def _optional_cost(value, label):
    if value is not None:
        _cost(value, label)


def _ids(value, label):
    if type(value) is not list or len(value) > 32:
        raise ValueError(f'invalid {label}')
    for item in value:
        _id(item, label)
    if len(set(value)) != len(value):
        raise ValueError(f'duplicate {label}')


def _observed_at(value):
    _text(value, 'catalog observed_at', 64)
    # The UTC marker and round-trip check exclude offsets, naive dates and
    # parser conveniences such as non-ISO separators.
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z', value) is None:
        raise ValueError('invalid catalog observed_at')
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError:
        raise ValueError('invalid catalog observed_at') from None
    if parsed.tzinfo != timezone.utc:
        raise ValueError('invalid catalog observed_at')


def _validated(arguments):
    raw = _json_bytes(arguments, 'input', _MAX_INPUT)
    request = json.loads(raw)
    _shape(request, _REQUEST_KEYS, 'request')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('unsupported request schema')
    _id(request['request_id'], 'request ID')

    primary = request['current_primary']
    _shape(primary, {'model_id', 'reasoning_effort'}, 'current primary')
    _text(primary['model_id'], 'primary model ID', 128)
    if (type(primary['reasoning_effort']) is not str or
            primary['reasoning_effort'] not in _EFFORTS):
        raise ValueError('unsupported primary effort')

    task = request['task']
    _shape(task, {'summary', 'difficulty', 'risk', 'context_tokens',
                  'required_tools'}, 'task')
    _text(task['summary'], 'task summary', 8192)
    if (type(task['difficulty']) is not str or task['difficulty'] not in _DIFFICULTIES or
            type(task['risk']) is not str or task['risk'] not in _RISKS):
        raise ValueError('invalid task class')
    _optional_positive(task['context_tokens'], 'context tokens', maximum=10_000_000)
    _ids(task['required_tools'], 'required tools')

    budget = request['budget']
    _shape(budget, {'max_estimated_cost', 'max_latency_ms', 'cost_unit'}, 'budget')
    _optional_cost(budget['max_estimated_cost'], 'maximum estimated cost')
    _optional_positive(budget['max_latency_ms'], 'maximum latency')
    _text(budget['cost_unit'], 'cost unit', 32)

    catalog = request['catalog']
    _shape(catalog, {'source', 'observed_at', 'models'}, 'catalog')
    _text(catalog['source'], 'catalog source', 512, nonempty=False)
    _observed_at(catalog['observed_at'])
    models = catalog['models']
    if type(models) is not list or not 1 <= len(models) <= 32:
        raise ValueError('invalid catalog model count')
    by_model = {}
    for model in models:
        _shape(model, _MODEL_KEYS, 'catalog model')
        _text(model['model_id'], 'model ID', 128)
        if model['model_id'] in by_model:
            raise ValueError('duplicate catalog model')
        efforts = model['reasoning_efforts']
        if (type(efforts) is not list or not 1 <= len(efforts) <= 8 or
                any(type(effort) is not str or effort not in _EFFORTS
                    for effort in efforts) or len(set(efforts)) != len(efforts)):
            raise ValueError('invalid catalog efforts')
        _text(model['description'], 'model description', 2048, nonempty=False)
        _optional_positive(model['context_window_tokens'], 'context capacity',
                           maximum=10_000_000)
        if model['tools'] is not None:
            _ids(model['tools'], 'model tools')
        _optional_cost(model['estimated_cost'], 'estimated cost')
        _optional_positive(model['estimated_latency_ms'], 'estimated latency')
        by_model[model['model_id']] = model

    primary_model = by_model.get(primary['model_id'])
    if (primary_model is None or
            primary['reasoning_effort'] not in primary_model['reasoning_efforts']):
        raise ValueError('unsupported current primary pair')

    profiles = request['profiles']
    if type(profiles) is not list or not 1 <= len(profiles) <= 15:
        raise ValueError('invalid profile count')
    ids, pairs = set(), set()
    for profile in profiles:
        _shape(profile, {'id', 'model_id', 'reasoning_effort'}, 'profile')
        _id(profile['id'], 'profile ID')
        if profile['id'] == '__abstain__' or profile['id'] in ids:
            raise ValueError('duplicate or reserved profile ID')
        ids.add(profile['id'])
        _text(profile['model_id'], 'profile model ID', 128)
        effort = profile['reasoning_effort']
        if type(effort) is not str or effort not in _EFFORTS:
            raise ValueError('unsupported profile effort')
        model = by_model.get(profile['model_id'])
        if model is None or effort not in model['reasoning_efforts']:
            raise ValueError('unsupported profile pair')
        pair = (profile['model_id'], effort)
        if pair in pairs:
            raise ValueError('duplicate profile pair')
        pairs.add(pair)
    return request, hashlib.sha256(raw).hexdigest(), hashlib.sha256(
        _json_bytes(catalog, 'catalog', _MAX_INPUT)).hexdigest(), by_model


def _screen(profile, model, task, budget):
    exclusions, unknowns = [], []
    context = task['context_tokens']
    capacity = model['context_window_tokens']
    if context is not None:
        if capacity is None:
            unknowns.append('context_capacity_unknown')
        elif capacity < context:
            exclusions.append('context_capacity_insufficient')
    required = set(task['required_tools'])
    if required:
        if model['tools'] is None:
            unknowns.append('tool_support_unknown')
        elif not required.issubset(model['tools']):
            exclusions.append('required_tool_unsupported')
    maximum = budget['max_estimated_cost']
    if maximum is not None:
        if model['estimated_cost'] is None:
            unknowns.append('cost_unknown')
        elif model['estimated_cost'] > maximum:
            exclusions.append('estimated_cost_over_budget')
    maximum = budget['max_latency_ms']
    if maximum is not None:
        if model['estimated_latency_ms'] is None:
            unknowns.append('latency_unknown')
        elif model['estimated_latency_ms'] > maximum:
            exclusions.append('estimated_latency_over_budget')
    return exclusions, unknowns


def advise_model_route(arguments, *, judge_factory=None):
    """Return bounded non-authorizing advice from an explicit closed catalog.

    Caller metadata is not a current-host attestation. The judge is created only
    after complete input validation and only if a candidate survives screening.
    """
    request, request_sha, catalog_sha, by_model = _validated(arguments)
    task, budget = request['task'], request['budget']
    profiles, candidates = [], []
    metadata_unknown = False
    for profile in request['profiles']:
        model = by_model[profile['model_id']]
        exclusions, unknowns = _screen(profile, model, task, budget)
        metadata_unknown |= bool(unknowns)
        profiles.append({**profile, 'eligible': not exclusions,
                         'exclusion_reasons': exclusions, 'unknowns': unknowns})
        if not exclusions:
            candidates.append({**profile,
                               'description': model['description'],
                               'context_window_tokens': model['context_window_tokens'],
                               'tools': model['tools'],
                               'estimated_cost': model['estimated_cost'],
                               'estimated_latency_ms': model['estimated_latency_ms'],
                               'unknowns': unknowns})
    result = {
        'schema_version': 1, 'status': 'unknown',
        'request_id': request['request_id'], 'request_sha256': request_sha,
        'catalog_sha256': catalog_sha,
        'current_primary': copy.deepcopy(request['current_primary']),
        'profiles': profiles, 'selected_profile_id': None,
        'recommendation': None, 'reasons': [], 'route': {},
        'catalog_provenance': 'caller_supplied_not_host_authenticated',
        'catalog_freshness': 'not_verified', 'semantic_quality': 'unqualified',
        'calibrated': False, 'main_model_review_required': True,
        'execution_authorized': False, 'model_switch_authorized': False,
        'provider_change_authorized': False, 'current_primary_changed': False,
        'task': copy.deepcopy(task), 'budget': copy.deepcopy(budget),
        'catalog_source': request['catalog']['source'],
        'catalog_observed_at': request['catalog']['observed_at'],
    }
    if not candidates:
        semantic_reason = 'no_compatible_profile'
    elif judge_factory is None:
        semantic_reason = 'semantic_unavailable'
    else:
        prepared = {
            'request_id': request['request_id'], 'request_sha256': request_sha,
            'catalog_sha256': catalog_sha,
            'current_primary': copy.deepcopy(request['current_primary']),
            'task': copy.deepcopy(task), 'budget': copy.deepcopy(budget),
            'candidates': candidates,
        }
        _json_bytes(prepared, 'prepared judge item', _MAX_INPUT)
        try:
            judge = judge_factory()
            if not callable(judge):
                raise TypeError('non-callable judge')
            reply = judge(copy.deepcopy(prepared))
        except Exception:
            raise ValueError('semantic judge failed') from None
        _shape(reply, {'selected_profile_id', 'route'}, 'semantic reply')
        _json_bytes(reply, 'semantic reply', _MAX_OUTPUT)
        selected = reply['selected_profile_id']
        if selected is not None and (type(selected) is not str or
                                     selected not in {item['id'] for item in candidates}):
            raise ValueError('invalid semantic selection')
        if type(reply['route']) is not dict or not reply['route']:
            raise ValueError('invalid semantic route')
        result['route'] = copy.deepcopy(reply['route'])
        if selected is None:
            semantic_reason = 'semantic_abstained'
        else:
            result['selected_profile_id'] = selected
            chosen = next(item for item in candidates if item['id'] == selected)
            result['recommendation'] = {key: chosen[key] for key in
                                        ('model_id', 'reasoning_effort')}
            result['status'] = 'advisory'
            semantic_reason = 'semantic_closed_set_advice'
    result['reasons'] = [semantic_reason,
                         'task_difficulty_' + task['difficulty'],
                         'task_risk_' + task['risk']]
    if metadata_unknown:
        result['reasons'].append('metadata_unknown')
    _json_bytes(result, 'output', _MAX_OUTPUT)
    return result
