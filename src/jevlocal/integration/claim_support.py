"""Local source-bound claim-support consumer; all judgments remain advisory."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re


_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
_LABELS = ('supports', 'contradicts', 'mixed', 'insufficient')
_INPUT_KEYS = {'schema_version', 'request_id', 'claim', 'sources'}
_SOURCE_KEYS = {'id', 'source_ref', 'start_line', 'text'}
_MAX_INPUT = 131072
_MAX_OUTPUT = 262144


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return
    if type(value) is list:
        for item in value:
            _plain(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('non-string JSON key')
            _plain(item)
        return
    raise ValueError('non-JSON value')


def _canonical(value, label, maximum):
    try:
        _plain(value)
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError(f'invalid {label} JSON') from None
    if len(raw) > maximum:
        raise ValueError(f'{label} too large')
    return raw


def _shape(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f'invalid {label} shape')


def _string(value, label, maximum, *, nonblank=True):
    if type(value) is not str or len(value) > maximum or (nonblank and not value.strip()):
        raise ValueError(f'invalid {label}')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError(f'invalid {label} encoding') from None


def _identifier(value, label):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f'invalid {label}')


def _validated(arguments):
    raw = _canonical(arguments, 'input', _MAX_INPUT)
    request = json.loads(raw)
    _shape(request, _INPUT_KEYS, 'request')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('unsupported request schema')
    _identifier(request['request_id'], 'request ID')
    _string(request['claim'], 'claim', 8192)
    sources = request['sources']
    if type(sources) is not list or len(sources) > 15:
        raise ValueError('invalid source count')
    ids, prepared, citations = set(), [], []
    for source in sources:
        _shape(source, _SOURCE_KEYS, 'source')
        _identifier(source['id'], 'source ID')
        if source['id'] in ids:
            raise ValueError('duplicate source ID')
        ids.add(source['id'])
        _string(source['source_ref'], 'source reference', 2048)
        line = source['start_line']
        if type(line) is not int or not 1 <= line <= 1_000_000_000:
            raise ValueError('invalid start line')
        _string(source['text'], 'source text', 8192)
        end = line + max(1, len(source['text'].splitlines())) - 1
        if end > 1_000_000_000:
            raise ValueError('invalid end line')
        digest = hashlib.sha256(source['text'].encode('utf-8')).hexdigest()
        prepared.append({'request_id': request['request_id'], 'source_id': source['id'],
                         'claim': request['claim'], 'source_ref': source['source_ref'],
                         'start_line': line, 'end_line': end, 'text': source['text'],
                         'text_sha256': digest})
        citations.append({'source_ref': source['source_ref'], 'start_line': line,
                          'end_line': end, 'text': source['text'], 'sha256': digest})
    return copy.deepcopy(request), raw, prepared, citations


def _probability(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'invalid {label}')
    return float(value)


def _judge_result(reply, prepared):
    reply_raw = _canonical(reply, 'judge reply', _MAX_OUTPUT)
    reply = json.loads(reply_raw)
    _shape(reply, {'decisions', 'route'}, 'judge reply')
    decisions, expected = reply['decisions'], {item['source_id'] for item in prepared}
    if type(decisions) is not dict or set(decisions) != expected:
        raise ValueError('invalid judge decisions')
    route = reply['route']
    if type(route) is not dict or not route:
        raise ValueError('invalid judge route')
    route_raw = _canonical(route, 'route', _MAX_OUTPUT)
    del route_raw
    result = {}
    for source_id in expected:
        decision = decisions[source_id]
        _shape(decision, {'label', 'probabilities', 'p_selected'}, 'decision')
        label = decision['label']
        if type(label) is not str or label not in _LABELS:
            raise ValueError('invalid decision label')
        probabilities = decision['probabilities']
        _shape(probabilities, set(_LABELS), 'probabilities')
        probabilities = {key: _probability(probabilities[key], 'probability')
                         for key in _LABELS}
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError('invalid probability sum')
        selected = _probability(decision['p_selected'], 'selected probability')
        if abs(selected - probabilities[label]) > 1e-6:
            raise ValueError('selected probability mismatch')
        if probabilities[label] < max(probabilities.values()):
            raise ValueError('selected label is not maximal')
        result[source_id] = (label, probabilities, selected)
    return result, copy.deepcopy(route)


def verify_claim_support(arguments, *, judge_factory=None) -> dict:
    """Validate once, bind one judge decision to each caller-supplied excerpt."""
    request, raw, prepared, citations = _validated(arguments)
    report_results = []
    if prepared and judge_factory is not None:
        try:
            judge = judge_factory()
            if not callable(judge):
                raise ValueError('semantic judge failed')
            reply = judge(copy.deepcopy(prepared))
            decisions, route = _judge_result(reply, prepared)
        except Exception as exc:
            raise ValueError('semantic judge failed') from exc
        for item, citation in zip(prepared, citations):
            label, probabilities, selected = decisions[item['source_id']]
            report_results.append({'source_id': item['source_id'], 'label': label,
                                   'citation': citation, 'probabilities': probabilities,
                                   'p_selected': selected})
        labels = [entry['label'] for entry in report_results]
        has_support, has_contradiction = 'supports' in labels, 'contradicts' in labels
        if 'mixed' in labels or (has_support and has_contradiction):
            status, reason = 'conflicting', 'conflicting_sources'
        elif has_support:
            status, reason = 'supported', 'source_evidence'
        elif has_contradiction:
            status, reason = 'contradicted', 'source_evidence'
        else:
            status, reason = 'unknown', 'insufficient_evidence'
    else:
        route = None
        for item, citation in zip(prepared, citations):
            report_results.append({'source_id': item['source_id'], 'label': 'unknown',
                                   'citation': citation, 'probabilities': None,
                                   'p_selected': None})
        status = 'unknown'
        reason = ('no_sources' if not prepared else 'semantic_unavailable')
    report = {'schema_version': 1, 'request_id': request['request_id'],
              'request_sha256': hashlib.sha256(raw).hexdigest(), 'claim': request['claim'],
              'status': status, 'reason': reason, 'source_results': report_results,
              'route': route, 'semantic_quality': 'unqualified', 'calibrated': False,
              'provenance': 'caller_supplied_unverified', 'main_model_review_required': True,
              'execution_authorized': False, 'file_write_authorized': False,
              'deletion_authorized': False}
    _canonical(report, 'report', _MAX_OUTPUT)
    return report
