"""F04 bounded actual diff and advisory review; never an authorization path."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import stat

from .completion_verification import (
    _check_components, _hash_string, _id, _local_path, _plain_json_types,
    _serialized, _shape, _version, _STAT_REPARSE,
)


_MAX_INPUT = 128 * 1024
_MAX_SOURCE = 256 * 1024
_MAX_CONTRACT = 32 * 1024
_MAX_TOTAL = 2 * 1024 * 1024
_MAX_ITEM = 120 * 1024
_MAX_OUTPUT = 256 * 1024
_SENSITIVE = {'security', 'permissions', 'interface', 'persistence'}
_CATEGORIES = _SENSITIVE | {'general'}
_DEVICE = re.compile(r'(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value):
    return _digest(_serialized(value, 'canonical value', _MAX_TOTAL))


def _relative(value):
    if (type(value) is not str or not value or '\x00' in value or '\\' in value or
            value.startswith('/') or ':' in value or
            any(part in ('', '.', '..') or part.endswith(('.', ' ')) or
                _DEVICE.fullmatch(part) or any(ord(char) < 32 for char in part)
                for part in value.split('/'))):
        raise ValueError('noncanonical workspace-relative path')
    return value


def _read(path, limit, remaining):
    """Read one checked regular file and reject an observed mid-read mutation."""
    try:
        before = path.stat()
        if before.st_size > limit or before.st_size > remaining:
            raise ValueError('file read budget exceeded')
        with path.open('rb') as stream:
            raw = stream.read(min(limit, remaining) + 1)
        after = path.stat()
    except OSError as exc:
        raise ValueError('file read failed') from exc
    if (len(raw) > limit or len(raw) > remaining or
            before.st_size != len(raw) or after.st_size != len(raw) or
            before.st_mtime_ns != after.st_mtime_ns or
            before.st_ctime_ns != after.st_ctime_ns or
            before.st_ino != after.st_ino):
        raise ValueError('file changed or read budget exceeded')
    _check_components(path)
    return raw


def _bound_file(binding, workspace, limit, remaining, label):
    _shape(binding, {'path', 'sha256'}, label)
    _hash_string(binding['sha256'], label)
    path = _local_path(binding['path'], workspace)
    raw = _read(path, limit, remaining)
    if _digest(raw) != binding['sha256']:
        raise ValueError(label + ' SHA256 mismatch')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError(label + ' must be UTF8 text') from exc
    return path, raw, text


def _absent_target(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError('target status unavailable') from exc
        if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) &
                                          _STAT_REPARSE):
            raise ValueError('symlink or reparse target forbidden')
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError('target ancestor is not directory')
    raise ValueError('deleted target still exists')


def _lines(text):
    return text.splitlines(keepends=True)


def _hunks(before, after):
    left, right = _lines(before), _lines(after)
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    for group in matcher.get_grouped_opcodes(3):
        lines = []
        left_start = group[0][1]
        right_start = group[0][3]
        left_count = group[-1][2] - left_start
        right_count = group[-1][4] - right_start
        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    lines.append({'kind': 'context', 'before_line': i + 1,
                                  'after_line': j + 1, 'text': left[i]})
            else:
                for i in range(i1, i2):
                    lines.append({'kind': 'remove', 'before_line': i + 1,
                                  'after_line': None, 'text': left[i]})
                for j in range(j1, j2):
                    lines.append({'kind': 'add', 'before_line': None,
                                  'after_line': j + 1, 'text': right[j]})
        yield {'before_start': left_start + 1 if left_count else left_start,
               'before_count': left_count,
               'after_start': right_start + 1 if right_count else right_start,
               'after_count': right_count, 'lines': lines}


def _contracts(records, workspace, paths, used):
    if type(records) is not list or len(records) > 32:
        raise ValueError('contracts count')
    seen = set()
    result = []
    for record in records:
        _shape(record, {'id', 'path', 'sha256', 'applies_to', 'category',
                        'mandatory_review', 'reviewer'}, 'contract')
        _id(record['id'], 'contract')
        if record['id'] in seen:
            raise ValueError('duplicate contract ID')
        seen.add(record['id'])
        if type(record['applies_to']) is not list or not record['applies_to']:
            raise ValueError('contract applies_to')
        applies = []
        for value in record['applies_to']:
            value = _relative(value)
            if value not in paths or value in applies:
                raise ValueError('contract applies_to must uniquely bind changes')
            applies.append(value)
        if type(record['category']) is not str or record['category'] not in _CATEGORIES:
            raise ValueError('contract category')
        if type(record['mandatory_review']) is not bool:
            raise ValueError('contract mandatory_review')
        if type(record['reviewer']) is not str or record['reviewer'] not in ('main_model', 'human'):
            raise ValueError('contract reviewer')
        # Contract has additional binding fields; validate its file separately.
        _hash_string(record['sha256'], 'contract')
        path = _local_path(record['path'], workspace)
        raw = _read(path, _MAX_CONTRACT, _MAX_TOTAL - used)
        if _digest(raw) != record['sha256']:
            raise ValueError('contract SHA256 mismatch')
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('contract must be UTF8 text') from exc
        if not text.strip() or '\x00' in text:
            raise ValueError('contract must have nonblank text')
        used += len(raw)
        result.append({'id': record['id'], 'path': record['path'],
                       'sha256': record['sha256'], 'applies_to': applies,
                       'category': record['category'],
                       'mandatory_review': record['mandatory_review'],
                       'reviewer': record['reviewer'], 'text': text,
                       'line_start': 1 if text else 0,
                       'line_end': len(_lines(text))})
    return result, used


def _validated(arguments):
    raw = _serialized(arguments, 'request', _MAX_INPUT)
    try:
        _plain_json_types(arguments)
    except RecursionError as exc:
        raise ValueError('request nesting depth') from exc
    request = json.loads(raw)
    _shape(request, {'schema_version', 'workspace', 'request_id', 'changes',
                     'contracts'}, 'request')
    _version(request['schema_version'], 'request')
    workspace = _local_path(request['workspace'], directory=True)
    _id(request['request_id'], 'request')
    records = request['changes']
    if type(records) is not list or not 1 <= len(records) <= 16:
        raise ValueError('changes count')
    paths = set()
    targets = set()
    changes = []
    used = 0
    for record in records:
        _shape(record, {'path', 'before', 'after'}, 'change')
        rel = _relative(record['path'])
        if rel in paths:
            raise ValueError('duplicate changed path')
        paths.add(rel)
        target = workspace.joinpath(*rel.split('/'))
        if target in targets:
            raise ValueError('duplicate current target')
        targets.add(target)
        if record['before'] is None and record['after'] is None:
            raise ValueError('change needs before or after')
        if record['before'] is not None:
            _, before_raw, before = _bound_file(record['before'], workspace,
                                                _MAX_SOURCE, _MAX_TOTAL - used,
                                                'before')
            used += len(before_raw)
            before_sha = record['before']['sha256']
        else:
            before_raw, before, before_sha = b'', '', None
        if record['after'] is not None:
            after_path, after_raw, after = _bound_file(record['after'], workspace,
                                                       _MAX_SOURCE, _MAX_TOTAL - used,
                                                       'after')
            if after_path != target:
                raise ValueError('after path is not current target')
            used += len(after_raw)
            after_sha = record['after']['sha256']
        else:
            _absent_target(target)
            after_raw, after, after_sha = b'', '', None
        if record['before'] is not None and record['after'] is not None and before_raw == after_raw:
            raise ValueError('unchanged bytes are not a change')
        change_sha = _canonical_sha({'path': rel, 'before_sha256': before_sha,
                                     'after_sha256': after_sha})
        changes.append({'path': rel, 'before_sha256': before_sha,
                        'after_sha256': after_sha, 'change_sha256': change_sha,
                        '_before': before, '_after': after})
    contracts, used = _contracts(request['contracts'], workspace, paths, used)
    return request, _digest(raw), changes, contracts


def _decision(reply, eligible):
    _shape(reply, {'decisions', 'route'}, 'semantic reply')
    decisions = reply['decisions']
    if type(decisions) is not dict or set(decisions) != set(eligible):
        raise ValueError('semantic decisions must exactly cover eligible hunks')
    if type(reply['route']) is not dict:
        raise ValueError('semantic route object required')
    _serialized(reply, 'semantic reply', _MAX_INPUT)
    for decision in decisions.values():
        _shape(decision, {'risk_score', 'categories'}, 'semantic decision')
        if type(decision['risk_score']) is not int or not 0 <= decision['risk_score'] <= 3:
            raise ValueError('risk_score must be integer 0..3')
        categories = decision['categories']
        _shape(categories, _SENSITIVE, 'semantic categories')
        if any(type(value) is not bool for value in categories.values()):
            raise ValueError('semantic category must be boolean')
    return decisions, reply['route']


def review_diff(arguments, *, judge_factory=None):
    """Check actual UTF-8 snapshots and optionally triage every contract-bound hunk.

    Source baselines are caller-supplied, not Git-attested. No source writes,
    tests, merge, execution grant, or model inference occur in this core.
    """
    request, request_sha, changes, contracts = _validated(arguments)
    eligible = {}
    items = []
    seen_hunks = set()
    for change in changes:
        applicable = [contract for contract in contracts
                      if change['path'] in contract['applies_to']]
        bindings = [{key: contract[key] for key in
                     ('id', 'path', 'sha256', 'category', 'mandatory_review',
                      'reviewer', 'line_start', 'line_end')}
                    for contract in applicable]
        obligations = [copy.deepcopy(binding) for binding in bindings
                       if binding['mandatory_review'] or binding['category'] in _SENSITIVE]
        hunks = []
        before_text, after_text = change.pop('_before'), change.pop('_after')
        actual_hunks = list(_hunks(before_text, after_text))
        structural_empty = not actual_hunks and (change['before_sha256'] is None or
                                                 change['after_sha256'] is None)
        if structural_empty:
            actual_hunks = [{'before_start': 0, 'before_count': 0,
                             'after_start': 0, 'after_count': 0, 'lines': []}]
        for hunk in actual_hunks:
            hunk_sha = _canonical_sha(hunk)
            hunk_id = _digest((change['change_sha256'] + ':' + hunk_sha).encode('ascii'))
            if hunk_id in seen_hunks:
                raise ValueError('duplicate hunk identity')
            seen_hunks.add(hunk_id)
            if len(seen_hunks) > 64:
                raise ValueError('hunk count exceeds budget')
            human = any(binding['reviewer'] == 'human' and
                        (binding['mandatory_review'] or binding['category'] in _SENSITIVE)
                        for binding in bindings)
            result = {'id': hunk_id, 'file_path': change['path'],
                      'change_sha256': change['change_sha256'],
                      'hunk_sha256': hunk_sha, 'hunk': hunk,
                      'contract_bindings': copy.deepcopy(bindings),
                      'obligations': copy.deepcopy(obligations),
                      'status': 'unknown', 'risk_score': None,
                      'categories': None, 'reasons': [],
                      'review_required': True, 'main_model_review_required': True,
                      'human_review_required': human}
            if structural_empty:
                result['reasons'].append('empty_file_structural_change')
                if not applicable:
                    result['reasons'].append('missing_applicable_contract')
            elif applicable:
                item = {'id': hunk_id, 'file_path': change['path'],
                        'change_sha256': change['change_sha256'],
                        'hunk_sha256': hunk_sha, 'hunk': copy.deepcopy(hunk),
                        'contracts': [{key: contract[key] for key in
                                       ('id', 'path', 'sha256', 'category',
                                        'mandatory_review', 'reviewer', 'text',
                                        'line_start', 'line_end')}
                                      for contract in applicable]}
                _serialized(item, 'semantic item', _MAX_ITEM)
                eligible[hunk_id] = result
                items.append(item)
            else:
                result['reasons'].append('missing_applicable_contract')
            hunks.append(result)
        if not hunks:
            raise ValueError('changed bytes produced no text hunk')
        change['hunks'] = hunks
    output = {'schema_version': 1, 'status': 'reviewed',
              'request_id': request['request_id'], 'request_sha256': request_sha,
              'changes': changes, 'route': {}, 'semantic_quality': 'unqualified',
              'provenance': 'caller_supplied_snapshots_not_git_attestation',
              'evidence_checked_at': datetime.now(timezone.utc).isoformat(),
              'binding_observation_scope': 'current_at_evidence_validation_only',
              'execution_authorized': False, 'action_authorized': False,
              'merge_authorized': False, 'review_complete': False}
    _serialized(output, 'output', _MAX_OUTPUT)
    if items and judge_factory is not None:
        if not callable(judge_factory):
            raise ValueError('judge_factory must be callable')
        try:
            judge = judge_factory()
            if not callable(judge):
                raise ValueError('judge factory returned noncallable')
            reply = judge(copy.deepcopy(items))
        except Exception as exc:
            raise ValueError('semantic judge failed') from exc
        decisions, route = _decision(reply, eligible)
        output['route'] = copy.deepcopy(route)
        for hunk_id, decision in decisions.items():
            result = eligible[hunk_id]
            result['status'] = 'advisory'
            result['risk_score'] = decision['risk_score']
            result['categories'] = copy.deepcopy(decision['categories'])
            result['reasons'] = ['semantic_risk_triage_only']
            if decision['risk_score'] >= 2 or any(decision['categories'].values()):
                result['human_review_required'] = True
                result['reasons'].append('semantic_human_escalation')
    else:
        for result in eligible.values():
            result['reasons'] = ['semantic_unavailable']
    _serialized(output, 'output', _MAX_OUTPUT)
    return output
