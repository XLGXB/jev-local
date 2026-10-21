"""F05 caller-supplied completion evidence checks; no command or model execution here."""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import re
import stat
from datetime import datetime, timezone


_ID = re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_UTC = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)\Z')
_KINDS = {'test_run', 'tool_result', 'diff'}
_CLAIM_KINDS = {'tests_passed', 'change_present', 'completion'}
_LABELS = {'supported', 'contradicted', 'unknown'}
_STAT_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_MAX_INPUT = 128 * 1024
_MAX_RECEIPT = 64 * 1024
_MAX_RECEIPTS = 256 * 1024
_MAX_SUBJECT = 16 * 1024 * 1024
_MAX_SUBJECTS = 64 * 1024 * 1024


def _serialized(value, label, limit):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':'), sort_keys=True).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError(label + ' must be finite JSON') from exc
    if len(raw) > limit:
        raise ValueError(label + ' exceeds byte budget')
    return raw


def _shape(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ' shape')


def _plain_json_types(value):
    """Reject Python object hooks before taking a canonical JSON snapshot."""
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError('JSON object key')
            _plain_json_types(child)
    elif type(value) is list:
        for child in value:
            _plain_json_types(child)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('plain JSON types required')


def _version(value, label):
    if type(value) is not int or value != 1:
        raise ValueError(label + ' schema version')


def _id(value, label):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError(label + ' ID')


def _hash_string(value, label):
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(label + ' SHA256')


def _text(value, label, *, nonempty=True, limit=8192):
    if (type(value) is not str or len(value) > limit or
            (nonempty and not value.strip())):
        raise ValueError(label + ' text')


def _utc(value, label, now):
    if type(value) is not str or not _UTC.fullmatch(value):
        raise ValueError(label + ' UTC time')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(label + ' UTC time') from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(label + ' must be UTC')
    if parsed > now:
        raise ValueError(label + ' is future')
    return parsed


def _check_components(path):
    """Reject symlink/reparse ancestors, not just a final linked file."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError('path unavailable') from exc
        if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) &
                                         _STAT_REPARSE):
            raise ValueError('symlink or reparse path forbidden')


def _local_path(value, workspace=None, *, directory=False):
    if type(value) is not str or not value or '\x00' in value:
        raise ValueError('local path required')
    path = Path(value)
    slash = value.replace('\\', '/')
    if (not path.is_absolute() or value.startswith(('\\\\', '//', '\\\\?', '\\\\.')) or
            any(part in ('.', '..') for part in slash.split('/')) or
            ':' in value[2:] or path.as_posix() != slash):
        raise ValueError('noncanonical local absolute path')
    if not directory:
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError('file outside workspace') from exc
    _check_components(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError('path unavailable') from exc
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('workspace is not directory')
    else:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('evidence/subject is not regular file')
    return path


def _file_sha(path, max_bytes, budget=None):
    size = 0
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open('rb') as stream:
            while chunk := stream.read(65536):
                size += len(chunk)
                if size > max_bytes or (budget is not None and size > budget):
                    raise ValueError('file read budget exceeded')
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ValueError('file read failed') from exc
    if (before.st_size != size or after.st_size != size or
            before.st_mtime_ns != after.st_mtime_ns or
            before.st_ctime_ns != after.st_ctime_ns or
            before.st_ino != after.st_ino):
        raise ValueError('file changed during verification')
    return digest.hexdigest(), size


def _json_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant ' + value)


def _receipt_json(path, expected_sha, remaining):
    actual, size = _file_sha(path, _MAX_RECEIPT, remaining)
    if actual != expected_sha:
        raise ValueError('evidence SHA256 mismatch')
    try:
        with path.open('rb') as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != actual:
            raise ValueError('evidence changed during parse')
        receipt = json.loads(raw.decode('utf-8'), object_pairs_hook=_json_pairs,
                             parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('strict UTF8 JSON receipt required') from exc
    return receipt, size


def _subjects(value, workspace, subject_budget):
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ValueError('subject_files count')
    seen = set()
    for binding in value:
        _shape(binding, {'path', 'sha256'}, 'subject')
        _hash_string(binding['sha256'], 'subject')
        path = _local_path(binding['path'], workspace)
        if path in seen:
            raise ValueError('duplicate subject path')
        seen.add(path)
        actual, size = _file_sha(path, _MAX_SUBJECT,
                                 _MAX_SUBJECTS - subject_budget)
        subject_budget += size
        if actual != binding['sha256']:
            raise ValueError('current subject SHA256 mismatch')
    return subject_budget


def _command(value):
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ValueError('command argv')
    for arg in value:
        _text(arg, 'command argument', limit=8192)


def _receipt(value, kind, run_id, window, now, workspace, subject_budget):
    common = {'schema_version', 'run_id', 'subject_files'}
    if kind == 'diff':
        keys = common | {'observed_at', 'diff'}
    else:
        keys = common | {'started_at', 'ended_at', 'status', 'command',
                         'exit_code', 'output'}
        if kind == 'test_run':
            keys |= {'tests'}
    _shape(value, keys, 'receipt')
    _version(value['schema_version'], 'receipt')
    _id(value['run_id'], 'receipt run')
    run_matches = value['run_id'] == run_id
    subject_budget = _subjects(value['subject_files'], workspace, subject_budget)
    if kind == 'diff':
        observed = _utc(value['observed_at'], 'observed_at', now)
        _text(value['diff'], 'diff', limit=_MAX_RECEIPT)
        return ('stale' if not run_matches or observed < window else 'current'), subject_budget
    started = _utc(value['started_at'], 'receipt started_at', now)
    ended = _utc(value['ended_at'], 'receipt ended_at', now)
    if started > ended:
        raise ValueError('receipt time order')
    if value['status'] not in ('executed', 'failed', 'skipped', 'not_run') or type(value['status']) is not str:
        raise ValueError('receipt status')
    _command(value['command'])
    _text(value['output'], 'output', nonempty=False, limit=_MAX_RECEIPT)
    executed = value['status'] in ('executed', 'failed')
    if executed:
        if type(value['exit_code']) is not int:
            raise ValueError('executed exit_code integer')
    elif value['exit_code'] is not None:
        raise ValueError('unexecuted exit_code must be null')
    if kind == 'test_run':
        _shape(value['tests'], {'passed', 'failed', 'errors', 'skipped'}, 'test counts')
        for count in value['tests'].values():
            if type(count) is not int or count < 0:
                raise ValueError('test count nonnegative integer')
    if not run_matches or started < window or ended < window:
        state = 'stale'
    elif value['status'] in ('skipped', 'not_run'):
        state = 'unrun'
    elif kind == 'test_run' and value['tests']['skipped']:
        state = 'skipped_tests'
    elif (value['status'] == 'failed' or value['exit_code'] != 0 or
          (kind == 'test_run' and
           (value['tests']['failed'] or value['tests']['errors']))):
        state = 'failed'
    elif kind == 'test_run' and value['tests']['passed'] == 0:
        state = 'zero_tests'
    else:
        state = 'current'
    return state, subject_budget


def _validated(arguments):
    request_raw = _serialized(arguments, 'request', _MAX_INPUT)
    try:
        _plain_json_types(arguments)
    except RecursionError as exc:
        raise ValueError('request nesting depth') from exc
    arguments = json.loads(request_raw)
    _shape(arguments, {'schema_version', 'workspace', 'run_id', 'started_at',
                       'claims', 'evidence'}, 'request')
    _version(arguments['schema_version'], 'request')
    workspace = _local_path(arguments['workspace'], directory=True)
    _id(arguments['run_id'], 'run')
    now = datetime.now(timezone.utc)
    window = _utc(arguments['started_at'], 'request started_at', now)
    claims = arguments['claims']
    if type(claims) is not list or not 1 <= len(claims) <= 32:
        raise ValueError('claims count')
    seen = set()
    safe_claims = []
    for claim in claims:
        _shape(claim, {'id', 'text', 'kind', 'evidence_ids'}, 'claim')
        _id(claim['id'], 'claim')
        if claim['id'] in seen:
            raise ValueError('duplicate claim ID')
        seen.add(claim['id'])
        _text(claim['text'], 'claim', limit=8192)
        if type(claim['kind']) is not str or claim['kind'] not in _CLAIM_KINDS:
            raise ValueError('claim kind')
        ids = claim['evidence_ids']
        if type(ids) is not list or not 1 <= len(ids) <= 16:
            raise ValueError('claim evidence_ids count')
        cited = set()
        for item_id in ids:
            _id(item_id, 'cited evidence')
            if item_id in cited:
                raise ValueError('duplicate cited evidence ID')
            cited.add(item_id)
        safe_claims.append({'id': claim['id'], 'text': claim['text'],
                            'kind': claim['kind'], 'evidence_ids': ids[:]})
    items = arguments['evidence']
    if type(items) is not list or len(items) > 64:
        raise ValueError('evidence count')
    evidence = {}
    paths = set()
    receipt_budget = 0
    subject_budget = 0
    for item in items:
        _shape(item, {'id', 'kind', 'path', 'sha256'}, 'evidence')
        _id(item['id'], 'evidence')
        if item['id'] in evidence:
            raise ValueError('duplicate evidence ID')
        if type(item['kind']) is not str or item['kind'] not in _KINDS:
            raise ValueError('evidence kind')
        _hash_string(item['sha256'], 'evidence')
        path = _local_path(item['path'], workspace)
        if path in paths:
            raise ValueError('duplicate evidence path')
        paths.add(path)
        receipt, size = _receipt_json(path, item['sha256'],
                                      _MAX_RECEIPTS - receipt_budget)
        receipt_budget += size
        state, subject_budget = _receipt(receipt, item['kind'],
                                         arguments['run_id'], window, now,
                                         workspace, subject_budget)
        evidence[item['id']] = {'id': item['id'], 'kind': item['kind'],
                                'artifact_path': item['path'], 'sha256': item['sha256'],
                                'receipt': receipt, 'state': state}
    return (safe_claims, evidence, datetime.now(timezone.utc).isoformat(),
            arguments['run_id'], hashlib.sha256(request_raw).hexdigest())


def _claim_guard(claim, evidence):
    refs = claim['evidence_ids']
    found = [evidence.get(item_id) for item_id in refs]
    bindings = []
    for item_id, item in zip(refs, found):
        if item is None:
            bindings.append({'id': item_id, 'state': 'missing'})
            continue
        receipt = item['receipt']
        binding = {'id': item_id, 'kind': item['kind'],
                   'artifact_path': item['artifact_path'], 'sha256': item['sha256'],
                   'state': item['state'], 'run_id': receipt['run_id'],
                   'subject_files': copy.deepcopy(receipt['subject_files'])}
        if item['kind'] == 'diff':
            binding['observed_at'] = receipt['observed_at']
        else:
            binding.update({key: copy.deepcopy(receipt[key]) for key in
                            ('started_at', 'ended_at', 'status', 'command', 'exit_code')})
            if item['kind'] == 'test_run':
                binding['tests'] = copy.deepcopy(receipt['tests'])
        bindings.append(binding)
    present = [item for item in found if item is not None]
    kinds = {item['kind'] for item in present}
    required = {'test_run'} if claim['kind'] == 'tests_passed' else (
        {'diff'} if claim['kind'] == 'change_present' else {'test_run', 'diff'})
    reasons = []
    if None in found:
        reasons.append('missing_reference')
    if not required <= kinds:
        reasons.append('required_evidence_kind_missing')
    for state, reason in (('stale', 'stale_evidence'), ('unrun', 'unrun_evidence'),
                          ('skipped_tests', 'skipped_tests'),
                          ('zero_tests', 'zero_tests')):
        if any(item['state'] == state for item in present):
            reasons.append(reason)
    if reasons:
        return 'unknown', reasons, bindings
    if any(item['state'] == 'failed' for item in present):
        return 'contradicted', ['failed_execution'], bindings
    return 'eligible', [], bindings


def verify_completion(arguments, *, judge_factory=None):
    """Verify bounded local receipts before optional lazy semantic judgement.

    A JSON receipt is an observed caller assertion, not execution attestation.
    This function never runs its command field or authorizes an action.
    """
    claims, evidence, evidence_checked_at, run_id, request_sha256 = _validated(arguments)
    results = []
    eligible = []
    for claim in claims:
        status, reasons, bindings = _claim_guard(claim, evidence)
        results.append({'id': claim['id'], 'status': status,
                        'reasons': reasons, 'evidence_bindings': bindings})
        if status == 'eligible':
            eligible.append({'id': claim['id'], 'text': claim['text'],
                             'kind': claim['kind'],
                             'evidence': [{key: item[key] for key in
                                           ('id', 'kind', 'artifact_path', 'sha256', 'receipt')}
                                          for item_id in claim['evidence_ids']
                                          for item in (evidence[item_id],)]})
    route = {}
    if eligible and judge_factory is not None:
        if not callable(judge_factory):
            raise ValueError('judge_factory must be callable')
        try:
            judge = judge_factory()
            if not callable(judge):
                raise ValueError('judge factory returned noncallable')
            reply = judge(copy.deepcopy(eligible))
        except Exception as exc:
            raise ValueError('semantic judge failed') from exc
        _shape(reply, {'decisions', 'route'}, 'semantic reply')
        decisions = reply['decisions']
        if (type(decisions) is not dict or set(decisions) !=
                {item['id'] for item in eligible} or
                any(type(value) is not str or value not in _LABELS
                    for value in decisions.values())):
            raise ValueError('semantic decisions must exactly cover eligible claims')
        if type(reply['route']) is not dict:
            raise ValueError('semantic route object required')
        _serialized(reply, 'semantic reply', _MAX_INPUT)
        route = reply['route']
        for result in results:
            if result['status'] == 'eligible':
                result['status'] = decisions[result['id']]
                result['reasons'] = ['semantic_' + decisions[result['id']]]
    else:
        for result in results:
            if result['status'] == 'eligible':
                result['status'] = 'unknown'
                result['reasons'] = ['semantic_unavailable']
    output = {'schema_version': 1, 'status': 'verified',
              'run_id': run_id, 'request_sha256': request_sha256,
              'claims': results, 'route': route,
              'semantic_quality': 'unqualified',
              'provenance': 'caller_supplied_receipts_not_execution_attestation',
              'evidence_checked_at': evidence_checked_at,
              'binding_observation_scope': 'current_at_evidence_validation_only',
              'execution_authorized': False, 'action_authorized': False}
    _serialized(output, 'output', _MAX_INPUT)
    return output
