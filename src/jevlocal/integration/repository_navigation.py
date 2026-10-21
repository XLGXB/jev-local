"""Bounded, metadata-only repository navigation advice from an explicit local root.

Windows uses a held-directory-handle lease; other platforms retain path-based
stable snapshots. This never reads candidate bodies or authorizes actions.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .completion_verification import (
    _check_components, _local_path, _plain_json_types, _serialized,
    _shape, _STAT_REPARSE,
)
from .review_diff import _relative


_ID = re.compile(r'[A-Za-z0-9_.:-]{1,64}\Z')
_DENIED = re.compile(
    r'(?i)(?:\.git|\.ssh|\.aws|\.azure|\.kube|\.gnupg|\.pki|\.codex|\.claude|'
    r'\.env(?:\..*)?|\.netrc|\.npmrc|\.pypirc|credentials(?:\..*)?|'
    r'secrets(?:\..*)?|id_rsa.*|id_ed25519.*|tokens?(?:\..*)?|'
    r'cookies?(?:\..*)?|login data|web data|local state|.*\.(?:pem|key|p12|pfx))\Z'
)
_REQUEST_KEYS = {'schema_version', 'request_id', 'repository_root',
                 'current_directory', 'query', 'candidate_names', 'max_hops'}
_MAX_INPUT = 65536
_MAX_OUTPUT = 262144
_MAX_SCAN = 256
_MAX_CHOICES = 15


def _sha(value, label='snapshot'):
    return hashlib.sha256(_serialized(value, label, _MAX_OUTPUT)).hexdigest()


def _component(name):
    _relative(name)
    if len(name) > 255 or '/' in name or any(ord(char) == 127 for char in name):
        raise ValueError('noncanonical immediate child name')
    if _DENIED.fullmatch(name):
        raise ValueError('forbidden credential component')
    return name


def _windows_root(value):
    """Validate a drive-local absolute root without observing the filesystem."""
    slash = value.replace('\\', '/')
    if (not re.fullmatch(r'[A-Za-z]:/(?:[^/]+(?:/[^/]+)*)?', slash) or
            Path(value).as_posix() != slash):
        raise ValueError('noncanonical local absolute path')
    for part in slash.split('/')[1:]:
        if part:
            _component(part)
    return Path(value)


def _new_windows_lease(root):
    from .windows_repository_handles import WindowsDirectoryLease
    return WindowsDirectoryLease(root)


def _request(arguments):
    try:
        _plain_json_types(arguments)
    except RecursionError as exc:
        raise ValueError('request nesting depth') from exc
    raw = _serialized(arguments, 'request', _MAX_INPUT)
    request = json.loads(raw)
    _shape(request, _REQUEST_KEYS, 'request')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('request schema version')
    if type(request['request_id']) is not str or not _ID.fullmatch(request['request_id']):
        raise ValueError('request ID')
    if type(request['query']) is not str or not request['query'].strip() or len(request['query']) > 8192:
        raise ValueError('query text')
    if type(request['max_hops']) is not int or not 1 <= request['max_hops'] <= 4:
        raise ValueError('max_hops')
    relative = request['current_directory']
    if type(relative) is not str or len(relative) > 4096:
        raise ValueError('current_directory length')
    if relative != '.':
        _relative(relative)
        for part in relative.split('/'):
            _component(part)
    names = request['candidate_names']
    if names is not None:
        if type(names) is not list or not 1 <= len(names) <= _MAX_CHOICES:
            raise ValueError('candidate_names count')
        for name in names:
            _component(name)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError('duplicate candidate name')
    root_value = request['repository_root']
    if type(root_value) is not str:
        raise ValueError('repository root')
    if os.name == 'nt':
        root = _windows_root(root_value)
    else:
        # Apply full component policy before path lookup; aliases are unsafe.
        for part in Path(root_value).parts[1:]:
            _component(part)
        root = _local_path(root_value, directory=True)
    current = root if relative == '.' else root.joinpath(*relative.split('/'))
    try:
        current.relative_to(root)
    except ValueError as exc:
        raise ValueError('directory outside root') from exc
    if os.name != 'nt':
        _local_path(current.as_posix(), directory=True)
    return request, hashlib.sha256(raw).hexdigest(), root, current


def _identity(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'size': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def _checked_info(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) & _STAT_REPARSE):
        raise ValueError('symlink or reparse path forbidden')
    return info


def _directory_chain(root, current, lease=None):
    if lease is not None:
        return lease.chain_identities(current)
    _check_components(current)
    parts = current.relative_to(root).parts
    paths = [root]
    for part in parts:
        paths.append(paths[-1] / part)
    chain = []
    for path in paths:
        info = _checked_info(path)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('navigation directory unavailable')
        chain.append(_identity(info))
    return chain


def _candidate(root, current, name, lease=None):
    path = current / name
    relative = path.relative_to(root).as_posix()
    if len(relative) > 4096:
        raise ValueError('relative_path length')
    if lease is not None:
        metadata = lease.child_metadata(current, name)
        if metadata is None:
            return None
        kind, identity = metadata['kind'], metadata['identity']
    else:
        try:
            info = _checked_info(path)
        except (OSError, ValueError):
            return None
        if stat.S_ISDIR(info.st_mode):
            kind = 'directory'
        elif stat.S_ISREG(info.st_mode):
            kind = 'file'
        else:
            return None
        identity = _identity(info)
    source = {'root': root.as_posix(), 'current': current.relative_to(root).as_posix(),
              'name': name, 'kind': kind, 'identity': identity}
    candidate_id = 'c_' + _sha(source)[:32]
    return {'id': candidate_id, 'name': name, 'relative_path': relative,
            'kind': kind, 'identity': identity}


def _snapshot(root, current, names, lease=None):
    chain = _directory_chain(root, current, lease)
    scope = 'all_immediate' if names is None else 'explicit_subset'
    excluded = 0
    scanned = 0
    overflow = False
    candidates = []
    if names is None:
        try:
            listing = os.scandir(current) if lease is None else lease.iter_names(current)
            with listing if lease is None else nullcontext(listing) as entries:
                for entry in entries:
                    if scanned == _MAX_SCAN:
                        overflow = True
                        break
                    scanned += 1
                    try:
                        name = _component(entry.name if lease is None else entry)
                    except ValueError:
                        excluded += 1
                        continue
                    candidate = _candidate(root, current, name, lease)
                    if candidate is None:
                        excluded += 1
                    else:
                        candidates.append(candidate)
        except OSError as exc:
            raise ValueError('directory discovery failed') from exc
    else:
        for name in names:
            scanned += 1
            candidate = _candidate(root, current, name, lease)
            if candidate is None:
                raise ValueError('explicit candidate unavailable or forbidden')
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item['name'].casefold(), item['name']))
    relative = current.relative_to(root).as_posix()
    if relative == '':
        relative = '.'
    digest = _sha({'root_identity': chain[0], 'current_identity': chain[-1],
                   'current_directory': relative, 'scope': scope,
                   'candidates': candidates, 'excluded_count': excluded,
                   'scanned_count': scanned, 'scan_overflow': overflow})
    hop = {'current_directory': relative, 'scope': scope,
           'candidates': candidates, 'excluded_count': excluded,
           'scanned_count': scanned, 'snapshot_sha256': digest,
           'path_evidence': {'repository_root_identity': chain[0],
                             'current_directory_identity': chain[-1],
                             'current_directory': relative,
                             'candidate_ids': [item['id'] for item in candidates],
                             'scope': scope, 'snapshot_sha256': digest,
                             'selected_candidate_identity': None},
           'selected_candidate_id': None, 'route': {}}
    return hop, chain, overflow


def _stable(root, current, chain, candidates, lease=None):
    try:
        if _directory_chain(root, current, lease) != chain:
            raise ValueError('path evidence changed')
        for item in candidates:
            observed = _candidate(root, current, item['name'], lease)
            if observed != item:
                raise ValueError('path evidence changed')
    except (OSError, ValueError) as exc:
        raise ValueError('path evidence changed') from exc


def navigate_repository(arguments, *, judge_factory=None):
    """Observe immediate metadata and return at most four advisory hops.

    A judge is optional, created lazily once, and never receives identities or
    absolute paths. No file data is opened or read.
    """
    request, request_sha, root, current = _request(arguments)
    context = _new_windows_lease(root) if os.name == 'nt' else nullcontext(None)
    with context as lease:
        return _navigate_with_lease(request, request_sha, root, current,
                                    judge_factory, lease)


def _navigate_with_lease(request, request_sha, root, current, judge_factory, lease):
    report = {'schema_version': 1, 'status': 'unknown',
              'request_id': request['request_id'], 'request_sha256': request_sha,
              'repository_root': root.as_posix(), 'query': request['query'],
              'hops': [], 'selected_file': None, 'next_directory': None,
              'reason': 'semantic_unavailable', 'semantic_quality': 'unqualified',
              'calibrated': False, 'main_model_review_required': True,
              'execution_authorized': False, 'file_write_authorized': False,
              'deletion_authorized': False,
              'path_concurrency_guarantee': ('windows_handle_relative_no_reparse'
                                             if lease is not None else
                                             'stable_snapshot_not_handle_anchored')}
    judge = None
    for index in range(1, request['max_hops'] + 1):
        names = request['candidate_names'] if index == 1 else None
        hop, chain, overflow = _snapshot(root, current, names, lease)
        hop['hop_index'] = index
        report['hops'].append(hop)
        candidates = hop['candidates']
        if overflow or len(candidates) > _MAX_CHOICES:
            report['status'] = 'requires_candidate_subset'
            report['reason'] = ('discovery_scan_limit_exceeded' if overflow
                                else 'candidate_capacity_exceeded')
            _stable(root, current, chain, candidates, lease)
            break
        if not candidates:
            report['status'] = 'empty_directory'
            report['reason'] = 'no_eligible_candidate'
            _stable(root, current, chain, candidates, lease)
            break
        if judge_factory is None:
            _stable(root, current, chain, candidates, lease)
            break
        prepared = {'request_id': request['request_id'], 'hop_index': index,
                    'query': request['query'], 'current_directory': hop['current_directory'],
                    'candidates': [{key: item[key] for key in
                                    ('id', 'name', 'relative_path', 'kind')}
                                   for item in candidates],
                    'snapshot_sha256': hop['snapshot_sha256']}
        _serialized(prepared, 'prepared navigation', _MAX_INPUT)
        _stable(root, current, chain, candidates, lease)
        if judge is None:
            if not callable(judge_factory):
                raise ValueError('judge_factory must be callable')
            try:
                judge = judge_factory()
            except Exception as exc:
                raise ValueError('semantic judge failed') from exc
            if not callable(judge):
                raise ValueError('judge factory returned noncallable')
        try:
            reply = judge(copy.deepcopy(prepared))
        except Exception as exc:
            raise ValueError('semantic judge failed') from exc
        _stable(root, current, chain, candidates, lease)
        _shape(reply, {'selected_candidate_id', 'route'}, 'semantic reply')
        _plain_json_types(reply)
        _serialized(reply, 'semantic reply', _MAX_INPUT)
        selected_id = reply['selected_candidate_id']
        selected = next((item for item in candidates if item['id'] == selected_id), None)
        if selected_id is not None and (type(selected_id) is not str or selected is None):
            raise ValueError('semantic selection outside candidates')
        if type(reply['route']) is not dict or not reply['route']:
            raise ValueError('semantic route object required')
        hop['route'] = copy.deepcopy(reply['route'])
        if selected is None:
            report['reason'] = 'semantic_abstained'
            break
        hop['selected_candidate_id'] = selected_id
        hop['path_evidence']['selected_candidate_identity'] = copy.deepcopy(selected['identity'])
        _stable(root, current, chain, candidates, lease)
        if selected['kind'] == 'file':
            report['status'] = 'file_selected'
            report['selected_file'] = selected['relative_path']
            report['reason'] = 'semantic_closed_set_advice'
            break
        if index == request['max_hops']:
            report['status'] = 'depth_limit'
            report['next_directory'] = selected['relative_path']
            report['reason'] = 'max_hops_reached'
            break
        current = root.joinpath(*selected['relative_path'].split('/'))
    _serialized(report, 'navigation output', _MAX_OUTPUT)
    return report
