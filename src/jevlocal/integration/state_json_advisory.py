"""Explicit, source-bound, advisory-only state-JSON SemDecide adapter.

Configuration admission uses only the standard library and never imports a model.
The native consumer and encoder are imported only inside an admitted worker.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


ROUTE = 'state_json_advisory_v1'
_FINGERPRINT = 'dc8c02b92391fde1abe361f485470b2b4a036fc315148e88d0d56cd83eb3c693'
_SIDECAR_SHA = 'e61da9d631e3093aacf00f00302e11e6887194ec0af7df5043488c78bd17622f'
_SIDECAR_PATH = Path('D:/Projects/JevLocal/runtime/kev-state-json-evaluation-01/calibration.json')
_CHECKPOINT = Path('D:/Projects/JevLocal/models/experiments/kev-state-json-sft-01/epoch-2')
_SOURCES = frozenset({
    'decision_ft.state_json_native', 'decision_ft.state_json_binding',
    'decision_ft.state_json', 'decision_ft.backends', 'decision_ft.data',
    'decision_ft.train', 'decision_deploy.common', 'decision_deploy.protocol',
    'decision_deploy.runtime', 'decision_deploy.identity', 'kev.api',
    'kev.checkpoint', 'kev.model', 'kev.data',
})
_CHECKPOINT_FILES = frozenset({
    'adapter_config.json', 'adapter_model.safetensors', 'chat_template.jinja',
    'head.pt', 'local_checkpoint.json', 'README.md',
    'state_representation.json', 'tokenizer.json', 'tokenizer_config.json',
})
_CONFIG_FIELDS = frozenset({
    'schema_version', 'route', 'spec', 'expected_identity',
    'expected_fingerprint', 'expected_binding', 'expected_sources',
    'expected_checkpoint_files', 'calibration',
})
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_SPEC = {'backend': 'kev', 'mode': 'native', 'purpose': 'development_evaluation',
         'device': 'cuda', 'weights_dtype': 'fp32', 'amp': 'none'}
_TEMPERATURE = 0.41738125681877136


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('nonfinite JSON')


def _read_json(path):
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    if text.startswith('\ufeff'):
        raise ValueError('UTF-8 BOM forbidden')
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _local_path(value, *, directory=False):
    if type(value) is not str or value.startswith(('\\\\', '//')):
        raise ValueError('absolute local path required')
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('absolute local path required')
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if (stat.S_ISLNK(info.st_mode) or
                getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('reparse path forbidden')
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('regular local path required')
    return path


def _hash_map(value, keys):
    if (type(value) is not dict or set(value) != keys or
            any(type(k) is not str or type(v) is not str or not _HEX.fullmatch(v)
                for k, v in value.items())):
        raise ValueError('invalid frozen hash map')


def _module_file(name):
    # PathFinder with each package's search locations avoids executing __init__
    # (and hence avoids importing torch) while resolving actual import origins.
    location = None
    parts = name.split('.')
    for index in range(len(parts)):
        spec = importlib.machinery.PathFinder.find_spec('.'.join(parts[:index + 1]), location)
        if spec is None:
            raise ValueError('required source unavailable')
        location = spec.submodule_search_locations
    if spec.origin is None:
        raise ValueError('required source has no file origin')
    path = _local_path(spec.origin)
    loaded = sys.modules.get(name)
    if loaded is not None:
        loaded_path = _local_path(getattr(loaded, '__file__', None))
        if loaded_path.resolve(strict=True) != path.resolve(strict=True):
            raise ValueError('loaded source origin differs from import origin')
    return path


def _checkpoint_files(path):
    actual = {}
    for entry in path.iterdir():
        local = _local_path(str(entry), directory=stat.S_ISDIR(entry.lstat().st_mode))
        if local.is_dir():
            raise ValueError('checkpoint must be a complete flat inventory')
        actual[entry.name] = _sha(local)
    return actual


def _sidecar(config):
    cal = config['calibration']
    if (type(cal) is not dict or set(cal) != {'path', 'sha256'} or
            type(cal['sha256']) is not str or cal['sha256'] != _SIDECAR_SHA):
        raise ValueError('invalid calibration pin')
    path = _local_path(cal['path'])
    if path != _SIDECAR_PATH or _sha(path) != _SIDECAR_SHA:
        raise ValueError('calibration path or byte SHA changed')
    data = _read_json(path)
    identity = config['expected_identity']
    if (type(data) is not dict or data.get('schema_version') != 1 or
            data.get('stage') != 'calibration' or
            data.get('runtime_fingerprint') != _FINGERPRINT or
            _digest(data.get('runtime_identity')) != _FINGERPRINT or
            _canonical(data.get('runtime_identity')) != _canonical(identity) or
            _canonical(data.get('representation_binding')) !=
            _canonical(config['expected_binding']) or
            data.get('selected_checkpoint') != str(_CHECKPOINT).replace('/', '\\') or
            data.get('checkpoint_files_sha256') != config['expected_checkpoint_files'] or
            data.get('consumer_source_sha256') != config['expected_sources']['decision_ft.state_json_native'] or
            data.get('fits', {}).get('0', {}).get('status') != 'bound_hit' or
            data.get('fits', {}).get('1', {}).get('status') != 'fitted' or
            data.get('fits', {}).get('2', {}).get('status') != 'bound_hit' or
            data.get('temperatures', {}).get('1') != _TEMPERATURE or
            data.get('effective_temperatures') != {'0': 1.0, '1': _TEMPERATURE, '2': 1.0}):
        raise ValueError('calibration identity or policy mismatch')
    return data


def validate_config(config):
    """Inspect pinned local bytes only; return a defensive copy, never load a model."""
    if (type(config) is not dict or set(config) != _CONFIG_FIELDS or
            type(config['schema_version']) is not int or config['schema_version'] != 1 or
            config['route'] != ROUTE or type(config['expected_identity']) is not dict or
            config['expected_fingerprint'] != _FINGERPRINT):
        raise ValueError('invalid state-JSON advisory config')
    spec = config['spec']
    if (type(spec) is not dict or set(spec) != set(_SPEC) | {'checkpoint'} or
            {k: spec[k] for k in _SPEC} != _SPEC or
            type(spec['checkpoint']) is not str):
        raise ValueError('invalid explicit state-JSON native spec')
    checkpoint = _local_path(spec['checkpoint'], directory=True)
    if checkpoint != _CHECKPOINT:
        raise ValueError('checkpoint path changed')
    identity = config['expected_identity']
    if (_digest(identity) != _FINGERPRINT or
            type(identity.get('representation')) is not dict or
            type(config['expected_binding']) is not dict or
            _canonical(identity['representation'].get('binding')) !=
            _canonical(config['expected_binding']) or
            identity.get('spec') != _SPEC or
            identity.get('consumer_source_sha256') !=
            identity.get('sources_sha256', {}).get('decision_ft.state_json_native')):
        raise ValueError('runtime identity or binding mismatch')
    _hash_map(config['expected_sources'], _SOURCES)
    _hash_map(config['expected_checkpoint_files'], _CHECKPOINT_FILES)
    if (identity.get('sources_sha256') != config['expected_sources'] or
            identity.get('checkpoint_files_sha256') != config['expected_checkpoint_files']):
        raise ValueError('identity source/checkpoint closure mismatch')
    for name, expected in config['expected_sources'].items():
        if _sha(_module_file(name)) != expected:
            raise ValueError('native consumer source hash mismatch')
    if _checkpoint_files(checkpoint) != config['expected_checkpoint_files']:
        raise ValueError('checkpoint tree inventory mismatch')
    marker = _read_json(_local_path(str(checkpoint / 'state_representation.json')))
    if (_canonical(marker) != _canonical(identity['representation'].get('marker')) or
            type(marker) is not dict or set(marker) !=
            {'schema_version', 'status', 'binding', 'training_freeze_sha256', 'epoch'} or
            type(marker['schema_version']) is not int or marker['schema_version'] != 1 or
            marker['status'] != 'experimental_requires_explicit_consumer' or
            type(marker['epoch']) is not int or marker['epoch'] <= 0 or
            type(marker['training_freeze_sha256']) is not str or
            not _HEX.fullmatch(marker['training_freeze_sha256']) or
            _canonical(marker['binding']) != _canonical(config['expected_binding'])):
        raise ValueError('checkpoint representation marker mismatch')
    _sidecar(config)
    return copy.deepcopy(config)


class StateJsonAdvisoryPredictor:
    """Own one native runtime and one immutable calibration snapshot."""

    def __init__(self, runtime, calibration, config):
        self.runtime = runtime
        self.calibration = copy.deepcopy(calibration)
        self._config = copy.deepcopy(config)
        self._config_digest = _digest(config)
        self._calibration_bytes = _canonical(calibration)

    def predict(self, request):
        from decision_ft.state_json_binding import prepare_inference_atoms

        native = self.runtime.native
        try:
            pairs = prepare_inference_atoms(request, expected_binding=self._config['expected_binding'])
            if list(request['questions']) != [qid for qid, _ in pairs]:
                raise ValueError('question order changed during representation')
            for qid, atom in pairs:
                encoded = native.encode(atom)
                if (type(encoded) is not dict or encoded.get('state_truncated') is not False or
                        type(encoded.get('ids')) is not list or
                        not 1 <= len(encoded['ids']) <= 512 or
                        atom.keys != _option_keys(request['questions'][qid])):
                    raise ValueError('native encoding truncated, over budget, or reordered')
            return self.runtime.predict(request, calibration=self.calibration)
        finally:
            native.cache.clear()


def _option_keys(question):
    kind = question['type']
    if kind == 'choice':
        return list(question['criteria'])
    if kind == 'score':
        return [str(i) for i in range(len(question['criteria']))]
    return ['false', 'true']


def build_predictor(config):
    snapshot = validate_config(config)
    for flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                 'HF_HUB_DISABLE_TELEMETRY', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'):
        if os.environ.get(flag) != '1':
            raise ValueError('process-start offline/safe-loading flags required')
    for module, flag in (('huggingface_hub.constants', 'HF_HUB_OFFLINE'),
                         ('transformers.utils.hub', '_is_offline_mode')):
        loaded = sys.modules.get(module)
        if loaded is not None and hasattr(loaded, flag) and getattr(loaded, flag) is not True:
            raise ValueError('native dependencies imported without offline mode')
    calibration = _sidecar(snapshot)
    from decision_ft.state_json_native import StateJsonNativeRuntime
    try:
        runtime = StateJsonNativeRuntime(
            snapshot['spec'], expected_binding=snapshot['expected_binding'],
            expected_checkpoint_files=snapshot['expected_checkpoint_files'],
            expected_sources=snapshot['expected_sources'])
        predictor = StateJsonAdvisoryPredictor(runtime, calibration, snapshot)
        identity(predictor, snapshot)
        return predictor
    except Exception:
        raise ValueError('state-JSON native initialization failed; no fallback used') from None


def identity(predictor, config):
    """Check in-memory route and calibration drift; no per-row disk hash."""
    if (type(predictor) is not StateJsonAdvisoryPredictor or
            _digest(config) != predictor._config_digest or
            _digest(predictor._config) != predictor._config_digest or
            _canonical(predictor.calibration) != predictor._calibration_bytes):
        raise ValueError('state-JSON advisory config/calibration drift')
    runtime = predictor.runtime
    if (runtime.spec != config['spec'] or
            _canonical(runtime.identity) != _canonical(config['expected_identity']) or
            _digest(runtime.identity) != _FINGERPRINT or
            runtime.fingerprint != _FINGERPRINT):
        raise ValueError('state-JSON native runtime identity drift')
    return _FINGERPRINT


def validate_prediction(envelope, prediction, config):
    from decision_deploy.protocol import validate_reply

    if (type(prediction) is not dict or set(prediction) !=
            {'runtime_fingerprint', 'answers', 'inference_total_ms'} or
            prediction['runtime_fingerprint'] != config['expected_fingerprint'] or
            type(prediction['answers']) is not dict or
            list(prediction['answers']) != list(envelope['questions'])):
        raise ValueError('state-JSON prediction identity or shape mismatch')
    for qid, question in envelope['questions'].items():
        answer = prediction['answers'][qid]
        fitted = question['type'] == 'score'
        temperature = _TEMPERATURE if fitted else 1.0
        if (type(answer) is not dict or set(answer) !=
                {'selected_key', 'probabilities', 'p_selected', 'calibrated', 'temperature'} or
                answer['calibrated'] is not fitted or
                type(answer['probabilities']) is not dict or
                list(answer['probabilities']) != _option_keys(question) or
                type(answer['temperature']) not in (int, float) or
                not math.isfinite(answer['temperature']) or
                answer['temperature'] != temperature):
            raise ValueError('state-JSON per-kind calibration policy mismatch')
    reply = {'request_id': envelope['request_id'],
             'observation_id': envelope['observation_id'], **prediction}
    validate_reply(envelope, reply)
    return reply


def metadata(config):
    return {'schema_version': 1, 'kind': ROUTE,
            'runtime_fingerprint': config['expected_fingerprint'],
            'calibration_sha256': config['calibration']['sha256'],
            'advisory_only': True, 'execution_authorized': False,
            'qualification_status': 'not_qualified'}
