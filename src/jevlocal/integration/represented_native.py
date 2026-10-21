"""Explicit, source-bound, uncalibrated StateJson native consumer bridge.

Admission is standard-library only. The native model is imported and loaded only
by an explicit build after all local provenance and offline checks pass.
"""
from __future__ import annotations

import copy
import importlib
import math
import os
from pathlib import Path
import re
import sys
import threading

from . import state_json_advisory as _pure


ROUTE = 'state_json_uncalibrated_v1'
_FIELDS = frozenset({
    'schema_version', 'route', 'spec', 'expected_identity',
    'expected_fingerprint', 'expected_binding', 'expected_sources',
    'expected_checkpoint_files', 'expected_adapter_sources',
    'expected_base_cache', 'calibration',
})
_ADAPTER_SOURCES = frozenset({
    'jevlocal.integration.represented_native',
    'jevlocal.integration.state_json_advisory',
})
_SPEC = {'backend': 'kev', 'mode': 'native',
         'purpose': 'development_evaluation', 'weights_dtype': 'fp32',
         'amp': 'none'}
_REVISION = re.compile(r'[0-9a-f]{40}\Z')
_REPO_PART = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
_SHARD = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z')
_BASE_REQUIRED = frozenset({'config.json', 'tokenizer.json',
                            'tokenizer_config.json'})
_TOKENIZER_OPTIONAL = frozenset({'merges.txt', 'vocab.json'})
_LEGACY_CACHE_ENV = ('HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
                     'PYTORCH_TRANSFORMERS_CACHE',
                     'PYTORCH_PRETRAINED_BERT_CACHE')


def _same_local_path(value, expected, *, directory=True):
    try:
        return (_pure._local_path(value, directory=directory).resolve(strict=True) ==
                expected.resolve(strict=True))
    except (ValueError, OSError, TypeError):
        return False


def _base_snapshot(pin):
    if (type(pin) is not dict or
            set(pin) != {'base', 'revision', 'hf_home', 'hub_cache', 'files_sha256'} or
            type(pin['base']) is not str or type(pin['revision']) is not str or
            not _REVISION.fullmatch(pin['revision'])):
        raise ValueError('invalid explicit base cache pin')
    parts = pin['base'].split('/')
    if (len(parts) != 2 or any(not _REPO_PART.fullmatch(part) or
                              part in ('.', '..') or '..' in part for part in parts)):
        raise ValueError('invalid explicit base repository')
    home = _pure._local_path(pin['hf_home'], directory=True)
    hub = _pure._local_path(pin['hub_cache'], directory=True)
    snapshot = _pure._local_path(str(hub / ('models--' + '--'.join(parts)) /
                                     'snapshots' / pin['revision']), directory=True)
    return home, hub, snapshot


def _check_base_cache(pin):
    _, _, snapshot = _base_snapshot(pin)
    files = pin['files_sha256']
    if (type(files) is not dict or not _BASE_REQUIRED <= set(files) or
            any(type(k) is not str or type(v) is not str or
                not _pure._HEX.fullmatch(v) for k, v in files.items())):
        raise ValueError('invalid base snapshot hash inventory')
    weights = set(files) - _BASE_REQUIRED - _TOKENIZER_OPTIONAL
    indexed = 'model.safetensors.index.json' in weights
    if indexed:
        weights.remove('model.safetensors.index.json')
        if (not weights or 'model.safetensors' in weights or
                any(not _SHARD.fullmatch(name) or '..' in name for name in weights)):
            raise ValueError('invalid base snapshot weight inventory')
    elif weights != {'model.safetensors'}:
        raise ValueError('invalid base snapshot weight inventory')
    # Compare the complete flat listing before reading even one file.
    if {entry.name for entry in snapshot.iterdir()} != set(files):
        raise ValueError('base snapshot flat inventory mismatch')
    for name, digest in files.items():
        local = _pure._local_path(str(snapshot / name))
        if _pure._sha(local) != digest:
            raise ValueError('base snapshot file hash mismatch')
        if name.endswith('.json'):
            if local.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('base snapshot JSON byte budget exceeded')
            if type(_pure._read_json(local)) is not dict:
                raise ValueError('base snapshot JSON object required')
    if indexed:
        index = _pure._read_json(snapshot / 'model.safetensors.index.json')
        if (type(index) is not dict or set(index) != {'metadata', 'weight_map'} or
                type(index['metadata']) is not dict or
                type(index['weight_map']) is not dict or
                not index['weight_map'] or
                any(type(key) is not str or not key or type(name) is not str or
                    not _SHARD.fullmatch(name) or '..' in name
                    for key, name in index['weight_map'].items()) or
                set(index['weight_map'].values()) != weights):
            raise ValueError('base weight index shard mismatch')


def _check_process_cache(pin):
    home, hub, _ = _base_snapshot(pin)
    for flag in ('HF_HOME', 'HF_HUB_CACHE'):
        expected = home if flag == 'HF_HOME' else hub
        value = os.environ.get(flag)
        if not value or not _same_local_path(value, expected):
            raise ValueError('process-start base cache environment mismatch')
    for flag in _LEGACY_CACHE_ENV:
        value = os.environ.get(flag)
        if value is not None and not _same_local_path(value, hub):
            raise ValueError('legacy cache override mismatch')
    try:
        hf = importlib.import_module('huggingface_hub.constants')
        transformers = importlib.import_module('transformers.utils.hub')
    except Exception:
        raise ValueError('base cache dependency constants unavailable') from None
    offline_function = getattr(transformers, 'is_offline_mode', None)
    cached_offline = getattr(transformers, '_is_offline_mode', None)
    try:
        function_offline = offline_function() if offline_function is not None else None
    except Exception:
        raise ValueError('native dependency offline state unavailable') from None
    if (getattr(hf, 'HF_HUB_OFFLINE', None) is not True or
            (cached_offline is not None and cached_offline is not True) or
            (offline_function is None and cached_offline is not True) or
            (offline_function is not None and function_offline is not True)):
        raise ValueError('native dependencies imported without offline mode')
    if (not _same_local_path(getattr(hf, 'HF_HOME', None), home) or
            not _same_local_path(getattr(hf, 'HF_HUB_CACHE', None), hub)):
        raise ValueError('loaded Hugging Face cache constants mismatch')
    effective = getattr(transformers, 'constants', None)
    if (effective is None or
            not _same_local_path(getattr(effective, 'HF_HOME', None), home) or
            not _same_local_path(getattr(effective, 'HF_HUB_CACHE', None), hub)):
        raise ValueError('loaded Transformers cache constants mismatch')
    for name in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE',
                 'PYTORCH_PRETRAINED_BERT_CACHE'):
        value = getattr(transformers, name, None)
        if value is not None and not _same_local_path(value, hub):
            raise ValueError('loaded Transformers legacy cache mismatch')


def _read_base_metadata(head):
    try:
        import torch
        metadata = torch.load(head, map_location='cpu', weights_only=True)
    except Exception:
        raise ValueError('base metadata safe load failed') from None
    if type(metadata) is not dict:
        raise ValueError('invalid base metadata')
    return metadata


def _current_binding(sources):
    """Mirror the pinned source policy without importing decision_ft (or torch)."""
    return {
        'schema_version': 1,
        'policy': {
            'schema_version': 1, 'id': 'jevlocal.state-json.v1',
            'serializer_sha256': sources['decision_ft.state_json'],
            'format': {
                'encoding': 'UTF-8 strict, no normalization',
                'scope': 'state_only',
                'json_dumps': {
                    'ensure_ascii': False, 'allow_nan': False,
                    'separators': [',', ':'], 'sort_keys': False,
                    'types': 'exact built-in JSON types',
                },
            },
        },
        'binding_sha256': sources['decision_ft.state_json_binding'],
    }


def validate_config(config):
    """Admit only the complete current local freeze; return a defensive copy."""
    if (type(config) is not dict or set(config) != _FIELDS or
            type(config['schema_version']) is not int or config['schema_version'] != 1 or
            config['route'] != ROUTE or config['calibration'] is not None):
        raise ValueError('invalid uncalibrated StateJson route config')
    spec = config['spec']
    if (type(spec) is not dict or set(spec) != set(_SPEC) | {'device', 'checkpoint'} or
            any(type(spec[k]) is not str or spec[k] != value for k, value in _SPEC.items()) or
            type(spec['device']) is not str or spec['device'] not in ('cpu', 'cuda') or
            type(spec['checkpoint']) is not str):
        raise ValueError('invalid explicit StateJson native spec')
    checkpoint = _pure._local_path(spec['checkpoint'], directory=True)
    identity = config['expected_identity']
    binding = config['expected_binding']
    fingerprint = config['expected_fingerprint']
    if (type(identity) is not dict or
            set(identity) != {'representation', 'consumer_source_sha256',
                              'checkpoint_files_sha256', 'sources_sha256',
                              'spec', 'environment'} or
            type(identity['representation']) is not dict or
            set(identity['representation']) != {'binding', 'marker'} or
            type(identity['environment']) is not dict or
            type(fingerprint) is not str or not _pure._HEX.fullmatch(fingerprint) or
            _pure._digest(identity) != fingerprint):
        raise ValueError('native runtime identity or fingerprint mismatch')
    _pure._hash_map(config['expected_sources'], _pure._SOURCES)
    _pure._hash_map(config['expected_checkpoint_files'], _pure._CHECKPOINT_FILES)
    _pure._hash_map(config['expected_adapter_sources'], _ADAPTER_SOURCES)
    sources = config['expected_sources']
    if (type(binding) is not dict or
            _pure._canonical(binding) != _pure._canonical(_current_binding(sources)) or
            _pure._canonical(identity['representation']['binding']) != _pure._canonical(binding) or
            identity['consumer_source_sha256'] != sources['decision_ft.state_json_native'] or
            identity['sources_sha256'] != sources or
            identity['checkpoint_files_sha256'] != config['expected_checkpoint_files'] or
            identity['spec'] != {key: spec[key] for key in spec if key != 'checkpoint'}):
        raise ValueError('native identity field mirror or representation binding mismatch')
    for name, expected in (*sources.items(), *config['expected_adapter_sources'].items()):
        if _pure._sha(_pure._module_file(name)) != expected:
            raise ValueError('native or adapter source hash mismatch')
    # Reject unlisted entries before opening or hashing their contents.
    names = set()
    for entry in checkpoint.iterdir():
        if entry.name not in _pure._CHECKPOINT_FILES or entry.name in names:
            raise ValueError('checkpoint flat inventory mismatch')
        names.add(entry.name)
    if names != _pure._CHECKPOINT_FILES:
        raise ValueError('checkpoint flat inventory mismatch')
    if _pure._checkpoint_files(checkpoint) != config['expected_checkpoint_files']:
        raise ValueError('checkpoint flat inventory mismatch')
    _check_base_cache(config['expected_base_cache'])
    marker_path = _pure._local_path(str(checkpoint / 'state_representation.json'))
    if marker_path.stat().st_size > 64 * 1024:
        raise ValueError('representation marker byte budget exceeded')
    marker = _pure._read_json(marker_path)
    if (type(marker) is not dict or
            set(marker) != {'schema_version', 'status', 'binding',
                            'training_freeze_sha256', 'epoch'} or
            type(marker['schema_version']) is not int or marker['schema_version'] != 1 or
            marker['status'] != 'experimental_requires_explicit_consumer' or
            type(marker['training_freeze_sha256']) is not str or
            not _pure._HEX.fullmatch(marker['training_freeze_sha256']) or
            type(marker['epoch']) is not int or marker['epoch'] <= 0 or
            _pure._canonical(marker['binding']) != _pure._canonical(binding) or
            _pure._canonical(marker) != _pure._canonical(identity['representation']['marker'])):
        raise ValueError('checkpoint representation marker mismatch')
    return copy.deepcopy(config)


class _RuntimeView:
    """Expose only route identity, never a mutable native runtime handle."""

    __slots__ = ('__runtime',)

    def __init__(self, runtime):
        object.__setattr__(self, '_RuntimeView__runtime', runtime)

    @property
    def mode(self):
        return self.__runtime.spec['mode']

    @property
    def backend(self):
        return self.__runtime.spec['backend']

    @property
    def fingerprint(self):
        return self.__runtime.fingerprint

    def __setattr__(self, _name, _value):
        raise AttributeError('runtime view is read-only')


class StateJsonRepresentedPredictor:
    """Own one native runtime and serialize complete preflight/predict cycles."""

    def __init__(self, runtime, config):
        self._runtime = runtime
        self.runtime = _RuntimeView(runtime)
        self.calibration = None
        self._config = copy.deepcopy(config)
        self._config_digest = _pure._digest(config)
        self._lock = threading.Lock()

    @property
    def metadata(self):
        config = self._config
        return copy.deepcopy({
            'schema_version': 1, 'kind': ROUTE, 'route': ROUTE,
            'config_sha256': self._config_digest,
            'runtime_fingerprint': config['expected_fingerprint'],
            'spec': config['spec'], 'binding': config['expected_binding'],
            'checkpoint_files_sha256': config['expected_checkpoint_files'],
            'expected_base_cache': config['expected_base_cache'],
            'sources_sha256': config['expected_sources'],
            'adapter_source_sha256': config['expected_adapter_sources'],
            'calibration': None, 'calibrated': False, 'advisory_only': True,
            'execution_authorized': False, 'semantic_quality': 'unqualified',
        })

    def predict(self, request):
        with self._lock:
            native = self._runtime.native
            try:
                identity(self, self._config)
                from decision_ft.state_json_binding import prepare_inference_atoms

                pairs = prepare_inference_atoms(
                    request, expected_binding=self._config['expected_binding'])
                if [qid for qid, _ in pairs] != list(request['questions']):
                    raise ValueError('question order changed during representation')
                for qid, atom in pairs:
                    encoded = native.encode(atom)
                    if (type(encoded) is not dict or
                            encoded.get('state_truncated') is not False or
                            type(encoded.get('ids')) is not list or
                            not 1 <= len(encoded['ids']) <= 512 or
                            any(type(token) is not int or token < 0
                                for token in encoded['ids']) or
                            atom.keys != _pure._option_keys(request['questions'][qid])):
                        raise ValueError('native representation truncated, over budget, or reordered')
                identity(self, self._config)
                result = self._runtime.predict(request, calibration=None)
                identity(self, self._config)
                _validate_prediction(request, result, self._config)
                return result
            except (ValueError, TypeError, KeyError, OSError):
                raise
            except Exception:
                raise ValueError('StateJson native prediction failed; no fallback used') from None
            finally:
                native.cache.clear()


def identity(predictor, config):
    """Check live route identity and mutable configuration on every call."""
    if (type(predictor) is not StateJsonRepresentedPredictor or
            predictor.calibration is not None or
            _pure._digest(config) != predictor._config_digest or
            _pure._digest(predictor._config) != predictor._config_digest):
        raise ValueError('StateJson route config or calibration drift')
    runtime = predictor._runtime
    expected = predictor._config
    if (runtime.spec != expected['spec'] or
            _pure._canonical(runtime.identity) != _pure._canonical(expected['expected_identity']) or
            _pure._digest(runtime.identity) != expected['expected_fingerprint'] or
            runtime.fingerprint != expected['expected_fingerprint'] or
            predictor.runtime.mode != 'native' or predictor.runtime.backend != 'kev'):
        raise ValueError('StateJson native runtime identity drift')
    return expected['expected_fingerprint']


def _validate_prediction(request, prediction, config):
    from decision_deploy.protocol import validate_reply

    if (type(prediction) is not dict or
            set(prediction) != {'runtime_fingerprint', 'answers', 'inference_total_ms'} or
            prediction['runtime_fingerprint'] != config['expected_fingerprint'] or
            type(prediction['answers']) is not dict or
            list(prediction['answers']) != list(request['questions'])):
        raise ValueError('StateJson raw prediction identity or question order mismatch')
    for qid, question in request['questions'].items():
        answer = prediction['answers'][qid]
        if (type(answer) is not dict or
                set(answer) != {'selected_key', 'probabilities', 'p_selected',
                                'calibrated', 'temperature'} or
                answer['calibrated'] is not False or
                type(answer['temperature']) not in (int, float) or
                not math.isfinite(answer['temperature']) or
                answer['temperature'] != 1.0 or
                type(answer['probabilities']) is not dict or
                list(answer['probabilities']) != _pure._option_keys(question)):
            raise ValueError('StateJson raw prediction calibration or candidate order mismatch')
    envelope = {'request_id': 'represented-native', 'observation_id': 'local',
                'state': request['state'], 'questions': request['questions']}
    validate_reply(envelope, {**prediction, 'request_id': envelope['request_id'],
                              'observation_id': envelope['observation_id']})


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
    _check_process_cache(snapshot['expected_base_cache'])
    metadata = _read_base_metadata(_pure._local_path(
        str(Path(snapshot['spec']['checkpoint']) / 'head.pt')))
    base = snapshot['expected_base_cache']
    if (metadata.get('base') != base['base'] or
            metadata.get('base_revision') != base['revision']):
        raise ValueError('checkpoint base metadata mismatch')
    try:
        from decision_ft.state_json_native import StateJsonNativeRuntime

        runtime = StateJsonNativeRuntime(
            snapshot['spec'], expected_binding=snapshot['expected_binding'],
            expected_checkpoint_files=snapshot['expected_checkpoint_files'],
            expected_sources=snapshot['expected_sources'])
        predictor = StateJsonRepresentedPredictor(runtime, snapshot)
        identity(predictor, snapshot)  # Check the actual, not merely claimed, environment.
        return predictor
    except Exception:
        # Keep the CLI rejection boundary without echoing dependency/checkpoint data.
        # No alternative model, generic Runtime, precision or calibration is attempted.
        raise ValueError('StateJson native initialization failed; no fallback used') from None
