"""Explicit uncalibrated context-bound Kev route; legacy routes are unchanged.

Configuration admission is standard-library only. This route always preserves
full evidence, refuses an unmarked legacy checkpoint, and never grants authority.
"""
from __future__ import annotations
import copy,os,threading
from pathlib import Path
from . import state_json_advisory as _pure
from . import represented_native as _legacy
from decision_ft.context_backend import (CONTEXT_MARKER,ENCODER_SOURCES,
                                        validate_context,context_digest,read_context_marker)
ROUTE='context_native_uncalibrated_v1'
_CONTEXT_SOURCES=frozenset({'decision_ft.context_backend','decision_ft.context_native'})
_ADAPTER_SOURCES=_legacy._ADAPTER_SOURCES | {'jevlocal.integration.context_native'}
_CHECKPOINT_FILES=_pure._CHECKPOINT_FILES | {CONTEXT_MARKER}
_FIELDS=_legacy._FIELDS | {'context','expected_context_sources'}
_SPEC=_legacy._SPEC

def validate_config(config):
 if (type(config) is not dict or set(config)!=_FIELDS or
     type(config['schema_version']) is not int or config['schema_version']!=1 or
     config['route']!=ROUTE or config['calibration'] is not None):
  raise ValueError('invalid explicit context route config')
 context=validate_context(config['context']);context_sha=context_digest(context)
 spec=config['spec']
 if (type(spec) is not dict or set(spec)!=set(_SPEC)|{'device','checkpoint','context_sha256'} or
     any(type(spec[k]) is not str or spec[k]!=v for k,v in _SPEC.items()) or
     type(spec['device']) is not str or spec['device'] not in ('cpu','cuda') or
     type(spec['checkpoint']) is not str or spec['context_sha256']!=context_sha):
  raise ValueError('invalid explicit context native spec')
 checkpoint=_pure._local_path(spec['checkpoint'],directory=True)
 for key,keys in (('expected_sources',_pure._SOURCES),('expected_context_sources',_CONTEXT_SOURCES),
                  ('expected_adapter_sources',_ADAPTER_SOURCES),('expected_checkpoint_files',_CHECKPOINT_FILES)):
  _pure._hash_map(config[key],keys)
 sources=config['expected_sources'];context_sources=config['expected_context_sources']
 identity=config['expected_identity'];binding=config['expected_binding']
 if (type(identity) is not dict or set(identity)!={'representation','context','consumer_source_sha256',
     'checkpoint_files_sha256','sources_sha256','context_sources_sha256','spec','environment'} or
     type(identity['representation']) is not dict or set(identity['representation'])!={'binding','marker'} or
     type(identity['context']) is not dict or set(identity['context'])!={'policy','marker'} or
     type(identity['environment']) is not dict or
     type(config['expected_fingerprint']) is not str or not _pure._HEX.fullmatch(config['expected_fingerprint']) or
     _pure._digest(identity)!=config['expected_fingerprint']):
  raise ValueError('context runtime identity/fingerprint mismatch')
 if (_pure._canonical(binding)!=_pure._canonical(_legacy._current_binding(sources)) or
     _pure._canonical(identity['representation']['binding'])!=_pure._canonical(binding) or
     _pure._canonical(identity['context']['policy'])!=_pure._canonical(context) or
     identity['consumer_source_sha256']!=context_sources['decision_ft.context_native'] or
     identity['sources_sha256']!=sources or identity['context_sources_sha256']!=context_sources or
     identity['checkpoint_files_sha256']!=config['expected_checkpoint_files'] or
     identity['spec']!={k:v for k,v in spec.items() if k!='checkpoint'}):
  raise ValueError('context identity mirror/binding mismatch')
 for name,expected in (*sources.items(),*context_sources.items(),*config['expected_adapter_sources'].items()):
  if _pure._sha(_pure._module_file(name))!=expected:raise ValueError('context route source hash mismatch')
 if {entry.name for entry in checkpoint.iterdir()}!=_CHECKPOINT_FILES:
  raise ValueError('context checkpoint complete flat inventory mismatch')
 if _pure._checkpoint_files(checkpoint)!=config['expected_checkpoint_files']:
  raise ValueError('context checkpoint file hash mismatch')
 _legacy._check_base_cache(config['expected_base_cache'])
 marker_path=_pure._local_path(str(checkpoint/'state_representation.json'))
 if marker_path.stat().st_size>64*1024:raise ValueError('representation marker byte budget')
 marker=_pure._read_json(marker_path)
 if (type(marker) is not dict or set(marker)!={'schema_version','status','binding','training_freeze_sha256','epoch'} or
     type(marker['schema_version']) is not int or marker['schema_version']!=1 or
     marker['status']!='experimental_requires_explicit_consumer' or
     type(marker['training_freeze_sha256']) is not str or not _pure._HEX.fullmatch(marker['training_freeze_sha256']) or
     type(marker['epoch']) is not int or marker['epoch']<=0 or
     _pure._canonical(marker['binding'])!=_pure._canonical(binding) or
     _pure._canonical(marker)!=_pure._canonical(identity['representation']['marker'])):
  raise ValueError('context representation marker mismatch')
 context_path=_pure._local_path(str(checkpoint/CONTEXT_MARKER))
 if context_path.stat().st_size>64*1024:raise ValueError('context marker byte budget')
 context_marker=read_context_marker(checkpoint,expected_context=context,
       expected_backend_sha256=context_sources['decision_ft.context_backend'],
       expected_encoder_sources={name:sources[name] for name in ENCODER_SOURCES})
 if _pure._canonical(context_marker)!=_pure._canonical(identity['context']['marker']):
  raise ValueError('context checkpoint marker identity mismatch')
 return copy.deepcopy(config)

class ContextNativePredictor:
 def __init__(self,runtime,config):
  self._runtime=runtime;self.runtime=_legacy._RuntimeView(runtime);self.calibration=None
  self._config=copy.deepcopy(config);self._config_digest=_pure._digest(config);self._lock=threading.Lock()
 @property
 def metadata(self):
  c=self._config
  return copy.deepcopy({'schema_version':1,'kind':ROUTE,'route':ROUTE,'config_sha256':self._config_digest,
    'runtime_fingerprint':c['expected_fingerprint'],'spec':c['spec'],'context':c['context'],
    'binding':c['expected_binding'],'checkpoint_files_sha256':c['expected_checkpoint_files'],
    'expected_base_cache':c['expected_base_cache'],'sources_sha256':c['expected_sources'],
    'context_sources_sha256':c['expected_context_sources'],'adapter_source_sha256':c['expected_adapter_sources'],
    'calibration':None,'calibrated':False,'advisory_only':True,'execution_authorized':False,'semantic_quality':'unqualified'})
 def predict(self,request):
  with self._lock:
   native=self._runtime.native
   try:
    identity(self,self._config)
    from decision_ft.state_json_binding import prepare_inference_atoms
    pairs=prepare_inference_atoms(request,expected_binding=self._config['expected_binding'])
    if [qid for qid,_ in pairs]!=list(request['questions']):raise ValueError('context question order changed')
    for qid,atom in pairs:
     encoded=native.encode(atom)
     if (type(encoded) is not dict or encoded.get('state_truncated') is not False or
         type(encoded.get('ids')) is not list or not 1<=len(encoded['ids'])<=self._config['context']['max_encoding_tokens'] or
         any(type(token) is not int or token<0 for token in encoded['ids']) or
         atom.keys!=_pure._option_keys(request['questions'][qid])):
      raise ValueError('context representation truncated, over budget or reordered')
    identity(self,self._config)
    result=self._runtime.predict(request,calibration=None)
    identity(self,self._config)
    _legacy._validate_prediction(request,result,self._config)
    return result
   except (ValueError,TypeError,KeyError,OSError):raise
   except Exception:
    raise ValueError('context native prediction failed; no fallback used') from None
   finally:native.cache.clear()

def identity(predictor,config):
 if (type(predictor) is not ContextNativePredictor or predictor.calibration is not None or
     _pure._digest(config)!=predictor._config_digest or _pure._digest(predictor._config)!=predictor._config_digest):
  raise ValueError('context route config/calibration drift')
 runtime=predictor._runtime;c=predictor._config
 if (runtime.spec!=c['spec'] or _pure._canonical(runtime.identity)!=_pure._canonical(c['expected_identity']) or
     _pure._digest(runtime.identity)!=c['expected_fingerprint'] or runtime.fingerprint!=c['expected_fingerprint'] or
     context_digest(runtime.native.context)!=context_digest(c['context']) or
     predictor.runtime.mode!='native' or predictor.runtime.backend!='kev'):
  raise ValueError('context native runtime identity drift')
 return c['expected_fingerprint']

def build_predictor(config):
 c=validate_config(config)
 for flag in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','TORCH_FORCE_WEIGHTS_ONLY_LOAD'):
  if os.environ.get(flag)!='1':raise ValueError('context process-start offline/safe-loading flags required')
 _legacy._check_process_cache(c['expected_base_cache'])
 metadata=_legacy._read_base_metadata(_pure._local_path(str(Path(c['spec']['checkpoint'])/'head.pt')))
 base=c['expected_base_cache']
 if metadata.get('base')!=base['base'] or metadata.get('base_revision')!=base['revision']:
  raise ValueError('context checkpoint base metadata mismatch')
 try:
  from decision_ft.context_native import ContextStateJsonNativeRuntime
  runtime=ContextStateJsonNativeRuntime(c['spec'],expected_binding=c['expected_binding'],
    expected_checkpoint_files=c['expected_checkpoint_files'],expected_sources=c['expected_sources'],
    context=c['context'],expected_context_sources=c['expected_context_sources'])
  predictor=ContextNativePredictor(runtime,c);identity(predictor,c);return predictor
 except Exception:
  raise ValueError('context native initialization failed; no fallback used') from None
