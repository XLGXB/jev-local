"""Passive local shadow decisions and separately verified observed outcomes.

Caller supplies authorized, sanitized pre-action input and owns one native
predictor. This library never executes a model-selected action, starts a worker,
persists raw state, or authorizes training. Feedback callbacks are trusted host
code, not JSON/model instructions, and must verify already collected evidence
without replaying the action. There is no cloud/teacher route.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
import time

from decision_deploy.protocol import validate_envelope, validate_reply


def request_digest(envelope: dict) -> str:
    validate_envelope(envelope)
    # Preserve insertion order: candidate order is part of the observed input.
    return hashlib.sha256(json.dumps(envelope, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


class ShadowObservation:
    """In-memory snapshot. receipt() deliberately excludes raw input text."""

    def __init__(self, request, record):
        self._request = copy.deepcopy(request)
        self._record = copy.deepcopy(record)

    def receipt(self) -> dict:
        return copy.deepcopy(self._record)


class LocalShadow:
    """One exclusively owned native predictor, serialized inside this instance."""

    def __init__(self, predictor):
        if predictor.runtime.mode != 'native' or predictor.runtime.backend not in ('laya', 'kev'):
            raise ValueError('shadow requires an explicitly supplied local native predictor')
        self._predictor = predictor
        self._fingerprint = predictor.runtime.fingerprint
        self._lock = threading.Lock()

    def observe(self, envelope: dict) -> ShadowObservation:
        snapshot = copy.deepcopy(envelope)
        keys = validate_envelope(snapshot)
        record = dict(schema_version=1, request_id=snapshot['request_id'],
            observation_id=snapshot['observation_id'], request_sha256=request_digest(snapshot),
            candidate_keys=keys, backend=self._predictor.runtime.backend,
            runtime_fingerprint=self._fingerprint, status='unavailable', answers={},
            shadow=True, execution_authorized=False, training_allowed=False, cloud_data_used=False)
        started = time.perf_counter()
        try:
            with self._lock:
                answer = self._predictor.predict(copy.deepcopy({
                    'state': snapshot['state'], 'questions': snapshot['questions']}))
            if (not isinstance(answer, dict) or
                    set(answer) != {'runtime_fingerprint', 'answers', 'inference_total_ms'} or
                    answer['runtime_fingerprint'] != self._fingerprint):
                raise ValueError('shadow runtime identity changed')
            validate_reply(snapshot, {'request_id': snapshot['request_id'],
                                     'observation_id': snapshot['observation_id'], **answer})
            record.update(status='observed', answers=copy.deepcopy(answer['answers']))
        except Exception as exc:
            # Missing shadow data is not a negative label, and never a live-route fallback.
            record['error_type'] = type(exc).__name__
        record['shadow_elapsed_ms'] = (time.perf_counter() - started) * 1000
        return ShadowObservation(snapshot, record)


def assess_feedback(observation: ShadowObservation, evidence: bytes, *, verifier, verifier_id: str) -> dict:
    """Run a host-owned program check, then compare only independently known labels.

    Verifier receives (original_snapshot, evidence_bytes), returning exactly
    request_sha256, outcome (success/failure/unknown), executed (qid -> key), and
    verified_labels (qid -> independently established key). Empty labels stay
    unknown. A pass/fail outcome alone never invents a correctness label.
    Labels may assess unexecuted alternatives only when independently decidable;
    never echo model labels or infer them solely from the executed action's
    outcome. "correct" measures this independent label, not action success.
    An unknown action outcome may coexist with independently verified labels;
    an empty label map, not outcome, expresses unknown proposition truth.
    """
    if not isinstance(observation, ShadowObservation) or not isinstance(evidence, bytes) or len(evidence) > 131072:
        raise ValueError('bounded evidence and an in-process observation required')
    if not callable(verifier) or not isinstance(verifier_id, str) or not verifier_id.strip() or len(verifier_id) > 128:
        raise ValueError('explicit host-owned verifier required')
    request, record = copy.deepcopy(observation._request), observation.receipt()
    if request_digest(request) != record['request_sha256']:
        raise ValueError('shadow snapshot changed')
    try:
        verifier_sha = hashlib.sha256(inspect.getsource(verifier).encode('utf-8')).hexdigest()
        verdict = verifier(copy.deepcopy(request), evidence)
    except Exception:
        raise ValueError('program feedback verification failed') from None
    if (not isinstance(verdict, dict) or set(verdict) != {'request_sha256', 'outcome', 'executed', 'verified_labels'} or
            verdict['request_sha256'] != record['request_sha256'] or
            verdict['outcome'] not in ('success', 'failure', 'unknown')):
        raise ValueError('feedback is not bound to the original observation')
    for field in ('executed', 'verified_labels'):
        values = verdict[field]
        if not isinstance(values, dict) or not set(values) <= set(record['candidate_keys']):
            raise ValueError('feedback question set changed')
        for qid, key in values.items():
            if not isinstance(key, str) or key not in record['candidate_keys'][qid]:
                raise ValueError('feedback candidate set changed')
    comparisons = {}
    for qid in record['candidate_keys']:
        local = record['answers'].get(qid, {}).get('selected_key')
        actual = verdict['executed'].get(qid)
        truth = verdict['verified_labels'].get(qid)
        comparisons[qid] = dict(local_key=local, executed_key=actual, verified_key=truth,
            correct=(local == truth) if local is not None and truth is not None else None,
            matches_executed=(local == actual) if local is not None and actual is not None else None,
            local_action_outcome=(verdict['outcome'] if local is not None and local == actual else 'unobserved'))
    record['feedback'] = dict(evidence_sha256=hashlib.sha256(evidence).hexdigest(),
        verifier_id=verifier_id, verifier_sha256=verifier_sha, outcome=verdict['outcome'],
        executed=copy.deepcopy(verdict['executed']), comparisons=comparisons)
    return record
