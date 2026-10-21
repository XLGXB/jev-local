"""Bounded, in-process coordination for passive native shadow observations.

The embedding host owns the predictor and independently executes its live action.
This module neither dispatches actions nor grants execution or training permission.
An undrained close leaves native work owned by the embedding process.
"""
from __future__ import annotations

import copy
from collections import deque
import math
import threading
import time
import weakref

from decision_deploy.protocol import validate_envelope
from jevlocal.integration.shadow import (
    LocalShadow, ShadowObservation, assess_feedback, request_digest,
)


def _timeout(value, *, allow_zero=False):
    if (type(value) not in (int, float) or value > 60 or
            (value < 0 if allow_zero else value <= 0) or not math.isfinite(value)):
        raise ValueError('finite bounded timeout required')
    return float(value)


class _Ticket:
    """Opaque caller-owned handle; its snapshot is never part of a receipt."""

    __slots__ = ('_owner', '_snapshot', '_record', '_deadline', '_stage',
                 '_observation', '_collected', '_result_lock', '_completed',
                 '__weakref__')

    def __init__(self, owner, snapshot, record, deadline):
        self._owner = owner
        self._snapshot = snapshot
        self._record = record
        self._deadline = deadline
        self._stage = 'new'
        self._observation = None
        self._collected = False
        self._result_lock = threading.Lock()
        self._completed = None


class ShadowHost:
    """One serialized worker for one exclusively owned native predictor."""

    def __init__(self, predictor, *, max_pending=4):
        if type(max_pending) is not int or not 1 <= max_pending <= 16:
            raise ValueError('max_pending must be an integer in 1..16')
        self._shadow = LocalShadow(predictor)
        self._backend = predictor.runtime.backend
        self._fingerprint = predictor.runtime.fingerprint
        self._max_pending = max_pending
        self._condition = threading.Condition()
        self._queue = deque()
        self._current = None
        self._pending = 0
        self._closed = False
        self._tickets = weakref.WeakSet()
        self._worker = threading.Thread(target=self._run, name='ShadowHost', daemon=False)
        self._worker.start()

    def _ticket(self, ticket):
        if not isinstance(ticket, _Ticket) or ticket._owner is not self:
            raise ValueError('foreign shadow ticket')

    @staticmethod
    def _unavailable(ticket, reason, *, error_type=None):
        record = copy.deepcopy(ticket._record)
        record['unavailable_reason'] = reason
        if error_type is not None:
            record['error_type'] = error_type
        return ShadowObservation(ticket._snapshot, record)

    def _retire_queued(self, ticket, reason):
        """Under the condition lock, remove queued work and its capacity."""
        self._queue.remove(ticket)
        ticket._observation = self._unavailable(ticket, reason)
        ticket._stage = 'done'
        self._pending -= 1
        self._condition.notify_all()

    def begin(self, envelope, *, timeout_s=5.0):
        timeout = _timeout(timeout_s)
        snapshot = copy.deepcopy(envelope)
        keys = validate_envelope(snapshot)
        record = dict(schema_version=1, request_id=snapshot['request_id'],
                      observation_id=snapshot['observation_id'],
                      request_sha256=request_digest(snapshot),
                      candidate_keys=keys, backend=self._backend,
                      runtime_fingerprint=self._fingerprint, status='unavailable',
                      answers={}, shadow=True, execution_authorized=False,
                      training_allowed=False, cloud_data_used=False)
        ticket = _Ticket(self, snapshot, record, time.monotonic() + timeout)
        with self._condition:
            if self._closed:
                ticket._observation = self._unavailable(ticket, 'closed')
                ticket._stage = 'done'
            elif self._pending >= self._max_pending:
                ticket._observation = self._unavailable(ticket, 'saturated')
                ticket._stage = 'done'
            else:
                ticket._stage = 'queued'
                self._queue.append(ticket)
                self._tickets.add(ticket)
                self._pending += 1
                self._condition.notify()
        return ticket

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                ticket = self._queue.popleft()
                if time.monotonic() >= ticket._deadline:
                    ticket._observation = self._unavailable(ticket, 'queue_expired')
                    ticket._stage = 'done'
                    self._pending -= 1
                    self._condition.notify_all()
                    del ticket
                    continue
                ticket._stage = 'running'
                self._current = ticket
            try:
                observation = self._shadow.observe(ticket._snapshot)
            except Exception as exc:
                # LocalShadow handles predictor errors itself. This guards other
                # unexpected failures without echoing private exception text.
                observation = self._unavailable(ticket, 'inference_failed',
                                                error_type=type(exc).__name__)
            # Publish the complete result under a separate short lock before
            # waiting for the queue condition. A collector at the deadline can
            # see an inference that finished on time even if this worker has
            # not yet acquired the queue condition.
            with ticket._result_lock:
                ticket._completed = (time.monotonic(), observation)
            with self._condition:
                if ticket._observation is None:
                    ticket._observation = (observation if ticket._completed[0] <= ticket._deadline
                                           else self._unavailable(ticket, 'timeout'))
                ticket._stage = 'done'
                self._current = None
                self._pending -= 1
                self._condition.notify_all()
            # An idle worker must not retain the last raw request through its
            # loop locals after caller-owned tickets have been released.
            del ticket, observation

    def collect(self, ticket):
        self._ticket(ticket)
        with self._condition:
            while ticket._observation is None:
                remaining = ticket._deadline - time.monotonic()
                if remaining <= 0:
                    with ticket._result_lock:
                        completed = ticket._completed
                    if completed is not None and completed[0] <= ticket._deadline:
                        ticket._observation = completed[1]
                    elif ticket._stage == 'queued':
                        self._retire_queued(ticket, 'queue_expired')
                    else:
                        # Running work still occupies its native slot until it
                        # actually returns; only its result is discarded here.
                        ticket._observation = self._unavailable(ticket, 'timeout')
                        self._condition.notify_all()
                    break
                self._condition.wait(remaining)
            ticket._collected = True
            self._tickets.discard(ticket)
            return ticket._observation

    def cancel(self, ticket):
        self._ticket(ticket)
        with self._condition:
            if ticket._collected:
                return
            if ticket._stage == 'queued':
                self._retire_queued(ticket, 'cancelled')
            else:
                # A completed but uncollected result is still future use, and
                # running native work retains capacity despite cancellation.
                ticket._observation = self._unavailable(ticket, 'cancelled')
                self._condition.notify_all()

    def feedback(self, ticket, evidence, *, verifier, verifier_id):
        return assess_feedback(self.collect(ticket), evidence,
                               verifier=verifier, verifier_id=verifier_id)

    def close(self, *, timeout_s=5.0):
        """Close admission; drained means no queued/running inference remains.

        A zero-time close can report drained before the idle worker exits. A
        running native call remains owned by the process until it really ends.
        """
        timeout = _timeout(timeout_s, allow_zero=True)
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            for ticket in list(self._queue):
                self._retire_queued(ticket, 'cancelled')
            for ticket in list(self._tickets):
                if not ticket._collected:
                    ticket._observation = self._unavailable(ticket, 'cancelled')
            self._condition.notify_all()
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            pending = self._pending
        if threading.current_thread() is not self._worker:
            self._worker.join(max(0.0, deadline - time.monotonic()))
        with self._condition:
            pending = self._pending
        return dict(closed=True, drained=(pending == 0), pending=pending)
