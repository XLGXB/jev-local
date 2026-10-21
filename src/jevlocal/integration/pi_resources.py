"""Process-owned Pi resource policy, CUDA cap, Kev call counters, and safe audit.

Nothing native is imported until ``start`` on the first valid request. The
guard is deliberately process-local; it is not a system-wide GPU quota.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid


POLICY_MAX_BYTES = 8192
AUDIT_MAX_BYTES = 2 * 1024 * 1024
AUDIT_MAX_RECORDS = 8192
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_KEYS = frozenset({
    "schema_version", "config_sha256", "device", "max_cuda_allocated_bytes",
    "max_model_loads", "max_tokenizer_loads", "max_forward_calls", "max_wall_seconds",
})
_LIMITS = {"model": "max_model_loads", "tokenizer": "max_tokenizer_loads",
           "forward": "max_forward_calls"}


class ResourceFault(RuntimeError):
    """A resource boundary or audit failure; never expose details on stdio."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate policy key")
        result[key] = value
    return result


def _regular_path(value: str) -> Path:
    if type(value) is not str or value.startswith(("\\\\", "//")):
        raise ValueError("absolute local policy path required")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("absolute local policy path required")
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("reparse policy path forbidden")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("regular policy file required")
    return path


def load_policy(path_text: str, config_sha256: str) -> tuple[Path, dict, str]:
    """Strictly admit a small local policy bound to actual config file bytes."""
    path = _regular_path(path_text)
    if type(config_sha256) is not str or not _HEX.fullmatch(config_sha256):
        raise ValueError("invalid config digest")
    with path.open("rb") as stream:
        raw = stream.read(POLICY_MAX_BYTES + 1)
    if not raw or len(raw) > POLICY_MAX_BYTES:
        raise ValueError("policy byte budget exceeded")
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("policy UTF-8 BOM forbidden")
    policy = json.loads(text, object_pairs_hook=_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite policy")))
    if type(policy) is not dict or set(policy) != _KEYS:
        raise ValueError("policy fields invalid")
    if (type(policy["schema_version"]) is not int or policy["schema_version"] != 1 or
            type(policy["config_sha256"]) is not str or
            policy["config_sha256"] != config_sha256 or
            type(policy["device"]) is not str or policy["device"] != "cuda"):
        raise ValueError("policy identity invalid")
    for key in ("max_cuda_allocated_bytes", "max_model_loads", "max_tokenizer_loads",
                "max_forward_calls", "max_wall_seconds"):
        value = policy[key]
        if type(value) is not int or value <= 0:
            raise ValueError("policy limit invalid")
    if (policy["max_model_loads"] != 1 or policy["max_tokenizer_loads"] != 1 or
            policy["max_forward_calls"] > 4096 or policy["max_wall_seconds"] > 86400):
        raise ValueError("policy limit out of range")
    return path, policy, hashlib.sha256(raw).hexdigest()


class ResourceGuard:
    """Owned by one worker; inject native modules only for CPU-only fake tests."""

    def __init__(self, path: Path, policy: dict, policy_sha256: str, *,
                 torch_module=None, kev_model=None, kev_checkpoint=None,
                 clock=time.monotonic):
        self.path, self.policy, self.policy_sha256 = path, policy, policy_sha256
        self.parent_pid = os.getppid()
        if type(self.parent_pid) is not int or self.parent_pid <= 0:
            raise ResourceFault("parent PID invalid")
        self.torch, self.kev_model, self.kev_checkpoint = torch_module, kev_model, kev_checkpoint
        self.clock = clock
        self.started = False
        self.start_time = None
        self.attempted = {key: 0 for key in _LIMITS}
        self.completed = {key: 0 for key in _LIMITS}
        self._originals = []
        self._records = 0
        self._bytes = 0
        self._closed = False
        name = f"{path.name}.resource.{os.getpid()}.{uuid.uuid4().hex}.jsonl"
        self.audit_path = path.with_name(name)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        self._fd = os.open(self.audit_path, flags, 0o600)
        try:
            self._audit("admitted", "none")
        except Exception:
            os.close(self._fd)
            self._closed = True
            raise

    @classmethod
    def from_file(cls, path_text: str, config_sha256: str, config: dict, **kwargs):
        path, policy, digest = load_policy(path_text, config_sha256)
        if config.get("spec", {}).get("device") != "cuda":
            raise ValueError("resource policy requires CUDA native config")
        return cls(path, policy, digest, **kwargs)

    def _memory(self) -> tuple[int | None, int | None]:
        if not self.started or self.torch is None:
            return None, None
        cuda = self.torch.cuda
        peak = cuda.max_memory_allocated()
        allocated = cuda.memory_allocated()
        if type(peak) is not int or type(allocated) is not int or peak < 0 or allocated < 0:
            raise ResourceFault("CUDA memory reading invalid")
        return peak, allocated

    def _check(self) -> None:
        if self.start_time is not None and self.clock() - self.start_time > self.policy["max_wall_seconds"]:
            raise ResourceFault("wall budget exceeded")
        peak, allocated = self._memory()
        if (peak is not None and
                (peak > self.policy["max_cuda_allocated_bytes"] or
                 allocated > self.policy["max_cuda_allocated_bytes"])):
            raise ResourceFault("CUDA allocated budget exceeded")
        for key, policy_key in _LIMITS.items():
            if self.attempted[key] > self.policy[policy_key]:
                raise ResourceFault("Kev call budget exceeded")

    def _wrap(self, owner, name: str, key: str) -> None:
        original = vars(owner)[name]
        @functools.wraps(original)
        def counted(*args, **kwargs):
            self.attempted[key] += 1
            self._check()  # Reject an excess attempt before the underlying call.
            try:
                result = original(*args, **kwargs)
                self.completed[key] += 1
                return result
            finally:
                self._check()
        setattr(owner, name, counted)
        self._originals.append((owner, name, original, counted))

    def start(self) -> None:
        if self.started:
            self._check()
            return
        self.start_time = self.clock()
        try:
            if self.torch is None:
                import torch
                self.torch = torch
            cuda = self.torch.cuda
            if not cuda.is_available():
                raise ResourceFault("CUDA unavailable")
            total = cuda.get_device_properties(0).total_memory
            cap = self.policy["max_cuda_allocated_bytes"]
            if type(total) is not int or not 0 < cap < total:
                raise ResourceFault("CUDA cap must be strictly below actual total")
            # Round toward zero so floating conversion cannot enlarge byte cap.
            cuda.set_per_process_memory_fraction(math.nextafter(cap / total, 0.0), 0)
            # Kev imports only after the own-process allocator cap is set.
            if self.kev_model is None:
                import kev.model as model
                self.kev_model = model
            if self.kev_checkpoint is None:
                import kev.checkpoint as checkpoint
                self.kev_checkpoint = checkpoint
            if self.kev_checkpoint.DecisionModel is not self.kev_model.DecisionModel:
                raise ResourceFault("Kev model alias changed")
            if self.kev_checkpoint.load_tokenizer is not self.kev_model.load_tokenizer:
                raise ResourceFault("Kev tokenizer alias changed")
            self._wrap(self.kev_model.DecisionModel, "__init__", "model")
            self._wrap(self.kev_model.DecisionModel, "forward_batch", "forward")
            self._wrap(self.kev_model, "load_tokenizer", "tokenizer")
            # checkpoint.load bound its own alias at module import time.
            original_alias = self.kev_checkpoint.load_tokenizer
            self.kev_checkpoint.load_tokenizer = self.kev_model.load_tokenizer
            self._originals.append((self.kev_checkpoint, "load_tokenizer",
                                    original_alias, self.kev_model.load_tokenizer))
            self.started = True
            self._check()
            self._audit("started", "none")
        except Exception:
            self._restore()
            raise

    def _restore(self) -> None:
        for owner, name, original, wrapper in reversed(self._originals):
            if vars(owner).get(name) is wrapper:
                setattr(owner, name, original)
        self._originals.clear()

    def before_factory(self) -> None:
        self.start()
        self._check()

    def after_factory(self) -> None:
        self._check()

    def before_predict(self) -> None:
        self._check()

    def after_predict(self) -> None:
        self._check()

    def record(self, phase: str, error_class: str = "none") -> None:
        self._check()
        self._audit(phase, error_class)

    def fault(self, phase: str, error_class: str) -> None:
        self._audit(phase, error_class)

    def _audit(self, phase: str, error_class: str) -> None:
        if self._closed or self._records >= AUDIT_MAX_RECORDS:
            raise ResourceFault("audit unavailable")
        if phase not in {"admitted", "started", "request", "terminal"} or error_class not in {
                "none", "invalid_request", "initialization_failed", "prediction_failed",
                "resource_limit", "audit_failed"}:
            raise ResourceFault("audit enum invalid")
        peak, allocated = self._memory()
        elapsed_ms = (int(max(0, self.clock() - self.start_time) * 1000)
                      if self.start_time is not None else None)
        record = {"schema_version": 1, "pid": os.getpid(), "parent_pid": self.parent_pid,
                  "policy_sha256": self.policy_sha256,
                  "config_sha256": self.policy["config_sha256"],
                  "phase": phase, "error_class": error_class,
                  "attempted": dict(self.attempted), "completed": dict(self.completed),
                  "own_peak_allocated_bytes": peak,
                  "own_allocated_bytes": allocated, "elapsed_ms": elapsed_ms}
        raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if self._bytes + len(raw) > AUDIT_MAX_BYTES or len(raw) > 1024:
            raise ResourceFault("audit byte budget exceeded")
        try:
            pending = memoryview(raw)
            while pending:
                written = os.write(self._fd, pending)
                if written <= 0:
                    raise OSError("audit write failed")
                pending = pending[written:]
        except OSError as exc:
            raise ResourceFault("audit write failed") from exc
        self._bytes += len(raw)
        self._records += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._audit("terminal", "none")
        finally:
            self._restore()
            os.close(self._fd)
            self._closed = True
