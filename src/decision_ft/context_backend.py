"""Explicit, source-bound Kev context adapter; importing this module needs only stdlib.

This does not change the native encoder's global training defaults. A checkpoint
made through this adapter needs an explicit consumer of its context marker.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


CONTEXT_MARKER = "context_policy.json"
ENCODER_SOURCES = frozenset({"kev.model", "kev.data", "decision_ft.backends"})
_CONTEXT_FIELDS = frozenset({"schema_version", "max_state", "max_branch", "max_packed", "max_encoding_tokens"})
_MARKER_FIELDS = frozenset({"schema_version", "status", "context", "context_sha256",
                            "context_backend_sha256", "encoder_sources_sha256"})
_STATUS = "experimental_context_requires_explicit_consumer"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], expected[key]) for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def validate_context(context: dict) -> dict:
    """Validate the one supported policy and return a detached exact mapping."""
    if type(context) is not dict or set(context) != _CONTEXT_FIELDS:
        raise ValueError("context requires exact policy fields")
    if any(type(value) is not int for value in context.values()):
        raise ValueError("context values must be exact integers")
    state = context["max_state"]
    if context["schema_version"] != 1 or not 384 <= state <= 7552:
        raise ValueError("unsupported context schema or state limit")
    extra = state - 384
    if context["max_branch"] != 1024 + extra or context["max_packed"] != 2048 + extra:
        raise ValueError("context row and packed limits must preserve branch budget")
    encoding = context["max_encoding_tokens"]
    if not state < encoding <= min(context["max_branch"], context["max_packed"]):
        raise ValueError("encoding budget must exceed state and fit row and packed limits")
    return copy.deepcopy(context)


def context_digest(context: dict) -> str:
    policy = validate_context(context)
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_sha(value: Any, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be lowercase SHA256")
    return value


def _validate_encoder_sources(value: Any) -> dict[str, str]:
    if type(value) is not dict or set(value) != ENCODER_SOURCES:
        raise ValueError("encoder source hashes require exact module names")
    return {name: _validate_sha(value[name], name) for name in sorted(ENCODER_SOURCES)}


def _encoder_source_hashes() -> dict[str, str]:
    """Hash real import origins without importing torch-dependent child modules."""
    result = {}
    for name in sorted(ENCODER_SOURCES):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"encoder source unavailable: {name}")
        path = Path(spec.origin)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"encoder source is not a regular file: {name}")
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if type(origin) is not str or Path(origin).resolve() != path.resolve():
                raise ValueError(f"loaded encoder source origin differs: {name}")
        result[name] = _sha_file(path)
    return result


def _checked_encoder_sources(expected: dict[str, str]) -> dict[str, str]:
    expected = _validate_encoder_sources(expected)
    if _encoder_source_hashes() != expected:
        raise ValueError("encoder source hash mismatch")
    return expected


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate context marker key: {key}")
        result[key] = value
    return result


def read_context_marker(
    directory: str | Path, *, expected_context: dict,
    expected_backend_sha256: str, expected_encoder_sources: dict[str, str],
) -> dict:
    """Reject absent, malformed, stale, or differently bound context markers."""
    expected_context = validate_context(expected_context)
    _validate_sha(expected_backend_sha256, "backend hash")
    expected_encoder_sources = _checked_encoder_sources(expected_encoder_sources)
    if _sha_file(__file__) != expected_backend_sha256:
        raise ValueError("context backend source hash mismatch")
    path = Path(directory) / CONTEXT_MARKER
    marker = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if type(marker) is not dict or set(marker) != _MARKER_FIELDS:
        raise ValueError("invalid context marker fields")
    if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
        raise ValueError("invalid context marker schema")
    if type(marker["status"]) is not str or marker["status"] != _STATUS:
        raise ValueError("invalid context marker status")
    if not _strict_equal(marker["context"], expected_context):
        raise ValueError("context marker policy mismatch")
    if (marker["context_sha256"] != context_digest(expected_context)
            or marker["context_backend_sha256"] != expected_backend_sha256
            or not _strict_equal(marker["encoder_sources_sha256"], expected_encoder_sources)):
        raise ValueError("context marker hash mismatch")
    return marker


class ContextKevBackend:
    """Context-scoped adapter of one already loaded native KevBackend."""

    name = "kev"

    def __init__(self, native_backend: Any, context: dict, *, expected_encoder_sources: dict[str, str]):
        self._context = validate_context(context)
        self.context_sha256 = context_digest(self._context)
        self._encoder_sources = _checked_encoder_sources(expected_encoder_sources)
        self.native = native_backend
        self.model = native_backend.model
        self.device = native_backend.device
        self.tok = native_backend.tok
        self.meta = native_backend.meta
        self.data = native_backend.data
        self.provenance = copy.deepcopy(native_backend.provenance)
        self.source_files = [*native_backend.source_files, __file__]
        self.cache: dict[tuple[str, str], dict] = {}

    @property
    def context(self) -> dict:
        return copy.deepcopy(self._context)

    @property
    def expected_encoder_sources(self) -> dict[str, str]:
        return dict(self._encoder_sources)

    def train_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        return self.native.train_groups(backbone_lr, head_lr)

    def source_manifest(self) -> dict[str, str]:
        _checked_encoder_sources(self._encoder_sources)
        sources = self.native.source_manifest()
        return {**sources, str(Path(__file__)): _sha_file(__file__)}

    def encode(self, atom: Any) -> dict:
        key = (self.context_sha256, atom.fingerprint)
        if key in self.cache:
            return self.cache[key]
        question = copy.deepcopy(atom.question)
        question["label"] = (atom.keys[0] if question["type"] == "choice"
                             else False if question["type"] == "noul" else 0)
        question["src"] = "local"
        record = self.data.materialize({"state": copy.deepcopy(atom.state),
                                        "questions": {"q": question}})
        if record["questions"][0]["keys"] != atom.keys:
            raise ValueError("Kev option order contract mismatch")
        encoded = self.model.encode(self.tok, record, strict=True,
                                    max_state=self._context["max_state"],
                                    max_branch=self._context["max_branch"])
        if type(encoded) is not dict or type(encoded.get("ids")) is not list:
            raise ValueError("native encoding contract changed")
        if encoded.get("state_truncated") is not False:
            raise ValueError("truncated evidence is not permitted")
        length = len(encoded["ids"])
        if length > self._context["max_branch"] or length > self._context["max_packed"]:
            raise ValueError("native encoding exceeds context row or packed budget")
        if length > self._context["max_encoding_tokens"]:
            raise ValueError("native encoding exceeds explicit token budget")
        encoded.pop("labels", None)
        self.cache[key] = encoded
        return encoded

    def forward(self, atoms: list[Any]) -> list[Any]:
        nested = self.model.forward_batch([self.encode(atom) for atom in atoms])
        return [row[0].float() for row in nested]

    def export(self, directory: Path) -> None:
        directory = Path(directory)
        sources = _checked_encoder_sources(self._encoder_sources)
        self.native.export(directory)
        marker = {
            "schema_version": 1,
            "status": _STATUS,
            "context": self.context,
            "context_sha256": self.context_sha256,
            "context_backend_sha256": _sha_file(__file__),
            "encoder_sources_sha256": sources,
        }
        with (directory / CONTEXT_MARKER).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(marker, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
