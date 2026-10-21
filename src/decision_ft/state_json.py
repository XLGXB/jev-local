"""Opt-in, source-bound JSON representation for a public request's state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


POLICY_ID = "jevlocal.state-json.v1"


def _check_json(value: Any, active: set[int]) -> None:
    """Reject values outside the exact built-in JSON value domain."""
    kind = type(value)
    if value is None or kind is bool or kind is int:
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return
    if kind is str:
        value.encode("utf-8", "strict")
        return
    if kind not in (list, dict):
        raise TypeError(f"not an exact built-in JSON type: {kind.__name__}")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic JSON value")
    active.add(identity)
    try:
        if kind is list:
            for item in value:
                _check_json(item, active)
        else:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("JSON object keys must be exact strings")
                key.encode("utf-8", "strict")
                _check_json(item, active)
    finally:
        active.remove(identity)


def render_state(state: Any) -> str:
    """Serialize a state as compact, ordered, strict UTF-8-compatible JSON."""
    _check_json(state, set())
    return json.dumps(
        state, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"), sort_keys=False,
    )


def policy_manifest() -> dict[str, Any]:
    """Return the immutable policy parameters and current serializer source digest."""
    return {
        "schema_version": 1,
        "id": POLICY_ID,
        "serializer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "format": {
            "encoding": "UTF-8 strict, no normalization",
            "scope": "state_only",
            "json_dumps": {
                "ensure_ascii": False,
                "allow_nan": False,
                "separators": [",", ":"],
                "sort_keys": False,
                "types": "exact built-in JSON types",
            },
        },
    }


def _same_json_shape(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_json_shape(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_json_shape(a, b) for a, b in zip(left, right)
        )
    return left == right


def prepare_public_request(raw: dict[str, Any], *, expected_policy: dict[str, Any]) -> dict[str, Any]:
    """Copy a public request, replacing only its state with bound JSON text."""
    if not _same_json_shape(expected_policy, policy_manifest()):
        raise ValueError("state JSON policy binding mismatch")
    if type(raw) is not dict or set(raw) - {"state", "questions", "model"}:
        raise ValueError("request must contain only public fields")
    if "state" not in raw or "questions" not in raw:
        raise ValueError("state and questions are required")
    questions = raw["questions"]
    if type(questions) is not dict or not questions:
        raise ValueError("questions must be a nonempty object")
    for question_id, question in questions.items():
        if type(question_id) is not str or type(question) is not dict:
            raise ValueError("question IDs and values must be public objects")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("question contains nonpublic or unknown fields")
        if question.get("type") not in ("noul", "choice", "score"):
            raise ValueError("unsupported question type")
    state_text = render_state(raw["state"])
    prepared = copy.deepcopy(raw)
    prepared["state"] = state_text
    return prepared
