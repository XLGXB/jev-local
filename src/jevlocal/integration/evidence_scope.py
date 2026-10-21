"""Pure, explicit evidence applicability preparation; never a relevance verdict."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata

from .context_tools import (
    _budget, _prepare_rank_evidence, _sanitize, _secrets, _size, _too_large,
)


_ADAPTER = "jevlocal-evidence-scope-v1"
_CANDIDATE_KEYS = {"id", "source", "start_line", "text"}
_BINDING_KEYS = {"object_id", "revision", "candidate_sha256"}
_ARGUMENT_KEYS = {"query", "candidates", "target", "bindings", "max_chars",
                  "rank_max_chars", "known_secrets"}


def _json(value: object) -> str:
    """Canonical, type-sensitive JSON; no hidden Python object coercion."""
    try:
        def check(item: object) -> None:
            if item is None or type(item) in (bool, str, int):
                return
            if type(item) is float and math.isfinite(item):
                return
            if type(item) is list:
                for member in item:
                    check(member)
                return
            if type(item) is dict:
                for key, member in item.items():
                    if type(key) is not str:
                        raise ValueError("non-JSON key")
                    check(member)
                return
            raise ValueError("non-JSON value")

        check(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (RecursionError, TypeError, OverflowError, UnicodeError):
        raise ValueError("invalid JSON value") from None


def _digest(value: object) -> str:
    try:
        return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
    except (RecursionError, TypeError, OverflowError, UnicodeError):
        raise ValueError("invalid JSON value") from None


def _candidate(candidate: object) -> dict:
    if (type(candidate) is not dict or
            not {"id", "source", "text"} <= set(candidate) or
            set(candidate) - _CANDIDATE_KEYS):
        raise ValueError("invalid candidate schema")
    # Preserve the original metadata and bytes. Shared rank validation supplies
    # the same bounds, ID grammar, and start-line rule as its consumer.
    _prepare_rank_evidence("validation", [candidate], max_chars=35)
    normalized = {"id": candidate["id"], "source": candidate["source"],
                  "start_line": candidate.get("start_line", 1),
                  "text": candidate["text"]}
    _json(normalized)
    return normalized


def candidate_digest(candidate: dict) -> str:
    """Hash a rank-schema candidate before any redaction or partitioning."""
    try:
        return _digest(_candidate(candidate))
    except (RecursionError, TypeError, OverflowError, UnicodeError):
        raise ValueError("invalid candidate") from None


def _identity(value: object, secrets: tuple[str, ...], *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    if (type(value) is not str or not value.strip() or len(value) > 256 or
            any(unicodedata.category(char) in ("Cc", "Cf", "Cs") for char in value) or
            _sanitize(value, secrets) != value):
        raise ValueError("invalid scope identity")
    return value


def _prepare(query: str, candidates: list[dict], target: dict,
             bindings: dict, *, max_chars: int, rank_max_chars: int,
             known_secrets: list[str] | tuple[str, ...]) -> dict:
    _budget(max_chars)
    _budget(rank_max_chars)
    secrets = _secrets(known_secrets)
    safe_query, safe_candidates = _prepare_rank_evidence(
        query, candidates, max_chars=rank_max_chars, known_secrets=secrets)
    if type(target) is not dict or set(target) != {"object_id", "revision"}:
        raise ValueError("target requires object_id and revision")
    safe_target = {key: _identity(target[key], secrets, nullable=False)
                   for key in ("object_id", "revision")}
    ids = {item["id"] for item in safe_candidates}
    if type(bindings) is not dict or set(bindings) != ids:
        raise ValueError("bindings must cover exactly all candidate IDs")

    safe_bindings = {}
    validated_bindings = {}
    # Finish validation of the full closed set before assigning any status.
    for original, safe in zip(candidates, safe_candidates):
        cid = safe["id"]
        binding = bindings[cid]
        if type(binding) is not dict or set(binding) != _BINDING_KEYS:
            raise ValueError("invalid binding schema")
        object_id = _identity(binding["object_id"], secrets, nullable=True)
        revision = _identity(binding["revision"], secrets, nullable=True)
        raw_digest = binding["candidate_sha256"]
        if type(raw_digest) is not str or raw_digest != candidate_digest(original):
            raise ValueError("candidate binding digest mismatch")
        validated_bindings[cid] = (object_id, revision)
        safe_bindings[cid] = {"object_id": object_id, "revision": revision,
                              "candidate_sha256": candidate_digest(safe)}
    applicability = []
    counts = {"total": len(safe_candidates), "match": 0,
              "mismatch": 0, "unknown": 0}
    matches = []
    for safe in safe_candidates:
        cid = safe["id"]
        object_id, revision = validated_bindings[cid]
        mismatches = [key + "_mismatch" for key, value in
                      (("object_id", object_id), ("revision", revision))
                      if value is not None and value != safe_target[key]]
        unknowns = [key + "_unknown" for key, value in
                    (("object_id", object_id), ("revision", revision))
                    if value is None]
        status = "mismatch" if mismatches else "unknown" if unknowns else "match"
        applicability.append({"id": cid, "status": status,
                              "reasons": mismatches if mismatches else unknowns})
        counts[status] += 1
        if status == "match":
            matches.append(safe)

    snapshot = {"query": safe_query, "candidates": safe_candidates,
                "target": safe_target, "bindings": safe_bindings,
                "rank_max_chars": rank_max_chars}
    rank_input = ({"query": safe_query, "candidates": matches,
                   "max_chars": rank_max_chars} if matches else None)
    result = {"version": 1, "adapter": _ADAPTER,
              "status": "prepared" if matches else "no_matches",
              "snapshot": snapshot, "snapshot_sha256": _digest(snapshot),
              "applicability": applicability, "counts": counts,
              "rank_input": rank_input,
              "rank_input_sha256": _digest(rank_input) if rank_input is not None else None,
              "execution_authorized": False, "training_allowed": False,
              "origin_authenticated": False}
    required = _size(result)
    return _too_large(required, max_chars) if required > max_chars else result


def scope_evidence(query: str, candidates: list[dict], target: dict,
                   bindings: dict, *, max_chars: int, rank_max_chars: int = 12000,
                   known_secrets: list[str] | tuple[str, ...] = ()) -> dict:
    """Partition an explicit, validated snapshot by exact declared identity."""
    try:
        return _prepare(query, candidates, target, bindings,
                        max_chars=max_chars, rank_max_chars=rank_max_chars,
                        known_secrets=known_secrets)
    except (RecursionError, TypeError, OverflowError, UnicodeError):
        raise ValueError("invalid evidence scope input") from None


def verify_scope_evidence(arguments: dict, prepared: dict,
                          rank_input: dict | None = None) -> dict:
    """Recompute the complete result; never execute, load, or follow a path."""
    if (type(arguments) is not dict or
            not {"query", "candidates", "target", "bindings", "max_chars"} <= set(arguments) or
            set(arguments) - _ARGUMENT_KEYS):
        raise ValueError("invalid scope arguments")
    # The library accepts tuple secrets as well as JSON lists; normalize only
    # this explicit option for the JSON-data check without mutating the caller.
    checked_arguments = dict(arguments)
    if type(checked_arguments.get("known_secrets")) is tuple:
        checked_arguments["known_secrets"] = list(checked_arguments["known_secrets"])
    _json(checked_arguments)
    _json(prepared)
    _json(rank_input)
    expected = scope_evidence(**arguments)
    if (expected.get("status") not in ("prepared", "no_matches") or
            type(prepared) is not dict or _json(prepared) != _json(expected)):
        raise ValueError("scope preparation mismatch")
    if expected["status"] == "prepared":
        if rank_input is None or _json(rank_input) != _json(expected["rank_input"]):
            raise ValueError("derived rank input mismatch")
    elif rank_input is not None:
        raise ValueError("no rank input permitted for no_matches")
    return {"status": "consistent", "counts": expected["counts"],
            "snapshot_sha256": expected["snapshot_sha256"],
            "rank_input_sha256": expected["rank_input_sha256"],
            "execution_authorized": False, "training_allowed": False,
            "origin_authenticated": False}
