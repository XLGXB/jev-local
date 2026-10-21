"""Explicit, non-authoritative context helpers for caller-supplied material.

No directory is indexed and no model is invoked here. A callback, when supplied,
is a caller-owned typed scoring route; failure never silently switches routes.
The lexical baseline is not Jev. Secret filtering is best-effort, not complete.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Callable


class ModelRouteError(RuntimeError):
    """A supplied scoring route failed; no implicit fallback was attempted."""


_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = re.compile(r"(?i)\b(error|exception|failed|fatal|traceback)\b|错误|失败|异常")
_PRIVATE = re.compile(r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?(?:-----END [^-\r\n]*PRIVATE KEY-----|\Z)", re.I | re.S)
_SECRET_PATTERNS = (
    re.compile(r"""(?i)["'](?:api[_-]?key|password|passwd|token|secret)["']\s*:\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,}]+)"""),
    re.compile(r"(?i)\bAuthorization\s*:\s*[^\r\n]+"),
    re.compile(r"(?i)\bCookie\s*:\s*[^\r\n]+"),
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|password|passwd|token|secret)\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)"),
)


def _string(value: object, name: str, *, allow_empty: bool = False, limit: int = 262144) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or len(value) > limit:
        raise ValueError(f"{name} must be a bounded string")
    return value


def _budget(value: int) -> None:
    if type(value) is not int or value < 35:
        raise ValueError("max_chars must be an integer >= 35")


def _sanitize(value: str, known_secrets: tuple[str, ...]) -> str:
    def replacement(original: str, marker: str) -> str:
        # Preserve each original newline sequence so source line references stay valid.
        return marker + "".join(re.findall(r"\r\n|\r|\n", original))

    result = _PRIVATE.sub(lambda match: replacement(match.group(), "[REDACTED_PRIVATE_KEY]"), value)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda match: replacement(match.group(), "[REDACTED]"), result)
    for secret in known_secrets:
        result = result.replace(secret, replacement(secret, "[REDACTED_KNOWN_SECRET]"))
    return result


def _secrets(known_secrets: object) -> tuple[str, ...]:
    if not isinstance(known_secrets, (list, tuple)) or len(known_secrets) > 128:
        raise ValueError("known_secrets must be a bounded list or tuple")
    for item in known_secrets:
        _string(item, "known secret", limit=4096)
    return tuple(sorted(known_secrets, key=len, reverse=True))


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", value.lower()))


def _lexical(query: str, value: str) -> float:
    words = _tokens(query)
    if not words:
        return 0.0
    overlap = len(words & _tokens(value)) / len(words)
    bonus = 0.2 if query.lower() in value.lower() else 0.0
    return min(1.0, overlap + bonus)


def _scores(callback: Callable, query: str, candidates: list[dict], ids: set[str]) -> dict[str, float]:
    try:
        answer = callback(query, [dict(item) for item in candidates])
    except Exception:
        raise ModelRouteError("typed callback failed; no fallback used") from None
    if not isinstance(answer, dict) or set(answer) != ids:
        raise ValueError("callback must score exactly the supplied closed-set IDs")
    for value in answer.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("callback scores/probabilities must be finite numbers in [0, 1]")
    return answer


def _size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _too_large(required: int, max_chars: int) -> dict:
    result = {"status": "requires_larger_budget", "required_chars": required}
    if _size(result) > max_chars:
        return {"status": "requires_larger_budget"}
    return result


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _store(store_dir: str | os.PathLike, text: str) -> str:
    root = Path(store_dir)
    if root.is_symlink():
        raise ValueError("store root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    digest = _hash(text)
    target = root / (digest + ".txt")
    if target.is_symlink() or target.resolve(strict=False).parent != root:
        raise ValueError("store path escapes caller-owned root")
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("stored content hash mismatch")
        return digest
    temp = root / ("." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_bytes(text.encode("utf-8"))
        os.replace(temp, target)
        if _hash(target.read_bytes().decode("utf-8")) != digest:
            raise ValueError("stored content hash mismatch after write")
    finally:
        temp.unlink(missing_ok=True)
    return digest


def restore_lines(store_dir: str | os.PathLike, sha256: str,
                  start_line: int, end_line: int) -> str:
    """Restore exact UTF-8 text including original newline characters (1-based)."""
    if not isinstance(sha256, str) or not _DIGEST.fullmatch(sha256):
        raise ValueError("invalid SHA-256; path components are forbidden")
    if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
        raise ValueError("invalid line range")
    root = Path(store_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid store root")
    root = root.resolve(strict=True)
    target = root / (sha256 + ".txt")
    if target.is_symlink() or target.resolve(strict=False).parent != root:
        raise ValueError("store path escapes caller-owned root")
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("stored content hash mismatch")
    lines = data.decode("utf-8").splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError("line range exceeds stored text")
    return "".join(lines[start_line - 1:end_line])


def rank_evidence(query: str, candidates: list[dict], *, max_chars: int,
                  store_dir: str | os.PathLike | None = None,
                  known_secrets: list[str] | tuple[str, ...] = (),
                  score_callback: Callable | None = None) -> dict:
    """Rank explicit text candidates; return cited excerpts, not instructions."""
    query, safe = _prepare_rank_evidence(query, candidates, max_chars=max_chars,
                                       store_dir=store_dir, known_secrets=known_secrets)
    return _rank_prepared_evidence(query, safe, max_chars=max_chars,
                                   store_dir=store_dir, score_callback=score_callback)


def _prepare_rank_evidence(query: str, candidates: list[dict], *, max_chars: int,
                           store_dir=None, known_secrets=()):
    """Internal shared validation boundary for active and passive host entries."""
    _budget(max_chars)
    secrets = _secrets(known_secrets)
    query = _sanitize(_string(query, "query", limit=8192), secrets)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
        raise ValueError("candidates must be a bounded nonempty list")
    safe = []
    ids = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or
                not {"id", "source", "text"} <= set(candidate) or
                set(candidate) - {"id", "source", "start_line", "text"}):
            raise ValueError("candidate requires id, source, text; start_line is optional")
        cid = _string(candidate["id"], "candidate ID", limit=128)
        if (not _ID.fullmatch(cid) or cid in ids or
                _sanitize(cid, secrets) != cid):
            raise ValueError("invalid or duplicate candidate ID")
        ids.add(cid)
        source = _sanitize(_string(candidate["source"], "source", limit=1024), secrets)
        original = _string(candidate["text"], "candidate text", limit=262144)
        if store_dir is not None and _PRIVATE.search(original):
            raise ValueError("sensitive block cannot be persisted")
        line = candidate.get("start_line", 1)
        if type(line) is not int or line < 1:
            raise ValueError("start_line must be positive")
        safe.append({"id": cid, "source": source, "start_line": line,
                     "text": _sanitize(original, secrets)})
    return query, safe


def _rank_prepared_evidence(query, safe, *, max_chars, store_dir=None,
                            score_callback=None):
    """Rank only a snapshot returned by _prepare_rank_evidence; not a tool API."""
    ids = {c["id"] for c in safe}
    callback_view = [{"id": c["id"], "source": c["source"],
                      "text": c["text"]} for c in safe]
    method = "typed_callback" if score_callback is not None else "lexical"
    scores = (_scores(score_callback, query, callback_view, ids) if score_callback is not None else
              {c["id"]: _lexical(query, c["text"]) for c in safe})
    ranked = []
    for c in safe:
        lines = c["text"].splitlines(keepends=True)
        line_index = max(range(len(lines)), key=lambda i: (_lexical(query, lines[i]), -i))
        ranked.append({"id": c["id"], "source": c["source"],
                       "score": scores[c["id"]], "quote": lines[line_index],
                       "start_line": c["start_line"] + line_index,
                       "end_line": c["start_line"] + line_index,
                       "sha256": _hash(c["text"]), "stored": store_dir is not None,
                       "restore_start_line": line_index + 1,
                       "restore_end_line": line_index + 1})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    result = {"status": "ok", "method": method, "results": ranked,
              "execution_authorized": False}
    if _size(result) > max_chars:
        return _too_large(_size(result), max_chars)
    if store_dir is not None:
        for c in safe:
            _store(store_dir, c["text"])
    return result


def filter_tool_output(command: str, exit_code: int, output: str, goal: str,
                       constraints: list[str], *, max_chars: int,
                       store_dir: str | os.PathLike | None = None,
                       known_secrets: list[str] | tuple[str, ...] = ()) -> dict:
    """Preserve reproducibility, mandatory context and complete failure output."""
    _budget(max_chars)
    secrets = _secrets(known_secrets)
    command = _sanitize(_string(command, "command", limit=8192), secrets)
    goal = _sanitize(_string(goal, "goal", limit=8192), secrets)
    output = _string(output, "output", allow_empty=True, limit=1048576)
    if store_dir is not None and _PRIVATE.search(output):
        raise ValueError("sensitive block cannot be persisted")
    output = _sanitize(output, secrets)
    if type(exit_code) is not int:
        raise ValueError("exit_code must be an integer")
    if not isinstance(constraints, list) or len(constraints) > 64:
        raise ValueError("constraints must be a bounded list")
    constraints = [_sanitize(_string(x, "constraint", limit=8192), secrets) for x in constraints]
    lines = output.splitlines(keepends=True)
    found = next((i for i, line in enumerate(lines) if _ERROR.search(line)), None)
    first = {"line": found + 1, "text": lines[found]} if found is not None else None
    retain_failure = exit_code != 0 or found is not None
    excerpt = output if retain_failure else "".join(lines[:3])
    result = {"status": "ok", "method": "full_failure_output" if retain_failure else "leading_context",
              "command": command, "exit_code": exit_code, "goal": goal,
              "constraints": constraints, "first_error": first,
              "excerpt": excerpt, "excerpt_start_line": 1,
              "excerpt_end_line": len(lines) if retain_failure else min(3, len(lines)), "sha256": _hash(output),
              "stored": store_dir is not None, "execution_authorized": False}
    if _size(result) > max_chars:
        return _too_large(_size(result), max_chars)
    if store_dir is not None:
        _store(store_dir, output)
    return result


def suggest_skills(task: str, enabled_skills: list[dict], *, max_chars: int,
                   decision_callback: Callable | None = None,
                   known_secrets: list[str] | tuple[str, ...] = ()) -> dict:
    """Suggest only from the caller's enabled closed set; never install or run."""
    task, safe = _prepare_suggest_skills(task, enabled_skills, max_chars=max_chars,
                                        known_secrets=known_secrets)
    return _suggest_prepared_skills(task, safe, max_chars=max_chars,
                                    decision_callback=decision_callback)


def _prepare_suggest_skills(task: str, enabled_skills: list[dict], *, max_chars: int,
                            known_secrets=()):
    """Freeze the same sanitized closed set for active and passive entries."""
    _budget(max_chars)
    secrets = _secrets(known_secrets)
    task = _sanitize(_string(task, "task", limit=8192), secrets)
    if not isinstance(enabled_skills, list) or len(enabled_skills) > 128:
        raise ValueError("enabled_skills must be a bounded list")
    safe = []
    ids = set()
    for item in enabled_skills:
        if not isinstance(item, dict) or set(item) != {"id", "description"}:
            raise ValueError("skill requires id and description")
        sid = _string(item["id"], "skill ID", limit=128)
        if (not _ID.fullmatch(sid) or sid in ids or
                _sanitize(sid, secrets) != sid):
            raise ValueError("invalid or duplicate skill ID")
        ids.add(sid)
        safe.append({"id": sid, "description": _sanitize(
            _string(item["description"], "skill description", limit=4096), secrets)})
    return task, safe


def _suggest_prepared_skills(task, safe, *, max_chars, decision_callback=None):
    """Compute the unchanged primary from a prepared skill snapshot."""
    ids = {item["id"] for item in safe}
    method = "typed_callback" if decision_callback is not None else "lexical"
    scores = (_scores(decision_callback, task, safe, ids) if decision_callback is not None else
              {s["id"]: _lexical(task, s["description"]) for s in safe})
    suggestions = [{"id": s["id"], "score": scores[s["id"]]}
                   for s in safe if scores[s["id"]] > 0]
    suggestions.sort(key=lambda x: (-x["score"], x["id"]))
    result = {"status": "ok", "method": method, "suggestions": suggestions,
              "execution_authorized": False}
    if _size(result) > max_chars:
        return _too_large(_size(result), max_chars)
    return result
