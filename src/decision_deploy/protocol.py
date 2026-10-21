"""Public wire contracts. Validation never grants permission to act."""
from __future__ import annotations

import json
import math
from typing import Any

from decision_ft.data import option_keys

MAX_BYTES = 128 * 1024
MAX_QUESTIONS = 8
MAX_CANDIDATES = 16


def _json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("finite JSON required") from exc


def _id(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("nonempty ID of at most 128 characters required")


def validate_request(request: dict) -> dict[str, list[str]]:
    """Validate the model's public input; return semantic candidate keys."""
    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        raise ValueError("request needs state and questions only")
    questions = request["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError("1..8 questions required")
    keysets = {}
    for qid, question in questions.items():
        _id(qid)
        if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("question labels and extra fields are forbidden")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError("question instructions required")
        keys = option_keys(question)
        if len(keys) > MAX_CANDIDATES:
            raise ValueError("deployment profile supports at most 16 candidates")
        for key in keys:
            _id(key)
        criteria = question.get("criteria")
        if criteria is not None:
            descriptions = criteria.values() if isinstance(criteria, dict) else criteria
            if any(not isinstance(value, str) or not value.strip() for value in descriptions):
                raise ValueError("candidate descriptions must be nonempty strings")
        keysets[qid] = keys
    if len(_json(request)) > MAX_BYTES:
        raise ValueError("request byte budget exceeded; no silent truncation")
    return keysets


def validate_envelope(envelope: dict) -> dict[str, list[str]]:
    if not isinstance(envelope, dict) or set(envelope) != {"request_id", "observation_id", "state", "questions"}:
        raise ValueError("invalid four-field envelope")
    _id(envelope["request_id"])
    _id(envelope["observation_id"])
    if len(_json(envelope)) > MAX_BYTES:
        raise ValueError("request byte budget exceeded")
    return validate_request({"state": envelope["state"], "questions": envelope["questions"]})


def validate_reply(envelope: dict, reply: dict) -> None:
    """Bind all answers to one observation and reject missing/extra decisions."""
    keysets = validate_envelope(envelope)
    if not isinstance(reply, dict):
        raise ValueError("reply must be an object")
    if set(reply) - {"request_id", "observation_id", "runtime_fingerprint", "answers", "inference_total_ms"}:
        raise ValueError("unknown response fields")
    if "inference_total_ms" in reply:
        elapsed = reply["inference_total_ms"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid inference timing")
    if any(reply.get(key) != envelope[key] for key in ("request_id", "observation_id")):
        raise ValueError("stale or unrelated decision response")
    if not isinstance(reply.get("runtime_fingerprint"), str) or not reply["runtime_fingerprint"]:
        raise ValueError("runtime identity required")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(keysets):
        raise ValueError("answer question set changed")
    for qid, keys in keysets.items():
        answer = answers[qid]
        if not isinstance(answer, dict):
            raise ValueError("answer must be an object")
        if set(answer) - {"selected_key", "probabilities", "p_selected", "calibrated", "temperature"}:
            raise ValueError("unknown answer fields")
        if "temperature" in answer:
            temperature = answer["temperature"]
            if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("invalid calibration temperature")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError("answer candidate set changed")
        values = list(probabilities.values())
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("invalid probabilities")
        if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        selected = answer.get("selected_key")
        if not isinstance(selected, str) or selected not in probabilities or probabilities[selected] != max(values):
            raise ValueError("selection must name a maximum-probability candidate")
        p_selected = answer.get("p_selected")
        if type(p_selected) not in (int, float) or not math.isfinite(p_selected) or not math.isclose(p_selected, probabilities[selected], abs_tol=1e-6):
            raise ValueError("inconsistent selected probability")
        if type(answer.get("calibrated")) is not bool:
            raise ValueError("explicit calibration status required")
    _json(reply)
