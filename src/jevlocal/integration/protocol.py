"""Legacy compatibility is informational only, never an execution authority."""
from __future__ import annotations

from copy import deepcopy

from decision_deploy.protocol import validate_envelope, validate_reply


def legacy_to_envelope(request: dict, *, request_id: str, observation_id: str) -> dict:
    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        raise ValueError("legacy request requires exactly state and questions")
    envelope = {"request_id": request_id, "observation_id": observation_id, **deepcopy(request)}
    validate_envelope(envelope)
    return envelope


def reply_to_legacy(envelope: dict, reply: dict) -> dict:
    """Map by semantic key, not by probability-array position.

    Old noul vectors are [yes, no]; the new protocol uses keyed false/true.
    A legacy consumer cannot express the qualification/approval contract, so the
    compatibility response is always shadow-only even for calibrated answers.
    """
    validate_reply(envelope, reply)
    results = {}
    for qid, question in envelope["questions"].items():
        answer = reply["answers"][qid]
        kind = question["type"]
        probs = answer["probabilities"]
        keys = list(question["criteria"]) if kind == "choice" else (
            [str(i) for i in range(len(question["criteria"]))] if kind == "score" else ["true", "false"])
        values = [probs[key] for key in keys]
        result = {"type": kind, "probabilities": values, "confidence": answer["p_selected"],
                  "calibrated": answer["calibrated"]}
        if kind == "noul":
            result["noul"] = probs["true"]
        elif kind == "choice":
            result["choice"] = keys.index(answer["selected_key"])
        else:
            result["score"] = sum(i * p for i, p in enumerate(values)) / (len(values) - 1)
        results[qid] = result
    return {"model": "jevlocal-native-compat", "results": results,
            "runtime_fingerprint": reply["runtime_fingerprint"],
            "request_id": envelope["request_id"], "observation_id": envelope["observation_id"],
            "shadow": True, "execution_authorized": False}
