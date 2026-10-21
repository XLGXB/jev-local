"""Wire-compatible with local_decision_deploy_v1's /v1/decide service."""
import math
import uuid

import httpx

from .contracts import Guard, loopback
from decision_deploy.protocol import validate_envelope, validate_reply


def select_reply(reply, criteria, task, dry_run=False):
    if not dry_run and reply.get("runtime_fingerprint") != task.expected_runtime_fingerprint:
        raise Guard("unqualified_model_runtime")
    a = reply.get("answers", {}).get("next_action", {})
    probs = a.get("probabilities", {})
    if not isinstance(probs, dict) or set(probs) != set(criteria):
        raise Guard("model_candidate_set_changed")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
        raise Guard("invalid_model_probabilities")
    key = a.get("selected_key")
    if key not in probs or key != max(probs, key=probs.get):
        raise Guard("invalid_model_selection")
    reported = a.get("p_selected")
    if type(reported) not in (int, float) or not math.isfinite(reported) or abs(reported - probs[key]) > 1e-6:
        raise Guard("inconsistent_selected_probability")
    if not dry_run and (a.get("calibrated") is not True or probs[key] < task.min_probability):
        raise Guard("model_calibration_or_confidence_gate")
    return key


class HTTPDecision:
    def __init__(self, endpoint, token):
        loopback(endpoint)
        if len(token) < 32:
            raise Guard("missing_local_decision_token")
        self.client = httpx.AsyncClient(base_url=endpoint, follow_redirects=False, trust_env=False,
                                       headers={"Authorization": "Bearer " + token})

    async def __call__(self, state, criteria, observation_id, timeout, task, dry_run):
        request_id = uuid.uuid4().hex
        payload = {"request_id": request_id, "observation_id": observation_id, "state": state,
                   "questions": {"next_action": {"type": "choice",
                       "instructions": "依据当前授权目标、可见候选和已验证记忆选择下一步。不得自行填写、提交或扩大权限。没有安全进展时选 needs_host。",
                       "criteria": criteria}}}
        try:
            validate_envelope(payload)
        except ValueError as exc:
            raise Guard("invalid_model_request") from exc
        response = await self.client.post("/v1/decide", json=payload, timeout=timeout)
        response.raise_for_status()
        if len(response.content) > 65536:
            raise Guard("oversized_model_response")
        reply = response.json()
        if not isinstance(reply, dict):
            raise Guard("invalid_model_response")
        if reply.get("request_id") != request_id or reply.get("observation_id") != observation_id:
            raise Guard("stale_model_response")
        try:
            validate_reply(payload, reply)
        except ValueError as exc:
            raise Guard("invalid_model_response") from exc
        return select_reply(reply, criteria, task, dry_run)

    async def close(self):
        await self.client.aclose()
