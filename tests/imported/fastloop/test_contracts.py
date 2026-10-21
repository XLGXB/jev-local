import copy
import json
import math
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from fastloop.authority import make_grant, verify_grant
from fastloop.contracts import Guard, Task, origin, validate_values
from fastloop.model_client import select_reply

ROOT = Path(__file__).resolve().parent  # package-scoped imported fixtures


def task():
    d = json.loads((ROOT / "examples/registry.json").read_text(encoding="utf-8"))["tasks"]["demo_profile"]
    d.update(execution_approved=True, expected_runtime_fingerprint="a" * 64)
    d["forms"][0]["input_effects_reviewed"] = True
    return Task.model_validate(d)


def handoff():
    return {"run_id": "r", "target_key": "t", "policy_hash": "p", "handoff_id": "h",
            "form_id": "f", "doc_hash": "d", "form_signature": "s", "field_ids": ["name"], "expires_at": 1250}


def test_grant_round_trip():
    k = Ed25519PrivateKey.generate(); h = handoff(); v = {"name": "example"}
    token = make_grant(k, h, v, now=1000)
    assert verify_grant(k.public_key(), token, h, v, now=1001)["operation"] == "fill_only"


@pytest.mark.parametrize("case", ["wrong_key", "tampered", "run", "target", "document", "form", "fields", "values", "expired", "boolean_only", "submit"])
def test_grant_fail_closed(case):
    k = Ed25519PrivateKey.generate(); h = handoff(); v = {"name": "example"}
    t = make_grant(k, h, v, now=1000); pub = k.public_key(); now = 1001
    if case == "wrong_key": pub = Ed25519PrivateKey.generate().public_key()
    if case == "tampered": t["claims"]["expires_at"] += 10
    if case == "run": h["run_id"] = "other"
    if case == "target": h["target_key"] = "other"
    if case == "document": h["doc_hash"] = "other"
    if case == "form": h["form_signature"] = "other"
    if case == "fields": v["extra"] = "value"
    if case == "values": v["name"] = "different"
    if case == "expired": now = 1300
    if case == "boolean_only": t = {"approved": True}
    if case == "submit": t["claims"]["allow_submit"] = True
    with pytest.raises(Guard): verify_grant(pub, t, h, v, now=now)


@pytest.mark.parametrize("case", ["missing", "extra", "too_long", "invalid_option", "checkbox_text"])
def test_field_validation(case):
    form = task().forms[0]
    values = {"display_name": "test", "note": "ok", "language": "zh", "notify": False}
    if case == "missing": values.pop("note")
    if case == "extra": values["password"] = "not-allowed"
    if case == "too_long": values["display_name"] = "x" * 81
    if case == "invalid_option": values["language"] = "xx"
    if case == "checkbox_text": values["notify"] = "false"
    with pytest.raises(Guard): validate_values(form, values)


@pytest.mark.parametrize("case", ["runtime", "probability", "nan", "candidate", "calibration", "confidence", "reported"])
def test_model_contract_guards(case):
    t = task(); criteria = {"click:open": "open", "needs_host": "stop"}
    r = {"runtime_fingerprint": "a" * 64, "answers": {"next_action": {
        "selected_key": "click:open", "probabilities": {"click:open": .999, "needs_host": .001},
        "p_selected": .999, "calibrated": True}}}
    a = r["answers"]["next_action"]
    if case == "runtime": r["runtime_fingerprint"] = "b" * 64
    if case == "probability": a["probabilities"]["click:open"] = 1.1
    if case == "nan": a["probabilities"]["click:open"] = math.nan
    if case == "candidate": a["probabilities"]["evil"] = 0
    if case == "calibration": a["calibrated"] = False
    if case == "confidence": a.update(probabilities={"click:open": .6, "needs_host": .4}, p_selected=.6)
    if case == "reported": a["p_selected"] = True
    with pytest.raises(Guard): select_reply(r, criteria, t)


def test_password_kind_cannot_be_configured():
    d = task().model_dump(); d["forms"][0]["fields"][0]["input_type"] = "password"
    with pytest.raises(ValidationError): Task.model_validate(d)


def test_strict_profile_rejects_privilege_flags():
    d = task().model_dump(); d["allow_submit"] = True
    with pytest.raises(ValidationError): Task.model_validate(d)


def test_origin_is_exact_not_prefix():
    assert origin("https://example.com/a?q=x") == "https://example.com"
    assert origin("https://example.com.attacker.test/a") != "https://example.com"


def test_string_cannot_enable_execution_permission():
    d = task().model_dump(); d["execution_approved"] = "true"
    with pytest.raises(ValidationError): Task.model_validate(d)
