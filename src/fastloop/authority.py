"""Trusted approval broker. PRIVATE KEY MUST BE OUTSIDE CODEX/EXECUTOR ACCESS.
The executor only imports verify_grant and only receives the public key.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .contracts import Guard, canonical, digest

BINDINGS = ("run_id", "target_key", "policy_hash", "handoff_id", "form_id", "doc_hash", "form_signature")


def make_grant(private: Ed25519PrivateKey, handoff: dict, values: dict, *, now=None, ttl=120) -> dict:
    """Call ONLY after an independent human/host review, never as an exposed model tool."""
    now = time.time() if now is None else now
    if set(values) != set(handoff["field_ids"]):
        raise Guard("grant_field_set_mismatch")
    exp = min(now + ttl, handoff["expires_at"])
    if not 0 < exp - now <= 300:
        raise Guard("handoff_expired")
    salt = secrets.token_hex(32)
    claims = {k: handoff[k] for k in BINDINGS}
    claims.update(v=1, aud="fastloop.form", operation="fill_only", allow_submit=False,
                  jti=secrets.token_hex(24), issued_at=now, expires_at=exp,
                  salt=salt, values_digest=digest({"salt": salt, "values": values}),
                  field_ids=sorted(values))
    signature = private.sign(canonical(claims))
    return {"claims": claims, "signature": base64.b64encode(signature).decode("ascii")}


def verify_grant(public: Ed25519PublicKey, token: dict, handoff: dict, values: dict,
                 *, now=None, check_time=True) -> dict:
    try:
        if set(token) != {"claims", "signature"}:
            raise Guard("invalid_grant_envelope")
        c = token["claims"]
        public.verify(base64.b64decode(token["signature"], validate=True), canonical(c))
        expected = set(BINDINGS) | {"v", "aud", "operation", "allow_submit", "jti", "issued_at", "expires_at", "salt", "values_digest", "field_ids"}
        if set(c) != expected or c["v"] != 1 or c["aud"] != "fastloop.form" or c["operation"] != "fill_only" or c["allow_submit"] is not False:
            raise Guard("grant_scope_invalid")
        if any(c[k] != handoff[k] for k in BINDINGS):
            raise Guard("grant_binding_mismatch")
        if c["field_ids"] != sorted(values) or set(values) != set(handoff["field_ids"]):
            raise Guard("grant_field_set_mismatch")
        if c["values_digest"] != digest({"salt": c["salt"], "values": values}):
            raise Guard("grant_values_changed")
        now = time.time() if now is None else now
        if check_time and not (c["issued_at"] <= now + 5 and now < c["expires_at"] <= min(c["issued_at"] + 300, handoff["expires_at"])):
            raise Guard("grant_expired")
        return c
    except Guard:
        raise
    except (InvalidSignature, ValueError, TypeError, KeyError) as exc:
        raise Guard("invalid_grant_signature") from exc


def write_private(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def main():
    p = argparse.ArgumentParser(description="Run as a trusted approver, NOT from Codex's tool process")
    sub = p.add_subparsers(dest="command", required=True)
    key = sub.add_parser("keygen")
    key.add_argument("--private", required=True); key.add_argument("--public", required=True)
    approve = sub.add_parser("approve")
    approve.add_argument("--private", required=True); approve.add_argument("--handoff", required=True)
    approve.add_argument("--values", required=True); approve.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "keygen":
        k = Ed25519PrivateKey.generate()
        write_private(Path(args.private), k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        Path(args.public).write_bytes(k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
        return
    handoff = json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    if "handoff" in handoff:
        handoff = handoff["handoff"]
    values = json.loads(Path(args.values).read_text(encoding="utf-8"))
    # Escape all control characters; page text is NOT rendered as terminal instructions.
    print(json.dumps({"operation": "fill_only (may trigger reviewed input/autosave effects)",
                      "run_id": handoff["run_id"], "origin": handoff["origin"], "form": handoff["form_id"],
                      "fields": handoff["fields"], "values": values, "submit": False}, ensure_ascii=True, indent=2))
    if input("Review the exact origin, fields and values. Type APPROVE_FILL: ") != "APPROVE_FILL":
        raise SystemExit("not approved")
    private = serialization.load_pem_private_key(Path(args.private).read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise SystemExit("Ed25519 private key required")
    token = make_grant(private, handoff, values)
    write_private(Path(args.out), canonical(token))
    print("Grant written. Its issuer key must remain inaccessible to Codex and the executor.")


if __name__ == "__main__":
    main()
