from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def origin(url: str) -> str:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise Guard("unsupported_origin")
    host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
    port = p.port or (443 if p.scheme == "https" else 80)
    suffix = "" if port == (443 if p.scheme == "https" else 80) else f":{port}"
    return f"{p.scheme}://{host}{suffix}"


def loopback(url: str) -> str:
    p = urlsplit(url)
    if p.scheme not in ("http", "ws") or p.hostname not in ("localhost", "127.0.0.1", "::1") or p.username or p.password:
        raise Guard("endpoint_must_be_loopback")
    return url


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise Guard("invalid_id")
    return value


class Guard(RuntimeError):
    """Safe error codes only: never put form values or browser exception text here."""


class State(StrEnum):
    NEW = "NEW"
    OBSERVE = "OBSERVE"
    DECIDE = "DECIDE"
    VALIDATE = "VALIDATE"
    SCROLL = "SCROLL"
    WAIT_CONTENT = "WAIT_CONTENT"
    CLICK = "CLICK"
    VERIFY_STEP = "VERIFY_STEP"
    AWAITING_FORM = "AWAITING_FORM"
    VALIDATE_GRANT = "VALIDATE_GRANT"
    CODEX_FILL = "CODEX_FILL"
    VERIFY_FORM = "VERIFY_FORM"
    READY = "READY"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    NEEDS_HOST = "NEEDS_HOST"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    CANCELLED = "CANCELLED"
    DRY_RUN = "DRY_RUN"
    CLOSED_AFTER_REVIEW = "CLOSED_AFTER_REVIEW"


TERMINAL = {State.NEEDS_VERIFICATION, State.NEEDS_HOST, State.OUTCOME_UNKNOWN,
            State.CANCELLED, State.DRY_RUN, State.CLOSED_AFTER_REVIEW}
# Any active state can stop. OUTCOME_UNKNOWN intentionally keeps the logical target lease.
EDGES = {
    State.NEW: {State.OBSERVE},
    State.OBSERVE: {State.DECIDE},
    State.DECIDE: {State.VALIDATE},
    State.VALIDATE: {State.SCROLL, State.CLICK, State.AWAITING_FORM},
    State.SCROLL: {State.WAIT_CONTENT},
    State.WAIT_CONTENT: {State.VERIFY_STEP},
    State.CLICK: {State.VERIFY_STEP},
    State.VERIFY_STEP: {State.OBSERVE},
    State.AWAITING_FORM: {State.VALIDATE_GRANT},
    State.VALIDATE_GRANT: {State.CODEX_FILL},
    State.CODEX_FILL: {State.VERIFY_FORM},
    State.VERIFY_FORM: {State.READY},
    State.READY: {State.OBSERVE},
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Check(Strict):
    selector: str = Field(min_length=1, max_length=400)
    text: str | None = None


class ClickSpec(Strict):
    id: str
    selector: str
    label: str
    expected_text: str
    postcondition: Check
    requires_forms: list[str] = Field(default_factory=list)


class ScrollSpec(Strict):
    id: str
    selector: str = "@document"
    axis: Literal["y", "x"] = "y"
    step_ratio: float = Field(default=0.75, ge=0.1, le=0.9)
    item_selector: str | None = None
    item_key_attr: str = "data-key"
    busy_selector: str | None = None
    max_steps: int = Field(default=10, ge=1, le=40)
    no_progress_limit: int = Field(default=2, ge=1, le=4)
    settle_timeout_ms: int = Field(default=1800, ge=200, le=10000)
    quiet_ms: int = Field(default=120, ge=30, le=1000)
    edge_wait_ms: int = Field(default=500, ge=0, le=5000)

    @model_validator(mode="after")
    def timing(self):
        if max(self.quiet_ms, self.edge_wait_ms) >= self.settle_timeout_ms:
            raise ValueError("settle timeout must exceed quiet and edge wait")
        return self


class FieldSpec(Strict):
    id: str
    selector: str
    label: str
    kind: Literal["text", "textarea", "select", "checkbox"] = "text"
    input_type: Literal["text", "email", "tel", "url"] = "text"
    sensitivity: Literal["ordinary"] = "ordinary"
    max_chars: int = Field(default=200, ge=1, le=4000)
    choices: list[str] = Field(default_factory=list)


class FormSpec(Strict):
    id: str
    selector: str
    label: str
    fields: list[FieldSpec] = Field(min_length=1, max_length=12)
    input_effects_reviewed: bool = False
    grant_ttl_seconds: int = Field(default=180, ge=10, le=300)


class Task(Strict):
    goal: str
    allowed_origins: list[str] = Field(min_length=1)
    execution_approved: bool = False
    expected_runtime_fingerprint: str = "UNQUALIFIED"
    min_probability: float = Field(default=0.99, gt=0, le=1)
    clicks: list[ClickSpec] = Field(default_factory=list)
    scrolls: list[ScrollSpec] = Field(default_factory=list)
    forms: list[FormSpec] = Field(default_factory=list)
    verifier: Check
    max_steps: int = Field(default=32, ge=1, le=100)
    max_scrolls: int = Field(default=16, ge=1, le=60)
    segment_ms: int = Field(default=15000, ge=1000, le=60000)
    max_active_ms: int = Field(default=60000, ge=1000, le=300000)
    run_ttl_seconds: int = Field(default=1800, ge=30, le=7200)
    model_timeout_ms: int = Field(default=5000, ge=100, le=15000)
    max_model_bytes: int = Field(default=12000, ge=1000, le=65536)
    memory_tail: int = Field(default=6, ge=1, le=12)

    @model_validator(mode="after")
    def safety(self):
        if any(origin(o) != o for o in self.allowed_origins):
            raise ValueError("origins must be normalized exact origins")
        if self.execution_approved and not re.fullmatch(r"[a-f0-9]{64}", self.expected_runtime_fingerprint):
            raise ValueError("approved profile must pin runtime SHA-256")
        for group in (self.clicks, self.scrolls, self.forms):
            ids = [safe_id(x.id) for x in group]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate profile id")
        form_ids = {f.id for f in self.forms}
        for click in self.clicks:
            if not set(click.requires_forms) <= form_ids:
                raise ValueError("unknown required form")
        for form in self.forms:
            ids = [safe_id(f.id) for f in form.fields]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate field id")
            for f in form.fields:
                if f.kind == "select" and not f.choices:
                    raise ValueError("select needs approved choices")
        return self

    def form(self, id: str) -> FormSpec:
        return next(f for f in self.forms if f.id == id)


def validate_values(form: FormSpec, values: dict) -> None:
    if not isinstance(values, dict) or set(values) != {f.id for f in form.fields}:
        raise Guard("field_set_mismatch")
    for field in form.fields:
        value = values[field.id]
        if field.kind == "checkbox":
            if type(value) is not bool:
                raise Guard("checkbox_requires_boolean")
        elif not isinstance(value, str) or len(value) > field.max_chars or "\x00" in value:
            raise Guard("invalid_field_value")
        elif field.kind == "select" and value not in field.choices:
            raise Guard("unapproved_select_value")
    canonical(values)
