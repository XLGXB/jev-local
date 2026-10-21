"""Complete bounded async state machine. No TODOs or implicit approval booleans.
The model can choose CLICK/SCROLL/HANDOFF only. Codex form writes use a separate signed path.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from .authority import verify_grant
from .contracts import EDGES, TERMINAL, Guard, State, Task, canonical, digest, validate_values

STABLE = TERMINAL | {State.AWAITING_FORM, State.READY}


class Machine:
    def __init__(self, *, task: Task, browser, predict, store, public_key,
                 run_id: str, target_id: str, target_key: str, project_id="local", task_id="task", dry_run=False):
        self.task, self.browser, self.predict, self.store, self.public_key = task, browser, predict, store, public_key
        self.run_id, self.target_id, self.target_key = run_id, target_id, target_key
        self.task_id, self.project_id, self.dry_run = task_id, project_id, dry_run
        self.policy_hash = digest(task.model_dump())
        self.scope = digest([project_id, task_id, self.policy_hash, target_key, dry_run])
        self.r = None
        self._charged = self._segment_start = 0.0

    def _segment(self):
        self._charged = self._segment_start = time.monotonic()

    def _charge(self):
        now = time.monotonic()
        self.r["active_ms"] += (now - self._charged) * 1000
        self._charged = now

    def remaining(self):
        active = self.r["active_ms"] + (time.monotonic() - self._charged) * 1000
        left = min(self.task.segment_ms / 1000 - (time.monotonic() - self._segment_start),
                   (self.task.max_active_ms - active) / 1000,
                   self.r["expires_at"] - time.time())
        if left <= 0:
            raise Guard("budget_or_run_expiry")
        return left

    async def call(self, coro):
        try:
            timeout = self.remaining()
        except BaseException:
            coro.close()
            raise
        return await asyncio.wait_for(coro, timeout=timeout)

    def _transition(self, nxt: State, reason: str | None = None):
        old = State(self.r["state"])
        if nxt not in TERMINAL and nxt not in EDGES.get(old, set()):
            raise Guard("illegal_state_transition")
        self._charge()
        self.r["state"] = nxt
        if reason:
            self.r["reason"] = reason
        self.store.save(self.r, {"from": old, "to": nxt, "reason": reason})

    def _load(self):
        self.r = self.store.get(self.run_id)
        if self.r["scope"] != self.scope:
            raise Guard("run_scope_mismatch")
        self._segment()

    def result(self, **extra):
        r = self.r
        names = {State.READY: "ready_to_resume", State.AWAITING_FORM: "awaiting_form"}
        out = {
            "status": names.get(State(r["state"]), r["state"].lower()),
            "state": r["state"], "run_id": r["run_id"], "target_id": r["target_id"],
            "controller": "codex" if r["state"] in (State.AWAITING_FORM, State.READY) else "stopped",
            "steps": r["steps"], "scrolls": r["scroll_count"], "active_ms": round(r["active_ms"], 2),
            "recent": r["memory"]["recent"][-self.task.memory_tail:],
        }
        for key in ("reason", "proposed", "form_result"):
            if key in r:
                out[key] = r[key]
        if r["state"] == State.AWAITING_FORM:
            out["handoff"] = r["handoff"]
        out.update(extra)
        return out

    def _abort(self, exc):
        reason = str(exc) if isinstance(exc, Guard) else f"operation_failed:{type(exc).__name__}"
        uncertain = bool(self.r.get("pending_effect") or self.r.get("grant_in_progress"))
        self._transition(State.OUTCOME_UNKNOWN if uncertain else State.NEEDS_HOST, reason)
        return self.result()

    async def start(self):
        if not self.dry_run and not self.task.execution_approved:
            raise Guard("task_not_execution_approved")
        now = time.time()
        record = {
            "run_id": self.run_id, "scope": self.scope, "target_id": self.target_id,
            "target_key": self.target_key, "project_id": self.project_id, "task_id": self.task_id,
            "policy_hash": self.policy_hash, "dry_run": self.dry_run, "state": State.NEW,
            "created_at": now, "expires_at": now + self.task.run_ttl_seconds,
            "steps": 0, "scroll_count": 0, "active_ms": 0.0, "pending_effect": None,
            "grant_in_progress": False, "used_grant": None, "handoff": None,
            "memory": {"doc": None, "clicks": [], "forms": {}, "scroll": {}, "attempts": {}, "recent": []},
        }
        created = self.store.create(record)
        self._load()
        if not created:
            if State(self.r["state"]) not in STABLE:
                return self._abort(Guard("interrupted_run_do_not_replay"))
            return self.result(idempotent_replay=True)
        return await self._drive()

    async def resume(self):
        self._load()
        if self.r["state"] != State.READY:
            if State(self.r["state"]) not in STABLE:
                return self._abort(Guard("interrupted_run_do_not_replay"))
            return self.result(resume_rejected=True)
        return await self._drive()

    def _doc_memory(self, snap):
        m = self.r["memory"]
        expected = self.r.pop("expected_resume_doc", None)
        if expected and snap.doc_hash != expected:
            raise Guard("document_changed_while_paused")
        if m["doc"] != snap.doc_hash:
            m.update(doc=snap.doc_hash, clicks=[], forms={}, scroll={})
            # attempts and recent remain bounded by the GLOBAL step budget.
        for form_id, fact in m["forms"].items():
            current = snap.data["forms"].get(form_id)
            if current and current["signature"] != fact["signature"]:
                raise Guard("verified_form_changed_requires_host_review")

    def _choices(self, snap):
        m, choices = self.r["memory"], {}
        for c in self.task.clicks:
            info = snap.data["clicks"].get(c.id)
            if info and info["viewport"] and info["hit"] and c.id not in m["clicks"] and all(fid in m["forms"] and fid in snap.data["forms"] and snap.data["forms"][fid]["signature"] == m["forms"][fid]["signature"] for fid in c.requires_forms):
                choices[f"click:{c.id}"] = c.label
        for f in self.task.forms:
            info = snap.data["forms"].get(f.id)
            if info and info["ready"] and f.id not in m["forms"]:
                choices[f"form:{f.id}"] = f"交还 Codex，为表单【{f.label}】取得逐字段填写授权；本模型不得输入。"
        if self.r["scroll_count"] < self.task.max_scrolls:
            for s in self.task.scrolls:
                info = snap.data["scrolls"].get(s.id)
                stats = m["scroll"].get(s.id, {"steps": 0, "up_stalls": 0, "down_stalls": 0})
                if not info or not info["visible"] or stats["steps"] >= s.max_steps:
                    continue
                size = info["height" if s.axis == "y" else "width"]
                if size <= 0:
                    continue
                if stats["down_stalls"] < s.no_progress_limit:
                    choices[f"scroll:{s.id}:down"] = f"在批准容器 {s.id} 沿 {s.axis} 正向滚动并观察新内容。"
                if info[s.axis] > 1 and stats["up_stalls"] < s.no_progress_limit:
                    choices[f"scroll:{s.id}:up"] = f"在批准容器 {s.id} 沿 {s.axis} 反向滚动并观察。"
        if len(choices) > 15:
            raise Guard("too_many_candidates_for_v1_model_contract")
        choices["needs_host"] = "证据不足、反复无进展或权限不够，交还 Codex。"
        return choices

    def _model_state(self, snap, choices):
        m = self.r["memory"]
        state = {
            "goal": self.task.goal, "origin": snap.data["document"]["origin"],
            "document_key": snap.doc_hash,
            "elements": [{"id": k, "description": v} for k, v in choices.items() if k != "needs_host"],
            "memory": {"recent_verified_actions": m["recent"][-self.task.memory_tail:],
                       "completed_click_ids": m["clicks"], "filled_form_ids": sorted(m["forms"]),
                       "scroll_progress": m["scroll"], "remaining_steps": self.task.max_steps - self.r["steps"]},
        }
        if len(canonical(state)) > self.task.max_model_bytes:
            raise Guard("model_state_budget_exceeded")
        return state

    async def _select(self, snap, choices):
        # predict adapter returns a candidate key only after validating raw response,
        # calibration, exact candidate set and runtime fingerprint.
        key = await self.call(self.predict(self._model_state(snap, choices), choices, snap.hash,
                                           min(self.remaining(), self.task.model_timeout_ms / 1000),
                                           self.task, self.dry_run))
        if key not in choices:
            raise Guard("model_selected_nonexistent_candidate")
        if key == "needs_host":
            raise Guard("model_requested_host")
        return key

    def _attempt(self, snap, key):
        code = digest([snap.doc_hash, snap.hash, key])
        attempts = self.r["memory"]["attempts"]
        if attempts.get(code, 0) >= 2:
            raise Guard("repeated_state_action_loop")
        attempts[code] = attempts.get(code, 0) + 1

    def _remember(self, key, before, after, **extra):
        event = {"action": key, "before": before.doc_hash, "after": after.doc_hash,
                 "observation_after": after.hash, "verified": True, **extra}
        m = self.r["memory"]
        m["recent"].append(event)
        m["recent"] = m["recent"][-self.task.memory_tail:]
        self.store.observation(self.r, after.data)
        self.store.save(self.r, event)

    def _handoff(self, snap, form_id):
        form = self.task.form(form_id)
        if not form.input_effects_reviewed:
            raise Guard("form_input_and_autosave_effects_not_reviewed")
        self.r["handoff"] = {
            "run_id": self.run_id, "target_key": self.target_key, "policy_hash": self.policy_hash,
            "handoff_id": secrets.token_hex(24), "form_id": form_id,
            "doc_hash": snap.doc_hash, "form_signature": snap.data["forms"][form_id]["signature"],
            "origin": snap.data["document"]["origin"],
            "expires_at": min(time.time() + form.grant_ttl_seconds, self.r["expires_at"]),
            "field_ids": [f.id for f in form.fields],
            "fields": [{"id": f.id, "label": f.label, "kind": f.kind, "max_chars": f.max_chars,
                        "choices": f.choices} for f in form.fields],
            "operation": "fill_only", "allow_submit": False,
        }
        self.r["used_grant"] = None

    async def _drive(self):
        snap = None
        try:
            self._transition(State.OBSERVE)
            while True:
                snap = await self.call(self.browser.snapshot())
                self._doc_memory(snap)
                self.r["last_observation"] = snap.hash
                self.store.observation(self.r, snap.data)
                if snap.data["verified"]:
                    self._transition(State.NEEDS_VERIFICATION, "task_predicate_observed_not_final_acceptance")
                    return self.result()
                if self.r["steps"] >= self.task.max_steps:
                    raise Guard("global_action_budget_exhausted")
                choices = self._choices(snap)
                if len(choices) < 2:
                    raise Guard("no_legal_action_or_scroll_progress")
                self._transition(State.DECIDE)
                key = await self._select(snap, choices)
                self._transition(State.VALIDATE)
                fresh = await self.call(self.browser.snapshot())
                try:
                    if fresh.hash != snap.hash:
                        raise Guard("observation_changed_before_dispatch")
                finally:
                    await fresh.close()
                if self.dry_run:
                    self.r["proposed"] = key
                    self._transition(State.DRY_RUN)
                    return self.result()
                self._attempt(snap, key)
                if key.startswith("form:"):
                    self._handoff(snap, key.split(":", 1)[1])
                    self._transition(State.AWAITING_FORM)
                    return self.result()
                if key.startswith("scroll:"):
                    _, id_, direction = key.split(":")
                    spec = next(s for s in self.task.scrolls if s.id == id_)
                    self._transition(State.SCROLL)
                    self.r["scroll_count"] += 1
                    effect = self.store.reserve(self.r, "scroll", key)
                    await self.call(self.browser.scroll(snap, spec, 1 if direction == "down" else -1))
                    self.store.returned(effect)
                    self._transition(State.WAIT_CONTENT)
                    await self.call(self.browser.settle(spec, snap))
                    self._transition(State.VERIFY_STEP)
                    after = await self.call(self.browser.snapshot())
                    try:
                        if after.doc_hash != snap.doc_hash:
                            raise Guard("navigation_during_scroll")
                        a, b = snap.data["scrolls"][id_], after.data["scrolls"].get(id_)
                        if b is None:
                            raise Guard("scroll_container_disappeared")
                        if a["node"] != b["node"]:
                            raise Guard("scroll_container_replaced")
                        progress_keys = ("x", "y", "width", "height", "extent_x", "extent_y", "content_hash")
                        progress = any(a[k] != b[k] for k in progress_keys) or snap.data["clicks"] != after.data["clicks"] or snap.data["forms"] != after.data["forms"]
                        stat = self.r["memory"]["scroll"].setdefault(id_, {"steps": 0, "up_stalls": 0, "down_stalls": 0})
                        stat["steps"] += 1
                        stat[direction + "_stalls"] = 0 if progress else stat[direction + "_stalls"] + 1
                        stat.update(position=b[spec.axis], extent=b["extent_" + spec.axis], content_hash=b["content_hash"])
                        self.store.verified(self.r)
                        self._remember(key, snap, after, progress=progress)
                    finally:
                        await after.close()
                else:
                    click_id = key.split(":", 1)[1]
                    spec = next(c for c in self.task.clicks if c.id == click_id)
                    self._transition(State.CLICK)
                    effect = self.store.reserve(self.r, "click", key)
                    await self.call(self.browser.click(snap, click_id, min(2500, int(self.remaining() * 1000))))
                    self.store.returned(effect)
                    self._transition(State.VERIFY_STEP)
                    await self.call(self.browser.wait_check(spec.postcondition, min(3000, int(self.remaining() * 1000))))
                    after = await self.call(self.browser.snapshot())
                    try:
                        self.r["memory"]["clicks"].append(click_id)
                        self.store.verified(self.r)
                        self._remember(key, snap, after)
                    finally:
                        await after.close()
                await snap.close(); snap = None
                self._transition(State.OBSERVE)
        except asyncio.CancelledError as exc:
            self._abort(exc)
            raise
        except Exception as exc:
            return self._abort(exc)
        finally:
            if snap is not None:
                await snap.close()

    async def refresh_form(self):
        self._load()
        if self.r["state"] != State.AWAITING_FORM:
            return self.result(refresh_rejected=True)
        snap = None
        try:
            snap = await self.call(self.browser.snapshot())
            form_id = self.r["handoff"]["form_id"]
            if not snap.data["forms"].get(form_id, {}).get("ready"):
                raise Guard("form_no_longer_ready")
            self._doc_memory(snap)
            self._handoff(snap, form_id)  # New nonce/document/expiry; ALL old grants are invalid.
            self._charge(); self.store.save(self.r, {"event": "handoff_refreshed"})
            return self.result()
        except Exception as exc:
            return self._abort(exc)
        finally:
            if snap:
                await snap.close()

    async def form_fill(self, token: dict, values: dict):
        self._load()
        snap = None
        consumed = False
        try:
            if not self.r.get("handoff"):
                raise Guard("no_pending_handoff")
            form = self.task.form(self.r["handoff"]["form_id"])
            validate_values(form, values)
            claims = verify_grant(self.public_key, token, self.r["handoff"], values, check_time=False)
            if self.store.grant_owner(claims["jti"]) == self.run_id:
                # No second mutation even if a response was lost. Reconcile interrupted effects.
                if State(self.r["state"]) not in STABLE:
                    return self._abort(Guard("interrupted_form_do_not_replay"))
                return self.result(idempotent_replay=True)
            if self.dry_run or not self.task.execution_approved or self.r["state"] != State.AWAITING_FORM:
                raise Guard("not_waiting_for_authorized_form")
            if not form.input_effects_reviewed:
                raise Guard("form_input_effects_not_reviewed")
            verify_grant(self.public_key, token, self.r["handoff"], values)
            if self.r["steps"] + len(form.fields) > self.task.max_steps:
                raise Guard("insufficient_budget_for_entire_form")
            snap = await self.call(self.browser.snapshot())
            initial = snap.data["forms"].get(form.id)
            if snap.doc_hash != self.r["handoff"]["doc_hash"] or not initial or not initial["ready"] or initial["signature"] != self.r["handoff"]["form_signature"]:
                raise Guard("form_or_document_changed_since_handoff")
            # Permission and exact current page have passed. Only now enter the write path.
            self._transition(State.VALIDATE_GRANT)
            self.store.consume_grant(self.r, claims["jti"])
            consumed = True
            expected_hashes = {k: v["value_hash"] for k, v in initial["fields"].items()}
            for f in form.fields:
                if time.time() >= claims["expires_at"]:
                    raise Guard("grant_expired_mid_form")
                current = await self.call(self.browser.snapshot())
                try:
                    current_form = current.data["forms"].get(form.id)
                    if current.doc_hash != snap.doc_hash or not current_form or current_form["identity"] != initial["identity"]:
                        raise Guard("form_schema_or_node_changed_mid_form")
                    if any(current_form["fields"][k]["value_hash"] != h for k, h in expected_hashes.items()):
                        raise Guard("unexpected_field_edit_mid_form")
                finally:
                    await current.close()
                effect = self.store.reserve(self.r, "codex_field", f"{form.id}:{f.id}")
                await self.call(self.browser.input(snap, form.id, f, values[f.id], min(3000, int(self.remaining() * 1000))))
                self.store.returned(effect)
                expected_hashes[f.id] = self.browser.value_hash(values[f.id])
                self.store.verified(self.r)
            self._transition(State.VERIFY_FORM)
            after = await self.call(self.browser.snapshot())
            try:
                completed = after.data["forms"].get(form.id)
                if after.doc_hash != snap.doc_hash or not completed or completed["identity"] != initial["identity"]:
                    raise Guard("form_replaced_after_input")
                if any(completed["fields"][k]["value_hash"] != h or not completed["fields"][k]["valid"] for k, h in expected_hashes.items()):
                    raise Guard("final_form_values_or_validity_failed")
                self.r["memory"]["forms"][form.id] = {"signature": completed["signature"], "fields": [f.id for f in form.fields]}
                self.r["form_result"] = {"form_id": form.id, "verified_field_ids": [f.id for f in form.fields], "submitted": False}
                self.r["expected_resume_doc"] = after.doc_hash
                self._remember(f"codex_fill:{form.id}", snap, after)
                self.r["grant_in_progress"] = False
            finally:
                await after.close()
            self._transition(State.READY)
            return self.result()
        except asyncio.CancelledError as exc:
            self._abort(exc)
            raise
        except Exception as exc:
            if consumed or self.r.get("grant_in_progress") or self.r.get("pending_effect"):
                return self._abort(exc)
            # A bad/stale/expired grant never spends itself and never writes a field.
            self._charge(); self.store.save(self.r, {"event": "grant_rejected", "error_type": type(exc).__name__})
            reason = str(exc) if isinstance(exc, Guard) else f"validation_failed:{type(exc).__name__}"
            return self.result(approval_rejected=True, rejection_reason=reason)
        finally:
            if snap:
                await snap.close()

    def cancel(self):
        self._load()
        if self.r.get("pending_effect") or self.r.get("grant_in_progress"):
            return self._abort(Guard("cancelled_with_possible_effect"))
        if State(self.r["state"]) in TERMINAL:
            return self.result()
        self._transition(State.CANCELLED, "cancelled_by_host")
        return self.result()
