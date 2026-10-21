"""Top-frame DOM adapter. CSS selectors come ONLY from a reviewed local task profile.
Snapshots contain hashes/metadata, not field values or unrestricted page text.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

from .contracts import Check, FieldSpec, FormSpec, Guard, ScrollSpec, Task, canonical, digest, origin

META_JS = """e => {
 const r=e.getBoundingClientRect(), s=getComputedStyle(e);
 let l=Math.max(0,r.left),t=Math.max(0,r.top),rr=Math.min(innerWidth,r.right),bb=Math.min(innerHeight,r.bottom);
 for(let p=e.parentElement;p;p=p.parentElement){
   const q=getComputedStyle(p),z=p.getBoundingClientRect();
   if (/(auto|scroll|hidden|clip)/.test(q.overflowX)){l=Math.max(l,z.left);rr=Math.min(rr,z.right)}
   if (/(auto|scroll|hidden|clip)/.test(q.overflowY)){t=Math.max(t,z.top);bb=Math.min(bb,z.bottom)}
 }
 const visible=!!(e.isConnected && r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none');
 const viewport=visible && rr>l+1 && bb>t+1;
 const hit=viewport?document.elementFromPoint((l+rr)/2,(t+bb)/2):null;
 return {tag:e.tagName,type:e.type||'',name:e.getAttribute('name')||'',
    autocomplete:e.getAttribute('autocomplete')||'',enabled:!e.disabled&&!e.readOnly,
    visible,viewport,hit:!!hit&&(hit===e||e.contains(hit)),connected:e.isConnected,
    rect:[r.x,r.y,r.width,r.height].map(x=>Math.round(x)),
    href:e.getAttribute('href'),target:e.getAttribute('target'),
    method:e.getAttribute('method'),action:e.getAttribute('action'),
    label:e.getAttribute('aria-label')||''};
}"""


@dataclass
class Snapshot:
    data: dict
    handles: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)

    @property
    def hash(self):
        return digest(self.data)

    @property
    def doc_hash(self):
        return digest(self.data["document"])

    async def close(self):
        for h in list(self.handles.values()) + list(self.fields.values()):
            try:
                await h.dispose()
            except Exception:
                pass
        self.handles.clear(); self.fields.clear()


class CDPBrowser:
    def __init__(self, page, session, task: Task, state_key: bytes):
        self.page, self.session, self.task, self.state_key = page, session, task, state_key
        self.unexpected = False
        self._popup = lambda _: setattr(self, "unexpected", True)
        self._dialog_tasks = set()
        def dialog(d):
            self.unexpected = True
            async def dismiss():
                try:
                    await d.dismiss()
                except Exception:
                    pass
            t = asyncio.create_task(dismiss()); self._dialog_tasks.add(t)
            t.add_done_callback(self._dialog_tasks.discard)
        self._dialog = dialog
        page.on("popup", self._popup); page.on("dialog", self._dialog)

    @classmethod
    async def attach(cls, page, task, state_key):
        s = await page.context.new_cdp_session(page)
        await s.send("DOM.enable")
        return cls(page, s, task, state_key)

    async def close(self):
        self.page.remove_listener("popup", self._popup)
        self.page.remove_listener("dialog", self._dialog)
        if self._dialog_tasks:
            await asyncio.gather(*self._dialog_tasks, return_exceptions=True)
        await self.session.detach()

    def assert_clean(self):
        if self.unexpected:
            raise Guard("unexpected_popup_or_dialog")
        if self.page.is_closed():
            raise Guard("target_closed")

    async def document(self) -> dict:
        self.assert_clean()
        frame = (await self.session.send("Page.getFrameTree"))["frameTree"]["frame"]
        url = self.page.url
        o = origin(url)
        if o not in self.task.allowed_origins:
            raise Guard("origin_outside_permission")
        if not frame.get("loaderId"):
            raise Guard("missing_document_loader_id")
        return {"frame_id": frame["id"], "loader_id": frame["loaderId"], "origin": o, "url_hash": digest(url)}

    async def exact(self, selector: str):
        loc = self.page.locator(selector)
        n = await loc.count()
        if n > 1:
            raise Guard("ambiguous_selector")
        return await loc.element_handle() if n else None

    async def node(self, selector: str) -> int:
        # Persistent across reconnects while the same DOM node/document exists.
        root = (await self.session.send("DOM.getDocument", {"depth": 0}))["root"]["nodeId"]
        node = (await self.session.send("DOM.querySelector", {"nodeId": root, "selector": selector}))["nodeId"]
        if not node:
            raise Guard("node_disappeared")
        return (await self.session.send("DOM.describeNode", {"nodeId": node}))["node"]["backendNodeId"]

    def value_hash(self, value):
        return hmac.new(self.state_key, canonical(value), hashlib.sha256).hexdigest()

    async def check(self, check: Check) -> bool:
        self.assert_clean()
        h = await self.exact(check.selector)
        if h is None:
            return False
        try:
            return await h.is_visible() and (check.text is None or (await h.inner_text()).strip() == check.text)
        finally:
            await h.dispose()

    async def _field(self, spec: FieldSpec, form_handle):
        h = await self.exact(spec.selector)
        if h is None:
            raise Guard("form_field_missing")
        try:
            if not await form_handle.evaluate("(form,e)=>form.contains(e)", h):
                raise Guard("field_outside_form")
            m = await h.evaluate(META_JS)
            m["node"] = await self.node(spec.selector)
            ac = m["autocomplete"].lower().split()
            if any(t in {"current-password", "new-password", "one-time-code"} or t.startswith("cc-") for t in ac):
                raise Guard("credential_field_forbidden")
            wanted = {"textarea": "TEXTAREA", "select": "SELECT"}.get(spec.kind, "INPUT")
            expected_type = "checkbox" if spec.kind == "checkbox" else spec.input_type
            if m["tag"] != wanted or (wanted == "INPUT" and m["type"] != expected_type):
                raise Guard("field_type_changed")
            if spec.kind == "checkbox":
                value = await h.is_checked()
            else:
                value = await h.input_value()
            if spec.kind == "select":
                options = await h.evaluate("e=>Array.from(e.options).map(o=>({value:o.value,disabled:o.disabled}))")
                if not set(spec.choices) <= {o["value"] for o in options if not o["disabled"]}:
                    raise Guard("select_options_changed")
                m["options_hash"] = digest(options)
            m["value_hash"] = self.value_hash(value)
            m["valid"] = await h.evaluate("e=>e.validity ? e.validity.valid : false")
            return h, m
        except BaseException:
            await h.dispose()
            raise

    async def _form(self, form: FormSpec, snap: Snapshot):
        h = await self.exact(form.selector)
        if h is None:
            return
        keep = False
        try:
            root = await h.evaluate(META_JS)
            if not root["visible"]:
                return
            if root["tag"] != "FORM":
                raise Guard("profile_form_must_be_form_element")
            if root["target"] not in (None, "", "_self") or origin(urljoin(self.page.url, root["action"] or self.page.url)) not in self.task.allowed_origins:
                raise Guard("form_target_outside_permission")
            root["node"] = await self.node(form.selector)
            fields = {}
            for f in form.fields:
                handle, meta = await self._field(f, h)
                fields[f.id] = meta
                snap.fields[f"{form.id}:{f.id}"] = handle
            stable_root = {k: root[k] for k in ("tag", "node", "action", "method", "target")}
            stable_fields = {k: {n: v for n, v in m.items() if n not in {"rect", "viewport", "hit"}} for k, m in fields.items()}
            identity = {k: {n: v for n, v in m.items() if n not in {"value_hash", "valid", "rect", "viewport", "hit"}} for k, m in fields.items()}
            snap.data["forms"][form.id] = {
                "signature": digest({"root": stable_root, "fields": stable_fields}),
                "identity": digest({"root": stable_root, "fields": identity}),
                "ready": root["viewport"] and all(m["visible"] and m["enabled"] for m in fields.values()),
                "fields": fields,
            }
            snap.handles[f"form:{form.id}"] = h; keep = True
        finally:
            if not keep:
                await h.dispose()

    async def scroll_sample(self, spec: ScrollSpec, h=None):
        own = h is None
        if h is None:
            if spec.selector == "@document":
                h = (await self.page.evaluate_handle("document.scrollingElement")).as_element()
            else:
                h = await self.exact(spec.selector)
        if h is None:
            return None, None
        try:
            metrics = await h.evaluate("""(e,a)=>{
                if(!e.isConnected)throw new Error('detached');
                const r=e.getBoundingClientRect();
                const items=a.selector?Array.from(e.querySelectorAll(a.selector)).filter(n=>{
                    const q=n.getBoundingClientRect();return q.bottom>Math.max(0,r.top)&&q.top<Math.min(innerHeight,r.bottom)&&q.right>0&&q.left<innerWidth
                }).map(n=>[n.getAttribute(a.attr),n.textContent]):[];
                return {x:Math.round(e.scrollLeft),y:Math.round(e.scrollTop),
                    width:e.clientWidth,height:e.clientHeight,extent_x:e.scrollWidth,extent_y:e.scrollHeight,
                    items,visible:e===document.scrollingElement || (r.width>0&&r.height>0&&r.bottom>0&&r.top<innerHeight)};
            }""", {"selector": spec.item_selector, "attr": spec.item_key_attr})
            items = metrics.pop("items")
            if len(items) > 300:
                raise Guard("too_many_visible_scroll_items")
            metrics["content_hash"] = digest(items)
            metrics["busy"] = bool(spec.busy_selector and await self.check(Check(selector=spec.busy_selector)))
            selector = spec.selector if spec.selector != "@document" else (await h.evaluate("e=>e.tagName.toLowerCase()"))
            metrics["node"] = await self.node(selector)
            return metrics, h
        except BaseException:
            if own:
                await h.dispose()
            raise

    async def snapshot(self) -> Snapshot:
        doc = await self.document()
        snap = Snapshot({"document": doc, "clicks": {}, "scrolls": {}, "forms": {}, "verified": False})
        try:
            for c in self.task.clicks:
                h = await self.exact(c.selector)
                if h is None:
                    continue
                keep = False
                try:
                    m = await h.evaluate(META_JS)
                    if not m["visible"] or not m["enabled"]:
                        continue
                    if (await h.inner_text()).strip() != c.expected_text:
                        raise Guard("click_label_changed")
                    if m["tag"] == "BUTTON":
                        if m["type"] != "button":
                            raise Guard("submit_click_forbidden")
                    elif m["tag"] == "A":
                        href = urljoin(self.page.url, m["href"] or "")
                        if m["target"] not in (None, "", "_self") or origin(href) not in self.task.allowed_origins:
                            raise Guard("link_outside_permission")
                        # Do not expose query strings to the model/memory.
                        m["href"] = digest(href)
                    else:
                        raise Guard("unsupported_click_element")
                    m["node"] = await self.node(c.selector)
                    snap.data["clicks"][c.id] = m
                    snap.handles[f"click:{c.id}"] = h; keep = True
                finally:
                    if not keep:
                        await h.dispose()
            for f in self.task.forms:
                await self._form(f, snap)
            for s in self.task.scrolls:
                metrics, h = await self.scroll_sample(s)
                if metrics is not None:
                    snap.data["scrolls"][s.id] = metrics
                    snap.handles[f"scroll:{s.id}"] = h
            snap.data["verified"] = await self.check(self.task.verifier)
            if doc != await self.document():
                raise Guard("document_changed_during_observation")
            return snap
        except BaseException:
            await snap.close()
            raise

    async def click(self, snap: Snapshot, click_id: str, timeout_ms: int):
        # Intentionally keep the original short-lived handle; NEVER re-target after detach.
        await snap.handles[f"click:{click_id}"].click(timeout=timeout_ms, force=False)

    async def scroll(self, snap: Snapshot, spec: ScrollSpec, direction: int):
        h = snap.handles[f"scroll:{spec.id}"]
        m = snap.data["scrolls"][spec.id]
        amount = max(1, round(m["height" if spec.axis == "y" else "width"] * spec.step_ratio)) * direction
        # Precise container mutation, never model-provided JS. Scroll events/IntersectionObserver
        # work normally. Wheel-only custom widgets need a separately audited adapter.
        await h.evaluate("(e,a)=>{if(!e.isConnected)throw new Error('detached'); e.scrollBy({left:a.x,top:a.y,behavior:'instant'})}",
                         {"x": amount if spec.axis == "x" else 0, "y": amount if spec.axis == "y" else 0})

    async def settle(self, spec: ScrollSpec, snap: Snapshot):
        h = snap.handles[f"scroll:{spec.id}"]
        start = last_change = time.monotonic()
        last = None
        while (time.monotonic() - start) * 1000 < spec.settle_timeout_ms:
            self.assert_clean()
            if digest(await self.document()) != snap.doc_hash:
                raise Guard("navigation_during_scroll")
            current, _ = await self.scroll_sample(spec, h)
            code = digest(current)
            now = time.monotonic()
            if code != last:
                last_change = now; last = code
            axis = spec.axis
            pos, extent, size = current[axis], current["extent_" + axis], current["height" if axis == "y" else "width"]
            at_edge = pos <= 1 or pos + size >= extent - 1
            min_wait = spec.edge_wait_ms if at_edge else spec.quiet_ms
            if not current["busy"] and (now - last_change) * 1000 >= spec.quiet_ms and (now - start) * 1000 >= min_wait:
                return current
            await asyncio.sleep(0.04)  # Condition polling, not a fixed post-action delay.
        raise Guard("scroll_did_not_stabilize")

    async def wait_check(self, check: Check, timeout_ms: int):
        until = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < until:
            self.assert_clean()
            await self.document()
            if await self.check(check):
                return
            await asyncio.sleep(0.04)
        raise Guard("postcondition_not_observed")

    async def input(self, snap: Snapshot, form_id: str, field: FieldSpec, value, timeout_ms: int):
        h = snap.fields[f"{form_id}:{field.id}"]
        await h.scroll_into_view_if_needed(timeout=timeout_ms)
        if field.kind == "checkbox":
            await h.set_checked(value, timeout=timeout_ms, force=False)
        elif field.kind == "select":
            await h.select_option(value=value, timeout=timeout_ms)
        else:
            await h.fill(value, timeout=timeout_ms)
        # No Enter, submit(), requestSubmit(), click submit or extra blur event.
        actual = await h.is_checked() if field.kind == "checkbox" else await h.input_value()
        if actual != value or not await h.evaluate("e=>e.isConnected && e.validity.valid"):
            raise Guard("input_postcondition_failed")
