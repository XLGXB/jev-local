"""Real Chromium and real CDP; model is a deterministic test double.
about:blank origin substitution is TEST-ONLY; production rejects that URL.
"""
import asyncio
import copy
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from playwright.async_api import async_playwright

from fastloop.authority import make_grant
from fastloop.browser import CDPBrowser
from fastloop.contracts import Guard, State, Task, digest, origin as real_origin
from fastloop.engine import Machine
from fastloop.service import page_target_id
from fastloop.store import Store

ROOT = Path(__file__).resolve().parent  # package-scoped imported fixtures
SITE = "http://127.0.0.1:8089"


def task():
    d = json.loads((ROOT / "examples/registry.json").read_text(encoding="utf-8"))["tasks"]["demo_profile"]
    d.update(execution_approved=True, expected_runtime_fingerprint="a" * 64, segment_ms=20000,
             max_active_ms=80000)
    d["forms"][0]["input_effects_reviewed"] = True
    d["scrolls"][0].update(quiet_ms=40, edge_wait_ms=120, settle_timeout_ms=800)
    return Task.model_validate(d)


class Predictor:
    def __init__(self): self.states = []
    async def __call__(self, state, criteria, obs_id, timeout, task, dry_run):
        self.states.append(copy.deepcopy(state))
        for prefix in ("click:", "form:", "scroll:"):
            for key in criteria:
                if key.startswith(prefix) and not key.endswith(":up"):
                    return key
        return "needs_host"


async def setup(endpoint, tmp_path, monkeypatch, custom=None, html=None):
    monkeypatch.setattr("fastloop.browser.origin", lambda u: SITE if u == "about:blank" else real_origin(u))
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(endpoint)
    page = await browser.contexts[0].new_page(); await page.set_viewport_size({"width": 1100, "height": 1000})
    await page.set_content(html or (ROOT / "examples/demo.html").read_text(encoding="utf-8"))
    t = custom or task(); store = Store(tmp_path); keys = Ed25519PrivateKey.generate(); pred = Predictor()
    b = await CDPBrowser.attach(page, t, store.state_key)
    tid = await page_target_id(page)
    kwargs = dict(task=t, predict=pred, store=store, public_key=keys.public_key(),
                  run_id="run-1", target_id=tid, target_key=digest([endpoint, tid]), task_id="demo", project_id="test")
    machine = Machine(browser=b, **kwargs)
    return pw, page, b, machine, keys, kwargs


async def cleanup(pw, page, adapter, store):
    await adapter.close(); store.close(); await page.close(); await pw.stop()


def values():
    return {"display_name": "PRIVATE_DEMO_NAME_27", "note": "PRIVATE_DEMO_NOTE_81", "language": "en", "notify": True}


def test_scroll_handoff_fill_resume_persistent(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            r = await m.start()
            assert r["state"] == State.AWAITING_FORM, r
            assert r["scrolls"] >= 1 and r["steps"] >= 2
            assert not await page.locator("#display_name").input_value()
            before_steps = r["steps"]
            grant = make_grant(keys, r["handoff"], values())
            # Reconnect both the browser adapter and the store: no in-memory-only receipt.
            await b.close(); kw["store"].close()
            kw["store"] = Store(tmp_path)
            b = await CDPBrowser.attach(page, kw["task"], kw["store"].state_key)
            m = Machine(browser=b, **kw)
            filled = await m.form_fill(grant, values())
            assert filled["state"] == State.READY, filled
            assert filled["steps"] == before_steps + 4
            assert filled["form_result"]["submitted"] is False
            repeated = await m.form_fill(grant, values())
            assert repeated["idempotent_replay"] and repeated["steps"] == filled["steps"]
            assert await page.evaluate("window.submissions") == 0
            done = await m.resume()
            assert done["state"] == State.NEEDS_VERIFICATION, done
            assert done["steps"] == filled["steps"] + 1
            calls = len(kw["predict"].states)
            again = await m.start()
            assert again["idempotent_replay"] and len(kw["predict"].states) == calls
            last_state = kw["predict"].states[-1]
            assert "profile" in last_state["memory"]["filled_form_ids"]
            assert any(e["action"] == "codex_fill:profile" for e in last_state["memory"]["recent_verified_actions"])
            assert "PRIVATE_DEMO" not in json.dumps(kw["store"].get("run-1"))
            assert "PRIVATE_DEMO" not in json.dumps(kw["predict"].states)
            for table in ("events", "observations"):
                for row in kw["store"].db.execute(f"SELECT body FROM {table}"):
                    assert "PRIVATE_DEMO" not in row[0]
            assert await page.evaluate("window.submissions") == 0
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["wrong_value", "expired", "replaced_node", "user_edit", "wrong_document", "approval_boolean"])
def test_form_rejection_before_write(cdp_endpoint, tmp_path, monkeypatch, case):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            r = await m.start(); assert r["state"] == State.AWAITING_FORM, r
            v = values(); grant = make_grant(keys, r["handoff"], v)
            if case == "wrong_value": v["note"] = "changed after approval"
            if case == "expired":
                import time
                grant = make_grant(keys, r["handoff"], v, now=time.time()-200, ttl=30)
            if case == "replaced_node": await page.locator("#display_name").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            if case == "user_edit": await page.locator("#display_name").fill("MANUAL")
            if case == "wrong_document":
                # A new main-frame document invalidates loaderId even though the form looks identical.
                await page.goto("about:blank")
                await page.set_content((ROOT / "examples/demo.html").read_text(encoding="utf-8"))
                await page.locator("#open").click()
            if case == "approval_boolean": grant = {"approved": True}
            before = await page.locator("#display_name").input_value()
            out = await m.form_fill(grant, v)
            assert out["approval_rejected"] and out["state"] == State.AWAITING_FORM, out
            assert await page.locator("#display_name").input_value() == before
            assert kw["store"].db.execute("SELECT count(*) FROM grants").fetchone()[0] == 0
            assert await page.evaluate("window.submissions") == 0
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_partial_fill_unknown_is_not_replayed(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            r = await m.start(); grant = make_grant(keys, r["handoff"], values())
            original = b.input; calls = []
            async def interrupted(*args):
                calls.append(1); await original(*args)
                raise RuntimeError("do not leak PRIVATE_DEMO_NAME_27")
            b.input = interrupted
            out = await m.form_fill(grant, values())
            assert out["state"] == State.OUTCOME_UNKNOWN, out
            assert await page.locator("#display_name").input_value() == values()["display_name"]
            assert not await page.locator("#note").input_value()
            again = await m.form_fill(grant, values())
            assert again["idempotent_replay"] and len(calls) == 1
            other = dict(kw, run_id="run-other")
            with pytest.raises(Guard, match="target_leased"):
                await Machine(browser=b, **other).start()
            assert "PRIVATE_DEMO" not in json.dumps(kw["store"].get("run-1"))
            assert kw["store"].db.execute("SELECT count(*) FROM leases").fetchone()[0] == 1
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_form_pause_keeps_lease_and_cancel_releases(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            assert (await m.start())["state"] == State.AWAITING_FORM
            with pytest.raises(Guard, match="target_leased"):
                await Machine(browser=b, **dict(kw, run_id="other")).start()
            assert m.cancel()["state"] == State.CANCELLED
            assert kw["store"].db.execute("SELECT count(*) FROM leases").fetchone()[0] == 0
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_refresh_invalidates_previous_grant(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            a = await m.start(); old = make_grant(keys, a["handoff"], values())
            bresult = await m.refresh_form()
            assert bresult["handoff"]["handoff_id"] != a["handoff"]["handoff_id"]
            assert bresult["steps"] == a["steps"]
            assert (await m.form_fill(old, values()))["approval_rejected"]
            good = make_grant(keys, bresult["handoff"], values())
            assert (await m.form_fill(good, values()))["state"] == State.READY
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["stale_click", "dry_run", "forbidden_submit", "model_fake_fill", "global_budget"])
def test_click_loop_guards(cdp_endpoint, tmp_path, monkeypatch, case):
    async def scenario():
        t = task()
        if case == "global_budget": t.max_steps = 1
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch, custom=t)
        try:
            await page.locator("#open").scroll_into_view_if_needed()
            if case == "dry_run": m = Machine(browser=b, **dict(kw, dry_run=True))
            if case == "forbidden_submit": await page.locator("#open").evaluate("e=>e.type='submit'")
            original = kw["predict"]
            async def predict(state, criteria, *rest):
                if case == "stale_click": await page.locator("#open").evaluate("e=>e.replaceWith(e.cloneNode(true))")
                if case == "model_fake_fill": return "fill:display_name"
                return await original(state, criteria, *rest)
            m.predict = predict
            out = await m.start()
            if case == "dry_run": assert out["state"] == State.DRY_RUN and out["steps"] == 0
            elif case == "global_budget": assert out["state"] == State.NEEDS_HOST and out["steps"] == 1
            else: assert out["state"] == State.NEEDS_HOST and out["steps"] == 0, out
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def scroll_only_task():
    d = task().model_dump(); d.update(clicks=[], forms=[], verifier={"selector": "#done"})
    d["scrolls"][0].update(selector="#list", item_selector=".row", max_steps=5, edge_wait_ms=100)
    return Task.model_validate(d)


@pytest.mark.parametrize("case", ["bottom", "lazy", "virtual", "busy_timeout", "horizontal"])
def test_scroll_state_and_progress(cdp_endpoint, tmp_path, monkeypatch, case):
    async def scenario():
        t = scroll_only_task()
        html = '''<style>#list{width:300px;height:100px;overflow:auto}.row{height:100px}</style>
        <div id="busy" hidden>loading</div><div id="list"><div class="row" data-key="a">a</div><div class="row" data-key="b">b</div></div>'''
        if case == "busy_timeout": t.scrolls[0].busy_selector = "#busy"
        if case == "horizontal":
            t.scrolls[0].axis = "x"
            html += "<style>.row{width:1000px}</style>"
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch, custom=t, html=html)
        try:
            if case == "bottom": await page.locator("#list").evaluate("e=>e.scrollTop=e.scrollHeight")
            if case == "lazy":
                await page.locator("#list").evaluate("""e=>e.addEventListener('scroll',()=>{
                    if(e.dataset.loaded)return; e.dataset.loaded='1';
                    setTimeout(()=>{let n=document.createElement('div');n.className='row';n.dataset.key='c';n.textContent='new';e.appendChild(n)},60)
                })""")
            if case == "virtual":
                await page.locator("#list").evaluate("""e=>e.addEventListener('scroll',()=>{
                    if(e.dataset.loaded)return; e.dataset.loaded='1';
                    e.querySelector('.row').dataset.key='replacement';e.querySelector('.row').textContent='virtual-new'; e.scrollTop=0;
                })""")
            if case == "busy_timeout": await page.locator("#busy").evaluate("e=>e.hidden=false")
            original = kw["predict"]; calls = []
            async def predict(*args):
                calls.append(1)
                if case != "bottom" and len(calls) > 1: return "needs_host"
                return await original(*args)
            m.predict = predict
            out = await m.start()
            memory = kw["store"].get("run-1")["memory"]
            if case == "busy_timeout":
                assert out["state"] == State.OUTCOME_UNKNOWN, out
            elif case == "bottom":
                assert out["state"] == State.NEEDS_HOST
                assert memory["scroll"]["list"]["down_stalls"] == 2 and out["scrolls"] == 2
            else:
                assert out["state"] == State.NEEDS_HOST and out["scrolls"] == 1, out
                assert memory["recent"][-1]["progress"] is True
                if case == "lazy": assert memory["scroll"]["list"]["extent"] == 300
                if case == "virtual": assert memory["scroll"]["list"]["position"] == 0
                if case == "horizontal": assert memory["scroll"]["list"]["position"] > 0
            assert await page.evaluate("window.scrollY") == 0  # precise nested container, not body.
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_verified_form_edit_before_resume_stops(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            r = await m.start(); grant = make_grant(keys, r["handoff"], values())
            assert (await m.form_fill(grant, values()))["state"] == State.READY
            await page.locator("#note").fill("edited outside executor")
            out = await m.resume()
            assert out["state"] == State.NEEDS_HOST
            assert out["reason"] == "verified_form_changed_requires_host_review"
            assert not await page.locator("#ready").is_visible()
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_cancelled_input_is_unknown(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch)
        try:
            r = await m.start(); grant = make_grant(keys, r["handoff"], values())
            original = b.input
            async def cancelled(*args):
                await original(*args)
                raise asyncio.CancelledError()
            b.input = cancelled
            with pytest.raises(asyncio.CancelledError): await m.form_fill(grant, values())
            assert kw["store"].get("run-1")["state"] == State.OUTCOME_UNKNOWN
            assert kw["store"].db.execute("SELECT count(*) FROM leases").fetchone()[0] == 1
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())


def test_document_root_scroll(cdp_endpoint, tmp_path, monkeypatch):
    async def scenario():
        t = scroll_only_task(); t.scrolls[0].selector = "@document"
        html = '<!doctype html><style>.row{height:1000px}</style><div class="row" data-key="1">first</div><div class="row" data-key="2">second</div><div class="row" data-key="3">third</div>'
        pw, page, b, m, keys, kw = await setup(cdp_endpoint, tmp_path, monkeypatch, custom=t, html=html)
        try:
            original = kw["predict"]; calls = []
            async def predict(*args):
                calls.append(1)
                return await original(*args) if len(calls) == 1 else "needs_host"
            m.predict = predict
            out = await m.start()
            assert out["scrolls"] == 1 and out["state"] == State.NEEDS_HOST, out
            assert await page.evaluate("window.scrollY") > 0
            assert kw["store"].get("run-1")["memory"]["recent"][-1]["progress"] is True
        finally:
            await cleanup(pw, page, b, kw["store"])
    asyncio.run(scenario())
