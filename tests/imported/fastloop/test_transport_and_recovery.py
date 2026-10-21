import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fastloop.contracts import Guard, State, Task
from fastloop.engine import Machine
from fastloop.model_client import HTTPDecision
from fastloop.store import Store

ROOT = Path(__file__).resolve().parent  # package-scoped imported fixtures


def task():
    d = json.loads((ROOT / "examples/registry.json").read_text(encoding="utf-8"))["tasks"]["demo_profile"]
    d.update(execution_approved=True, expected_runtime_fingerprint="a" * 64)
    return Task.model_validate(d)


@pytest.mark.parametrize("stale", [False, True])
def test_v1_http_wire_contract(stale):
    async def scenario():
        c = HTTPDecision("http://127.0.0.1:8787", "x" * 40)
        await c.client.aclose()
        def handler(request):
            p = json.loads(request.content)
            assert request.url.path == "/v1/decide"
            assert request.headers["Authorization"] == "Bearer " + "x" * 40
            assert set(p) == {"request_id", "observation_id", "state", "questions"}
            assert p["questions"]["next_action"]["type"] == "choice"
            assert "label" not in p["questions"]["next_action"]
            return httpx.Response(200, json={"request_id": p["request_id"], "observation_id": "wrong" if stale else p["observation_id"],
                "runtime_fingerprint": "a" * 64, "answers": {"next_action": {
                "selected_key": "scroll:list:down", "p_selected": .999, "calibrated": True,
                "probabilities": {"scroll:list:down": .999, "needs_host": .001}}}})
        c.client = httpx.AsyncClient(base_url="http://127.0.0.1:8787", transport=httpx.MockTransport(handler),
                                   headers={"Authorization": "Bearer " + "x" * 40})
        try:
            call = c({}, {"scroll:list:down": "scroll", "needs_host": "host"}, "obs", 1, task(), False)
            if stale:
                with pytest.raises(Guard, match="stale_model_response"): await call
            else:
                assert await call == "scroll:list:down"
        finally:
            await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["before_dispatch", "after_reserve", "grant_in_progress"])
def test_restart_does_not_replay_effects(tmp_path, case):
    async def scenario():
        store = Store(tmp_path)
        kw = dict(task=task(), browser=None, predict=None, store=store, public_key=Ed25519PrivateKey.generate().public_key(),
                  run_id="crash", target_id="target", target_key="target-key")
        m = Machine(**kw)
        async def simulate_process_exit():
            m._transition(State.OBSERVE)
            if case == "after_reserve": store.reserve(m.r, "click", "open")
            if case == "grant_in_progress":
                m.r["grant_in_progress"] = True; store.save(m.r)
            raise SystemExit("test-only simulated crash")
        m._drive = simulate_process_exit
        with pytest.raises(SystemExit): await m.start()
        before = store.get("crash")["steps"]
        store.close(); kw["store"] = Store(tmp_path)
        try:
            r = await Machine(**kw).start()
            expected = State.NEEDS_HOST if case == "before_dispatch" else State.OUTCOME_UNKNOWN
            assert r["state"] == expected and r["steps"] == before
            retained = kw["store"].db.execute("SELECT count(*) FROM leases").fetchone()[0]
            assert retained == (0 if case == "before_dispatch" else 1)
        finally:
            kw["store"].close()
    asyncio.run(scenario())


def test_scope_change_cannot_resume_old_run(tmp_path):
    async def scenario():
        store = Store(tmp_path); key = Ed25519PrivateKey.generate().public_key()
        base = dict(task=task(), browser=None, predict=None, store=store, public_key=key,
                    run_id="same", target_id="target", target_key="target-key", project_id="A")
        m = Machine(**base)
        async def paused_fixture():
            m.r["state"] = State.AWAITING_FORM; store.save(m.r)
            return m.result()
        m._drive = paused_fixture
        try:
            await m.start()
            with pytest.raises(Guard, match="run_scope_mismatch"):
                await Machine(**dict(base, project_id="B")).start()
            d = task().model_dump(); d["goal"] = "different task"
            with pytest.raises(Guard, match="run_scope_mismatch"):
                await Machine(**dict(base, task=Task.model_validate(d))).start()
        finally:
            store.close()
    asyncio.run(scenario())


def test_mcp_tools_optional():
    pytest.importorskip("mcp", reason="MCP SDK unavailable in this offline runtime")
    from fastloop.mcp_server import mcp
    async def scenario():
        tools = {t.name: t for t in await mcp.list_tools()}
        assert set(tools) == {"browser_targets", "browser_status", "browser_run", "browser_form_fill",
                              "browser_refresh_form", "browser_resume", "browser_cancel"}
        assert tools["browser_form_fill"].annotations.readOnlyHint is False
        assert not any("approve" in name or "sign" in name for name in tools)
    asyncio.run(scenario())
