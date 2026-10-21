"""Real isolated Chromium dispatch, process death, and service recovery.

Only the predictor and qualification receipt are test doubles. The click,
durable reservation, service entry point, browser, and fixture HTTP effect
are real. This does not qualify a native model or authorize a real site.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from playwright.sync_api import sync_playwright
import pytest

from fastloop.contracts import State, digest
from fastloop.store import Store


RUN_ID = "browser-crash-run"
TASK_ID = "local-click"
PROJECT_ID = "browser-crash-fixture"
ROOT = Path(__file__).resolve().parents[2]


class _FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _FixtureHandler)
        self.effects = 0
        self.effects_lock = threading.Lock()

    def count(self):
        with self.effects_lock:
            return self.effects


class _FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = (b"<!doctype html><title>Local crash fixture</title>"
                b"<button id='act' type='button' "
                b"onclick=\"fetch('/effect',{method:'POST'})\">Act</button>"
                b"<span id='done' hidden>done</span>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/effect":
            self.send_error(404)
            return
        with self.server.effects_lock:
            self.server.effects += 1
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_):
        pass


def _unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait(predicate, timeout, message, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError(f"{message}; child exit={process.returncode}")
        time.sleep(0.05)
    raise AssertionError(message)


def _endpoint_ready(endpoint):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(endpoint + "/json/version", timeout=0.25) as response:
            return bool(json.load(response).get("Browser"))
    except (OSError, ValueError):
        return False


def _registry(root, endpoint, site):
    public = Ed25519PrivateKey.generate().public_key()
    key = root / "approval-public.pem"
    key.write_bytes(public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    registry = root / "registry.json"
    registry.write_text(json.dumps({
        "schema_version": 2,
        "project_id": PROJECT_ID,
        "controller": "fastloop",
        "control_mode": "isolated-cdp",
        "cdp": endpoint,
        "decision_service": "http://127.0.0.1:1",
        "state_dir": str(root / "state"),
        "approval_public_key": str(key),
        "tasks": {TASK_ID: {
            "goal": "Click the one local fixture button",
            "allowed_origins": [site],
            "execution_approved": True,
            "expected_runtime_fingerprint": "a" * 64,
            "clicks": [{
                "id": "act", "selector": "#act", "label": "Act",
                "expected_text": "Act", "postcondition": {"selector": "#done"},
            }],
            "verifier": {"selector": "#done"},
            "max_steps": 2,
            "segment_ms": 30000,
            "max_active_ms": 60000,
        }},
    }), encoding="utf-8")
    return registry


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH", ""))))
    return env


def _flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _snapshot(registry, run_id=RUN_ID):
    config = json.loads(registry.read_text(encoding="utf-8"))
    store = Store(config["state_dir"])
    try:
        row = store.get(run_id)
        return {
            "record": row,
            "effects": store.db.execute(
                "SELECT id, kind, status FROM effects WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall(),
            "leases": store.db.execute(
                "SELECT target, run_id FROM leases WHERE run_id=?", (run_id,)
            ).fetchall(),
            "grants": store.db.execute("SELECT count(*) FROM grants").fetchone()[0],
        }
    finally:
        store.close()


def _invoke(registry, operation, run_id=RUN_ID, target_id=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "invoke", str(registry),
           operation, run_id]
    if target_id is not None:
        cmd.append(target_id)
    done = subprocess.run(cmd, env=_env(), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=12,
                          creationflags=_flags())
    assert done.returncode == 0, (done.returncode, done.stderr)
    return json.loads(done.stdout)


def _review_close(registry):
    output = registry.parent / "review-close.json"
    done = subprocess.run(
        [sys.executable, "-m", "fastloop.service", "--registry", str(registry),
         "--out", str(output), "review-close", "--run", RUN_ID],
        env=_env(), input="CLOSE_WITHOUT_CREDIT\n", capture_output=True,
        text=True, timeout=12, creationflags=_flags(),
    )
    assert done.returncode == 0, (done.returncode, done.stderr)
    return json.loads(output.read_text(encoding="utf-8"))


def _invoke_child(registry, operation, run_id, target_id):
    # The two doubles are confined to the isolated child process. Neither is
    # evidence of production qualification, model quality, or user approval.
    import fastloop.service as service
    from fastloop.browser import CDPBrowser

    class _Predictor:
        async def __call__(self, state, criteria, obs_id, timeout, task, dry_run):
            return "click:act" if "click:act" in criteria else "needs_host"

        async def close(self):
            pass

    service.require_qualification = lambda cfg, task: None
    service.HTTPDecision = lambda *args: _Predictor()
    if operation == "dispatch":
        # Admission gate: the parent binds this process to its Windows Job
        # before this child starts Playwright's driver subprocess.
        gate = registry.parent / "worker-admitted.ready"
        _wait(gate.is_file, 10, "worker was not admitted to its Job")
        original_click = CDPBrowser.click
        marker = registry.parent / "click-returned.ready"

        async def click_then_pause(self, snap, click_id, timeout_ms):
            await original_click(self, snap, click_id, timeout_ms)
            marker.write_text("real CDPBrowser.click returned", encoding="ascii")
            while True:
                await asyncio.sleep(1)

        CDPBrowser.click = click_then_pause
        operation = "start"
    kwargs = {"task_id": TASK_ID, "target_id": target_id, "dry_run": False} if operation == "start" else {}
    result = asyncio.run(service.invoke(registry, operation, run_id, **kwargs))
    print(json.dumps(result), flush=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows hidden-child crash recovery fixture")
def _close_owned_browser(connection):
    if connection.is_connected():
        session = connection.new_browser_cdp_session()
        session.send("Browser.close")


def test_real_browser_dispatch_crash_is_quarantined_without_replay(tmp_path):
    if os.name != "nt":
        pytest.skip("This fixture verifies Windows Job-owned executor recovery")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set explicit TEST_CHROMIUM to an isolated Chromium executable")
    executable = Path(executable).resolve()
    assert executable.is_file(), f"TEST_CHROMIUM not found: {executable}"

    server = _FixtureServer()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    site = f"http://127.0.0.1:{server.server_address[1]}"
    profile = tmp_path / "chromium-profile"
    endpoint = f"http://127.0.0.1:{_unused_port()}"
    browser_log = browser_process = worker = None
    worker_job = None
    try:
        profile.mkdir()
        browser_log = (tmp_path / "chromium.log").open("w", encoding="utf-8")
        browser_process = subprocess.Popen([
            str(executable), "--headless=new", "--disable-dev-shm-usage",
            "--disable-background-networking", "--disable-extensions", "--disable-sync",
            "--no-proxy-server", "--no-first-run", "--no-default-browser-check",
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={endpoint.rsplit(':', 1)[1]}",
            f"--user-data-dir={profile}", "about:blank",
        ], stdin=subprocess.DEVNULL, stdout=browser_log, stderr=browser_log,
            creationflags=_flags())
        _wait(lambda: _endpoint_ready(endpoint), 10, "fixture CDP did not start", browser_process)
        with sync_playwright() as playwright, ExitStack() as owned_browser:
            connection = playwright.chromium.connect_over_cdp(endpoint)
            owned_browser.callback(connection.close)
            owned_browser.callback(_close_owned_browser, connection)
            page = connection.contexts[0].new_page()
            owned_browser.callback(page.close)
            page.goto(site + "/", wait_until="load", timeout=5000)
            session = page.context.new_cdp_session(page)
            target_id = session.send("Target.getTargetInfo")["targetInfo"]["targetId"]
            session.detach()
            registry = _registry(tmp_path, endpoint, site)
            command = [sys.executable, str(Path(__file__).resolve()), "invoke",
                       str(registry), "dispatch", RUN_ID, target_id]
            worker_out = (tmp_path / "worker.stdout").open("w", encoding="utf-8")
            worker_err = (tmp_path / "worker.stderr").open("w", encoding="utf-8")
            try:
                from jevlocal.integration.semdecide_cli import _Job
                worker_job = _Job()
                worker = subprocess.Popen(command, env=_env(), stdin=subprocess.DEVNULL,
                                          stdout=worker_out, stderr=worker_err,
                                          creationflags=_flags())
                worker_job.assign(worker)
                (tmp_path / "worker-admitted.ready").write_text("admitted", encoding="ascii")
            finally:
                worker_out.close()
                worker_err.close()
            _wait(lambda: (tmp_path / "click-returned.ready").is_file(), 12,
                  "real CDPBrowser.click did not return", worker)
            _wait(lambda: server.count() == 1, 5, "fixture did not observe exactly one click effect", worker)
            screenshot = tmp_path / "after-effect.png"
            page.screenshot(path=str(screenshot), full_page=True)
            before = _snapshot(registry)
            assert before["record"]["state"] == State.CLICK
            assert before["record"]["pending_effect"]
            assert before["record"]["steps"] == 1
            assert before["effects"] == [(before["record"]["pending_effect"], "click", "dispatching")]
            assert before["leases"] == [(digest([endpoint, target_id]), RUN_ID)]
            assert before["grants"] == 0
            assert _invoke(registry, "cancel")["status"] == "busy"

            # Terminate only this Popen child; never resolve an old PID by name.
            worker.terminate()
            worker.wait(timeout=5)
            recovered = _invoke(registry, "start", target_id=target_id)
            assert recovered["state"] == State.OUTCOME_UNKNOWN
            assert recovered["reason"] == "interrupted_run_do_not_replay"
            unknown = _snapshot(registry)
            assert unknown["record"]["reason"] == "interrupted_run_do_not_replay"
            assert unknown["record"]["pending_effect"] == before["record"]["pending_effect"]
            assert unknown["effects"] == before["effects"]
            assert unknown["leases"] == before["leases"]
            assert unknown["grants"] == 0
            assert _invoke(registry, "status")["state"] == State.OUTCOME_UNKNOWN
            assert _invoke(registry, "cancel")["state"] == State.OUTCOME_UNKNOWN
            after_cancel = _snapshot(registry)
            assert after_cancel["record"]["pending_effect"] == before["record"]["pending_effect"]
            assert after_cancel["effects"] == before["effects"]
            assert after_cancel["leases"] == before["leases"]
            resumed = _invoke(registry, "resume")
            assert resumed["state"] == State.OUTCOME_UNKNOWN and resumed["resume_rejected"]
            repeated = _invoke(registry, "start", target_id=target_id)
            assert repeated["state"] == State.OUTCOME_UNKNOWN and repeated["idempotent_replay"]
            other = _invoke(registry, "start", "another-run", target_id)
            assert other == {"status": "rejected", "reason": "target_leased", "run_id": "another-run"}
            assert server.count() == 1

            # The test operator has inspected only this local fixture's count.
            closed = _review_close(registry)
            assert closed == {"state": State.CLOSED_AFTER_REVIEW, "run_id": RUN_ID,
                              "business_credit": False}
            final = _snapshot(registry)
            assert final["record"]["state"] == State.CLOSED_AFTER_REVIEW
            assert final["record"]["pending_effect"] == before["record"]["pending_effect"]
            assert final["effects"] == before["effects"]
            assert final["leases"] == [] and final["grants"] == 0
            assert _invoke(registry, "status")["state"] == State.CLOSED_AFTER_REVIEW
            assert server.count() == 1
            (tmp_path / "crash-recovery-evidence.json").write_text(json.dumps({
                "fixture": "loopback-only isolated Chromium click",
                "python_version": sys.version.split()[0],
                "playwright_version": version("playwright"),
                "chromium_version": connection.version,
                "browser_pid": browser_process.pid,
                "executor_pid": worker.pid,
                "side_effect_count": server.count(),
                "states": [before["record"]["state"], unknown["record"]["state"],
                           final["record"]["state"]],
                "recovery_reason": recovered["reason"],
                "same_pending_effect_through_review": (
                    before["record"]["pending_effect"] ==
                    unknown["record"]["pending_effect"] ==
                    final["record"]["pending_effect"]),
                "effect_status_through_review": "dispatching",
                "unknown_lease_retained": unknown["leases"] == before["leases"],
                "review_closed_lease_released": final["leases"] == [],
                "grant_count_through_review": 0,
                "review_close_business_credit": closed["business_credit"],
                "test_doubles": ["deterministic predictor", "qualification check"],
                "real_boundaries": ["service.invoke", "Store.reserve", "CDPBrowser.click",
                                    "loopback HTTP POST", "review-close CLI"],
                "screenshot": screenshot.name,
            }, indent=2) + "\n", encoding="utf-8")
    finally:
        if worker_job is not None:
            try:
                worker_job.terminate()
                worker_job.wait_empty(time.monotonic() + 5)
            finally:
                worker_job.close()
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
        if browser_process is not None:
            if browser_process.poll() is None:
                try:
                    # ExitStack sent Browser.close to this owned CDP instance.
                    browser_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    browser_process.terminate()
                    try:
                        browser_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        browser_process.kill()
                        browser_process.wait(timeout=5)
        if browser_log is not None:
            browser_log.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        if profile.exists() and profile.resolve().is_relative_to(tmp_path.resolve()):
            shutil.rmtree(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("invoke",))
    parser.add_argument("registry", type=Path)
    parser.add_argument("operation")
    parser.add_argument("run_id")
    parser.add_argument("target_id", nargs="?")
    args = parser.parse_args()
    _invoke_child(args.registry, args.operation, args.run_id, args.target_id)
