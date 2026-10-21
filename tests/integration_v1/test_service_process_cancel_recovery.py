"""Windows OS-process recovery through the real service.invoke cancel path.

Synthetic state only: no CDP connection, model request, qualification, or grant.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from filelock import FileLock
import pytest

from fastloop.contracts import State, digest
from fastloop.engine import Machine
from fastloop.service import config, invoke, machine_kwargs
from fastloop.store import Store


RUN_ID = "service-cancel-recovery"
TASK_ID = "synthetic-task"
TARGET_ID = "synthetic-target"


def _registry(root: Path) -> Path:
    key_path = root / "approval-public.pem"
    public = Ed25519PrivateKey.generate().public_key()
    key_path.write_bytes(public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    path = root / "registry.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "project_id": "synthetic-project",
        "controller": "fastloop",
        "control_mode": "isolated-cdp",
        "cdp": "http://127.0.0.1:9222",
        "decision_service": "http://127.0.0.1:8000",
        "state_dir": str(root / "state"),
        "approval_public_key": str(key_path),
        "tasks": {TASK_ID: {
            "goal": "Synthetic cancellation recovery",
            "allowed_origins": ["https://example.invalid"],
            "verifier": {"selector": "#never-used"},
            "execution_approved": False,
            "expected_runtime_fingerprint": "a" * 64,
        }},
    }), encoding="utf-8")
    return path


def _prepare(registry: Path) -> None:
    cfg = config(registry)
    store = Store(cfg["state_dir"])
    kw = machine_kwargs(cfg, store, RUN_ID, task_id=TASK_ID, target_id=TARGET_ID, dry_run=False)
    lock = FileLock(str(store.root / (kw["target_key"] + ".lock")))
    try:
        with lock:
            machine = Machine(browser=None, predict=None, **kw)
            now = time.time()
            record = {
                "run_id": RUN_ID, "scope": machine.scope, "target_id": TARGET_ID,
                "target_key": digest([cfg["cdp"], TARGET_ID]),
                "project_id": cfg["project_id"], "task_id": TASK_ID,
                "policy_hash": machine.policy_hash, "dry_run": False,
                "state": State.NEW, "created_at": now,
                "expires_at": now + machine.task.run_ttl_seconds,
                "steps": 0, "scroll_count": 0, "active_ms": 0.0,
                "pending_effect": None, "grant_in_progress": False,
                "used_grant": None, "handoff": None,
                "memory": {"doc": None, "clicks": [], "forms": {},
                           "scroll": {}, "attempts": {}, "recent": []},
            }
            assert store.create(record)
            machine._load()
            for state in (State.OBSERVE, State.DECIDE, State.VALIDATE, State.CLICK):
                machine._transition(state)
            effect = store.reserve(machine.r, "click", "click:synthetic")
            assert effect == store.get(RUN_ID)["pending_effect"]
            (registry.parent / "committed.ready").write_text("ready", encoding="ascii")
            while True:
                time.sleep(60)
    finally:
        store.close()


def _never(*args, **kwargs):
    raise AssertionError("browser/model/qualification path must not run")


def _invoke_child(registry: Path, operation: str) -> None:
    import fastloop.service as service

    service.async_playwright = _never
    service.HTTPDecision = _never
    service.require_qualification = _never
    result = asyncio.run(invoke(registry, operation, RUN_ID))
    print(json.dumps(result), flush=True)


def _command(mode: str, registry: Path, operation: str | None = None) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), mode, str(registry)]
    if operation is not None:
        cmd.append(operation)
    return cmd


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH", ""))))
    return env


def _snapshot(registry: Path) -> dict:
    cfg = config(registry)
    store = Store(cfg["state_dir"])
    try:
        row = store.get(RUN_ID)
        return {
            "record": row,
            "effects": store.db.execute(
                "SELECT id, kind, status FROM effects WHERE run_id=? ORDER BY rowid", (RUN_ID,)
            ).fetchall(),
            "lease": store.db.execute(
                "SELECT target, run_id FROM leases WHERE run_id=?", (RUN_ID,)
            ).fetchall(),
            "grants": store.db.execute("SELECT count(*) FROM grants").fetchone()[0],
        }
    finally:
        store.close()


def _call(registry: Path, operation: str, flags: int) -> tuple[dict, dict]:
    command = _command("invoke", registry, operation)
    started = time.time()
    child = subprocess.Popen(
        command, env=_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=flags,
    )
    try:
        stdout, stderr = child.communicate(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
    event = {"argv": command, "pid": child.pid, "started_at": started,
             "ended_at": time.time(), "exit_code": child.returncode,
             "stdout": stdout, "stderr": stderr}
    assert child.returncode == 0, event
    return json.loads(stdout), event


@pytest.mark.skipif(os.name != "nt", reason="Windows-only OS-process/FileLock regression")
def test_service_cancel_after_locked_process_exit(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    flags = subprocess.CREATE_NO_WINDOW
    command = _command("prepare", registry)
    started = time.time()
    child = subprocess.Popen(
        command, env=_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        creationflags=flags,
    )
    events = []
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "committed.ready").is_file() and time.monotonic() < deadline:
            assert child.poll() is None, "prepare child exited before commit"
            time.sleep(0.025)
        assert (tmp_path / "committed.ready").is_file(), "prepare child did not commit"
        before = _snapshot(registry)
        assert before["record"]["state"] == State.CLICK
        assert before["record"]["project_id"] == "synthetic-project"
        assert before["record"]["task_id"] == TASK_ID
        assert before["record"]["dry_run"] is False
        assert before["record"]["target_key"] == digest([config(registry)["cdp"], TARGET_ID])
        assert before["record"]["pending_effect"]
        assert before["record"]["steps"] == 1
        assert before["effects"] == [(before["record"]["pending_effect"], "click", "dispatching")]
        assert before["lease"] == [(before["record"]["target_key"], RUN_ID)]
        busy, event = _call(registry, "cancel", flags)
        events.append(event)
        assert busy == {"status": "busy", "run_id": RUN_ID}
        assert _snapshot(registry) == before

        child.terminate()
        child.wait(timeout=5)
        events.insert(0, {"argv": command, "pid": child.pid, "started_at": started,
                          "ended_at": time.time(), "exit_code": child.returncode,
                          "stdout": "", "stderr": child.stderr.read()})
        cancelled, event = _call(registry, "cancel", flags)
        events.append(event)
        assert cancelled["state"] == State.OUTCOME_UNKNOWN
        assert cancelled["reason"] == "cancelled_with_possible_effect"
        assert cancelled["steps"] == 1
        after = _snapshot(registry)
        assert after["record"]["state"] == State.OUTCOME_UNKNOWN
        assert after["record"]["scope"] == before["record"]["scope"]
        assert after["record"]["pending_effect"] == before["record"]["pending_effect"]
        assert after["record"]["steps"] == before["record"]["steps"]
        assert after["effects"] == before["effects"]
        assert after["lease"] == before["lease"]
        assert after["grants"] == before["grants"] == 0
        status, event = _call(registry, "status", flags)
        events.append(event)
        assert status["state"] == State.OUTCOME_UNKNOWN
        assert status["steps"] == 1
        print("SERVICE_CANCEL_PROCESS_EVIDENCE=" + json.dumps(events), flush=True)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stderr:
            child.stderr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "invoke"))
    parser.add_argument("registry", type=Path)
    parser.add_argument("operation", nargs="?")
    args = parser.parse_args()
    if args.mode == "prepare":
        _prepare(args.registry)
    else:
        assert args.operation in {"cancel", "status"}
        _invoke_child(args.registry, args.operation)
