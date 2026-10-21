"""OS-process crash recovery for synthetic, durable fastloop state.

This is not a browser or host-qualification test. A child commits SQLite state,
signals readiness, and is terminated before an independent child calls Machine.start.
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

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from filelock import FileLock, Timeout
import pytest

from fastloop.contracts import State, Task
from fastloop.engine import Machine
from fastloop.store import Store


RUN_ID = "crash"
TARGET_KEY = "target-key"


def _task() -> Task:
    return Task.model_validate({
        "goal": "Test-only process recovery",
        "allowed_origins": ["https://example.invalid"],
        "verifier": {"selector": "#never-used"},
        "execution_approved": True,
        "expected_runtime_fingerprint": "a" * 64,
    })


class _FailOnCall:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected browser/predict call: {name}")

    def __call__(self, *args, **kwargs):
        raise AssertionError("unexpected prediction call")


def _machine(store: Store) -> Machine:
    return Machine(
        task=_task(), browser=_FailOnCall(), predict=_FailOnCall(), store=store,
        public_key=Ed25519PrivateKey.generate().public_key(), run_id=RUN_ID,
        target_id="target", target_key=TARGET_KEY,
    )


def _effect_rows(store: Store) -> list[tuple[str, str]]:
    return store.db.execute(
        "SELECT kind, status FROM effects WHERE run_id=? ORDER BY rowid", (RUN_ID,)
    ).fetchall()


def _lease_count(store: Store) -> int:
    return store.db.execute(
        "SELECT count(*) FROM leases WHERE target=? AND run_id=?", (TARGET_KEY, RUN_ID)
    ).fetchone()[0]


def _prepare_child(root: Path, case: str) -> None:
    store = Store(root)
    lock = FileLock(str(root / (TARGET_KEY + ".lock")))
    try:
        with lock:
            machine = _machine(store)

            async def stop_at_committed_checkpoint():
                machine._transition(State.OBSERVE)
                if case == "after_reserve":
                    machine._transition(State.DECIDE)
                    machine._transition(State.VALIDATE)
                    machine._transition(State.CLICK)
                    store.reserve(machine.r, "click", "click:fixture")
                elif case == "grant_in_progress":
                    # Persist only the interruption marker. No grant is made or
                    # represented; no form input, approval token, or host exists.
                    machine.r["state"] = State.CODEX_FILL
                    machine.r["grant_in_progress"] = True
                    store.save(machine.r)
                else:
                    assert case == "before_dispatch"
                return machine.result()

            machine._drive = stop_at_committed_checkpoint
            asyncio.run(machine.start())
            # The marker is written only after the Store transaction committed.
            (root / "committed.ready").write_text(case, encoding="ascii")
            while True:
                time.sleep(60)
    finally:
        store.close()


def _recover_child(root: Path) -> None:
    store = Store(root)
    try:
        before = store.get(RUN_ID)
        effects_before = _effect_rows(store)
        lease_before = _lease_count(store)
        result = asyncio.run(_machine(store).start())
        after = store.get(RUN_ID)
        print(json.dumps({
            "result": result,
            "before": before,
            "after": after,
            "effects_before": effects_before,
            "effects_after": _effect_rows(store),
            "lease_before": lease_before,
            "lease_after": _lease_count(store),
            "grant_count": store.db.execute("SELECT count(*) FROM grants").fetchone()[0],
        }), flush=True)
    finally:
        store.close()


def _child_command(mode: str, root: Path, case: str | None = None) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), mode, str(root)]
    if case is not None:
        command.append(case)
    return command


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH", ""))))
    return env


def _await_committed(root: Path, child: subprocess.Popen) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (root / "committed.ready").is_file():
            assert child.poll() is None, "preparation process exited after checkpoint"
            return
        if child.poll() is not None:
            pytest.fail(f"preparation process exited before checkpoint: {child.returncode}")
        time.sleep(0.025)
    pytest.fail("preparation process did not commit within 10 seconds")


@pytest.mark.skipif(os.name != "nt", reason="Windows process and FileLock regression")
@pytest.mark.parametrize("case", ["before_dispatch", "after_reserve", "grant_in_progress"])
@pytest.mark.parametrize("stop", ["terminate", "kill"])
def test_os_process_crash_recovery(tmp_path: Path, case: str, stop: str) -> None:
    root = tmp_path / case
    root.mkdir()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child = subprocess.Popen(
        _child_command("prepare", root, case), env=_child_env(),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        _await_committed(root, child)
        # The OS lock is live while the producer exists. It is independent of
        # the durable logical lease in SQLite.
        with pytest.raises(Timeout):
            FileLock(str(root / (TARGET_KEY + ".lock"))).acquire(timeout=0)
        getattr(child, stop)()
        child.wait(timeout=5)
        assert child.poll() is not None

        with FileLock(str(root / (TARGET_KEY + ".lock")), timeout=1):
            recovered = subprocess.run(
                _child_command("recover", root), env=_child_env(),
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                creationflags=flags, timeout=10, check=True,
            )
        observed = json.loads(recovered.stdout)
        expected = State.NEEDS_HOST if case == "before_dispatch" else State.OUTCOME_UNKNOWN
        assert observed["result"]["state"] == expected
        assert observed["after"]["state"] == expected
        assert observed["result"]["steps"] == observed["before"]["steps"]
        assert observed["after"]["steps"] == observed["before"]["steps"]
        assert observed["effects_after"] == observed["effects_before"]
        assert observed["grant_count"] == 0
        if case == "before_dispatch":
            assert observed["effects_before"] == []
            assert observed["lease_before"] == 1
            assert observed["lease_after"] == 0
        else:
            assert observed["lease_before"] == observed["lease_after"] == 1
            if case == "after_reserve":
                assert observed["effects_before"] == [["click", "dispatching"]]
                assert observed["before"]["pending_effect"] == observed["after"]["pending_effect"]
            else:
                assert observed["effects_before"] == []
                assert observed["before"]["grant_in_progress"] is True
                assert observed["after"]["grant_in_progress"] is True
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "recover"))
    parser.add_argument("root", type=Path)
    parser.add_argument("case", nargs="?")
    args = parser.parse_args()
    if args.mode == "prepare":
        assert args.case is not None
        _prepare_child(args.root, args.case)
    else:
        _recover_child(args.root)
