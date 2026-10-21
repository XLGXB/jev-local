"""SQLite durability plus a logical tab lease spanning MCP calls. No field values stored."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

from .contracts import Guard, State, canonical, digest, safe_id


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        secret = self.root / "state-hmac.key"
        try:
            fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32)); f.flush(); os.fsync(f.fileno())
        except FileExistsError:
            pass
        self.state_key = secret.read_bytes()
        if len(self.state_key) != 32:
            raise Guard("invalid_state_key")
        self.db = sqlite3.connect(self.root / "state.sqlite", timeout=5)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, scope TEXT NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS leases(target TEXT PRIMARY KEY, run_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS effects(id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, status TEXT);
        CREATE TABLE IF NOT EXISTS grants(id TEXT PRIMARY KEY, run_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run_id TEXT, at REAL, body TEXT);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, run_id TEXT, body TEXT);
        """)
        self.db.commit()

    def get(self, run_id: str) -> dict:
        row = self.db.execute("SELECT body FROM runs WHERE id=?", (safe_id(run_id),)).fetchone()
        if not row:
            raise Guard("unknown_run")
        return json.loads(row[0])

    def create(self, record: dict) -> bool:
        safe_id(record["run_id"])
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            prior = self.db.execute("SELECT scope FROM runs WHERE id=?", (record["run_id"],)).fetchone()
            if prior:
                if prior[0] != record["scope"]:
                    raise Guard("run_scope_mismatch")
                return False
            lease = self.db.execute("SELECT run_id FROM leases WHERE target=?", (record["target_key"],)).fetchone()
            if lease:
                raise Guard("target_leased")
            self.db.execute("INSERT INTO leases VALUES (?,?)", (record["target_key"], record["run_id"]))
            self.db.execute("INSERT INTO runs VALUES (?,?,?)", (record["run_id"], record["scope"], canonical(record).decode()))
        return True

    def save(self, record: dict, event: dict | None = None) -> None:
        with self.db:
            self.db.execute("UPDATE runs SET body=? WHERE id=?", (canonical(record).decode(), record["run_id"]))
            if event:
                self.db.execute("INSERT INTO events(run_id,at,body) VALUES (?,?,?)",
                                (record["run_id"], time.time(), canonical(event).decode()))
            if State(record["state"]) in {State.NEEDS_VERIFICATION, State.NEEDS_HOST, State.CANCELLED,
                                          State.DRY_RUN, State.CLOSED_AFTER_REVIEW}:
                self.db.execute("DELETE FROM leases WHERE target=? AND run_id=?", (record["target_key"], record["run_id"]))

    def reserve(self, record: dict, kind: str, action: str) -> str:
        effect = secrets.token_hex(16)
        record["pending_effect"] = effect
        record["steps"] += 1
        with self.db:
            self.db.execute("INSERT INTO effects VALUES (?,?,?,?)", (effect, record["run_id"], kind, "dispatching"))
            self.db.execute("UPDATE runs SET body=? WHERE id=?", (canonical(record).decode(), record["run_id"]))
            self.db.execute("INSERT INTO events(run_id,at,body) VALUES (?,?,?)",
                            (record["run_id"], time.time(), canonical({"effect": effect, "kind": kind, "action": action, "status": "dispatching"}).decode()))
        return effect

    def returned(self, effect: str) -> None:
        with self.db:
            self.db.execute("UPDATE effects SET status='returned' WHERE id=?", (effect,))

    def verified(self, record: dict) -> None:
        effect = record["pending_effect"]
        record["pending_effect"] = None
        with self.db:
            self.db.execute("UPDATE effects SET status='verified' WHERE id=?", (effect,))
            self.db.execute("UPDATE runs SET body=? WHERE id=?", (canonical(record).decode(), record["run_id"]))

    def consume_grant(self, record: dict, grant_id: str) -> None:
        # Called only after signature/scope/value/page checks; spend BEFORE first input.
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            actual = self.get(record["run_id"])
            if actual["state"] != State.VALIDATE_GRANT or actual.get("used_grant"):
                raise Guard("grant_state_race")
            try:
                self.db.execute("INSERT INTO grants VALUES (?,?)", (grant_id, record["run_id"]))
            except sqlite3.IntegrityError as exc:
                raise Guard("grant_replayed") from exc
            record["used_grant"] = grant_id
            record["state"] = State.CODEX_FILL
            record["grant_in_progress"] = True
            self.db.execute("UPDATE runs SET body=? WHERE id=?", (canonical(record).decode(), record["run_id"]))
            self.db.execute("INSERT INTO events(run_id,at,body) VALUES (?,?,?)",
                            (record["run_id"], time.time(), canonical({"state": "CODEX_FILL", "grant": grant_id, "status": "consumed"}).decode()))

    def grant_owner(self, grant_id: str) -> str | None:
        row = self.db.execute("SELECT run_id FROM grants WHERE id=?", (grant_id,)).fetchone()
        return row[0] if row else None

    def observation(self, record: dict, data: dict) -> None:
        # Explicit safe projection; not a claim to retain the full original DOM.
        safe = {"kind": "typed_observation_projection", "full_snapshot_sha256": digest(data),
                "document": data["document"], "scrolls": data["scrolls"], "verified": data["verified"],
                "clicks": {}, "forms": {}}
        for id_, m in data["clicks"].items():
            safe["clicks"][id_] = {k: m[k] for k in ("node", "tag", "type", "visible", "viewport", "hit", "rect")}
        for id_, m in data["forms"].items():
            safe["forms"][id_] = {k: m[k] for k in ("signature", "identity", "ready")}
            safe["forms"][id_]["fields"] = {
                fid: {k: field[k] for k in ("node", "value_hash", "valid", "enabled", "visible")}
                for fid, field in m["fields"].items()}
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?)",
                            (digest([record["run_id"], safe]), record["run_id"], canonical(safe).decode()))

    def close(self):
        self.db.close()
