"""Durable per-run receipts. Unknown outcomes must NEVER be replayed automatically."""
from __future__ import annotations
import json,re,sqlite3,time
from pathlib import Path
from .common import digest,save

class Journal:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/"journal.sqlite",timeout=2)
        self.db.execute("PRAGMA journal_mode=WAL");self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, fingerprint TEXT, status TEXT, result TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, run_id TEXT, observation TEXT, action TEXT, status TEXT)")
        self.db.commit()
    def begin(self,run_id,fingerprint):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",run_id):raise ValueError("invalid run_id")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row=self.db.execute("SELECT fingerprint,status,result FROM runs WHERE id=?",(run_id,)).fetchone()
            if row:
                if row[0]!=fingerprint:raise ValueError("run_id was used for a different target/task/mode")
                return json.loads(row[2]) if row[2] else {"status":"outcome_unknown","run_id":run_id,"reason":"incomplete durable run exists; inspect evidence, do not replay"}
            self.db.execute("INSERT INTO runs VALUES (?,?,?,NULL)",(run_id,fingerprint,"started"))
        return None
    def reserve(self,run_id,observation,action):
        action_id=digest([run_id,observation,action])
        with self.db:self.db.execute("INSERT INTO actions VALUES (?,?,?,?,?)",(action_id,run_id,observation,action,"dispatching"))
        return action_id
    def action_done(self,action_id):
        with self.db:self.db.execute("UPDATE actions SET status='returned' WHERE id=?",(action_id,))
    def evidence(self,run_id,observation):
        hash_=digest(observation);path=self.root/run_id/f"{hash_}.json"
        if not path.exists():save(path,observation)
        return {"path":str(path),"sha256":hash_,"hash_kind":"canonical_json_sha256"}
    def finish(self,run_id,result):
        with self.db:self.db.execute("UPDATE runs SET status=?,result=? WHERE id=?",(result["status"],json.dumps(result,ensure_ascii=False),run_id))
        return result
    def close(self):self.db.close()
