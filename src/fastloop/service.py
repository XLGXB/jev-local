"""CDP connection, cross-process tab lock and CLI. Does not modify Codex internals."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from filelock import FileLock, Timeout
from playwright.async_api import async_playwright

from .browser import CDPBrowser
from .contracts import Guard, State, Task, canonical, digest, loopback, origin
from .engine import Machine
from .model_client import HTTPDecision
from .store import Store


def read(path):
    def pairs(entries):
        out = {}
        for k, v in entries:
            if k in out:
                raise Guard("duplicate_json_key")
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def config(path):
    cfg = read(path)
    if cfg.get("schema_version") != 2:
        raise Guard("registry_requires_schema_v2")
    loopback(cfg["cdp"]); loopback(cfg["decision_service"])
    if not cfg.get("project_id"):
        raise Guard("project_id_required")
    if cfg.get("controller") != "fastloop":
        raise Guard("single_fastloop_controller_required")
    if cfg.get("control_mode") != "isolated-cdp":
        raise Guard("supported_host_adapter_not_connected_no_cdp_fallback")
    return cfg


def require_qualification(cfg, task):
    """Quality evidence is necessary, but never replaces task/grant approval."""
    from jevlocal.integration.qualification import verify_receipt
    evidence = cfg.get("qualification")
    required = {"receipt", "calibration_path", "evaluation_path", "policy_path", "dataset_dir"}
    optional = {"parity_path", "golden_path", "bundle_manifest_path"}
    if not isinstance(evidence, dict) or not required <= set(evidence) or set(evidence) - required - optional:
        raise Guard("qualification_evidence_required")
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in evidence.values()):
        raise Guard("qualification_paths_must_be_explicit_absolute")
    try:
        receipt = read(evidence["receipt"])
        paths = {key: value for key, value in evidence.items() if key != "receipt"}
        valid = verify_receipt(receipt, runtime_fingerprint=task.expected_runtime_fingerprint, **paths)
        policy_bytes = Path(evidence["policy_path"]).read_bytes()
        policy = json.loads(policy_bytes)
        # Evaluate exactly the action slice that the receipt qualified. Even a
        # stricter cutoff has a different coverage/error sample, not a new receipt.
        threshold_bound = (hashlib.sha256(policy_bytes).hexdigest() == receipt.get("policy_sha256") and
                           task.min_probability == policy["action_confidence_threshold"])
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise Guard("runtime_quality_evidence_not_qualified")
    if not threshold_bound:
        raise Guard("task_threshold_outside_qualified_policy")


async def page_target_id(page):
    session = await page.context.new_cdp_session(page)
    try:
        return (await session.send("Target.getTargetInfo"))["targetInfo"]["targetId"]
    finally:
        await session.detach()


async def targets(cfg):
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cfg["cdp"], timeout=5000)
        result = []
        for context in browser.contexts:
            for page in context.pages:
                try:
                    site = origin(page.url)
                except Guard:
                    site = "unsupported"
                result.append({"target_id": await page_target_id(page), "origin": site, "url_hash": digest(page.url)})
        return result


def machine_kwargs(cfg, store, run_id, *, task_id=None, target_id=None, dry_run=None):
    if task_id is None:
        r = store.get(run_id)
        if r["project_id"] != cfg["project_id"]:
            raise Guard("cross_project_run_forbidden")
        task_id, target_id, dry_run = r["task_id"], r["target_id"], r["dry_run"]
    task = Task.model_validate(cfg["tasks"][task_id])
    public = serialization.load_pem_public_key(Path(cfg["approval_public_key"]).read_bytes())
    if not isinstance(public, Ed25519PublicKey):
        raise Guard("Ed25519_public_key_required")
    return dict(task=task, public_key=public, run_id=run_id, target_id=target_id,
                target_key=digest([cfg["cdp"], target_id]), project_id=cfg["project_id"],
                task_id=task_id, dry_run=dry_run, store=store)


async def invoke(registry, operation, run_id, *, task_id=None, target_id=None,
                 dry_run=True, token=None, values=None):
    cfg = config(registry)
    store = Store(cfg["state_dir"])
    lock = client = adapter = None
    try:
        kw = machine_kwargs(cfg, store, run_id, task_id=task_id if operation == "start" else None,
                            target_id=target_id, dry_run=dry_run)
        lock = FileLock(str(store.root / (kw["target_key"] + ".lock")))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return {"status": "busy", "run_id": run_id}
        if operation in {"status", "cancel"}:
            engine = Machine(browser=None, predict=None, **kw)
            if operation == "status":
                engine._load()
                return engine.result()
            return engine.cancel()
        if not kw["dry_run"]:
            require_qualification(cfg, kw["task"])
        if operation in {"start", "resume"}:
            client = HTTPDecision(cfg["decision_service"], os.environ.get("LOCAL_DECISION_TOKEN", ""))
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(cfg["cdp"], timeout=5000)
            matches = []
            for context in browser.contexts:
                for page in context.pages:
                    if await page_target_id(page) == kw["target_id"]:
                        matches.append(page)
            if len(matches) != 1:
                raise Guard("explicit_target_not_found")
            adapter = await CDPBrowser.attach(matches[0], kw["task"], store.state_key)
            engine = Machine(browser=adapter, predict=client, **kw)
            try:
                if operation == "start":
                    return await engine.start()
                if operation == "resume":
                    return await engine.resume()
                if operation == "fill":
                    return await engine.form_fill(token, values)
                if operation == "refresh":
                    return await engine.refresh_form()
                raise Guard("unknown_operation")
            finally:
                await adapter.close(); adapter = None
            # Deliberately do NOT close an externally owned browser.
    except Guard as exc:
        return {"status": "rejected", "reason": str(exc), "run_id": run_id}
    except Exception as exc:
        # Do not reflect exception messages which may contain secrets/field values.
        return {"status": "error", "error_type": type(exc).__name__, "run_id": run_id}
    finally:
        if client:
            await client.close()
        if lock and lock.is_locked:
            lock.release()
        store.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True)
    p.add_argument("--out")
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("targets")
    start = sub.add_parser("start")
    start.add_argument("--task", required=True); start.add_argument("--target", required=True)
    start.add_argument("--run", required=True); start.add_argument("--execute", action="store_true")
    for name in ("status", "resume", "refresh", "cancel", "review-close"):
        s = sub.add_parser(name); s.add_argument("--run", required=True)
    fill = sub.add_parser("fill")
    fill.add_argument("--run", required=True); fill.add_argument("--grant", required=True); fill.add_argument("--values", required=True)
    a = p.parse_args()
    if a.op == "targets":
        out = asyncio.run(targets(config(a.registry)))
    elif a.op == "review-close":
        cfg = config(a.registry); store = Store(cfg["state_dir"])
        try:
            r = store.get(a.run)
            if r["project_id"] != cfg["project_id"]:
                raise Guard("cross_project_run_forbidden")
            with FileLock(str(store.root / (r["target_key"] + ".lock")), timeout=0):
                if r["state"] != State.OUTCOME_UNKNOWN:
                    raise Guard("only_unknown_outcomes_may_be_review_closed")
                print("Inspect real page and effects first. No success credit will be created.")
                if input("Type CLOSE_WITHOUT_CREDIT: ") != "CLOSE_WITHOUT_CREDIT":
                    raise Guard("not_confirmed")
                r["state"] = State.CLOSED_AFTER_REVIEW
                store.save(r, {"event": "quarantine_released_after_manual_review", "business_credit": False})
                out = {"state": r["state"], "run_id": a.run, "business_credit": False}
        finally:
            store.close()
    else:
        kwargs = {}
        if a.op == "start":
            kwargs = dict(task_id=a.task, target_id=a.target, dry_run=not a.execute)
        if a.op == "fill":
            kwargs = dict(token=read(a.grant), values=read(a.values))
        out = asyncio.run(invoke(a.registry, a.op, a.run, **kwargs))
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
