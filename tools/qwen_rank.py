"""Explicit isolated Qwen relevance ranking; never changes lexical primary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from jevlocal.integration import qwen_context as qc

PYTHON = Path("D:/Projects/JevLocal/.venvs/qwen-cuda/Scripts/python.exe")
MAX_JSON = 4 * 1024 * 1024


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON + 1)
    return qc.strict_json(raw, max_bytes=MAX_JSON)


def _write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(qc.encoded(value) + b"\n")


def _offline():
    def denied(*_a, **_kw):
        raise OSError("offline Qwen worker")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def _worker(payload_path, result_path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TORCH_FORCE_WEIGHTS_ONLY_LOAD"):
        if os.environ.get(key) != "1":
            raise ValueError("offline flags required at process start")
    _offline()
    payload = _read(payload_path)
    if not isinstance(payload, dict) or set(payload) != {"query", "safe", "snapshot_sha256", "config", "identity", "invocation_id"}:
        raise ValueError("worker payload shape")
    query, safe = payload["query"], payload["safe"]
    if not isinstance(query, str) or not isinstance(safe, list) or not 1 <= len(safe) <= 128:
        raise ValueError("worker candidate shape")
    if qc.digest({"query": query, "candidates": safe}) != payload["snapshot_sha256"]:
        raise ValueError("worker snapshot binding")
    identity = qc.validate_config(payload["config"])
    if identity != payload["identity"]:
        raise ValueError("worker model identity")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 unavailable")
    checkpoint = Path(payload["config"]["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                         trust_remote_code=False, padding_side="left")
    prompt_rows = qc.prompts(query, safe)
    lengths = qc.check_prompt_lengths(prompt_rows, tokenizer.encode)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if len(no_ids) != 1 or len(yes_ids) != 1 or no_ids == yes_ids:
        raise ValueError("yes/no token shape")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
             trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    scores = []
    forwards = 0
    source_by_id = {c["id"]: c["source"] for c in safe}
    for offset in range(0, len(prompt_rows), 4):
        batch = prompt_rows[offset:offset+4]
        inputs = tokenizer([x["text"] for x in batch], padding=True, truncation=False,
                           add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            forwards += 1
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, [no_ids[0], yes_ids[0]]].float()
        if tuple(logits.shape) != (len(batch), 2) or not torch.isfinite(logits).all().item():
            raise ValueError("native logits invalid")
        for row, pair in zip(batch, logits.cpu().tolist()):
            scores.append({"id": row["id"],
                           "source_sha256": hashlib.sha256(source_by_id[row["id"]].encode("utf-8")).hexdigest(),
                           "no_logit": pair[0], "yes_logit": pair[1]})
    _write_new(result_path, {"snapshot_sha256": payload["snapshot_sha256"], "identity": identity,
             "invocation_id": payload["invocation_id"], "scores": scores,
             "usage": {"judgments": len(prompt_rows), "forwards": forwards, "returned": len(scores),
                       "token_lengths": dict(zip((x["id"] for x in prompt_rows), lengths))}})


def invoke(input_path, config_path, *, output_path=None, timeout=60.0, popen=subprocess.Popen):
    started = time.monotonic()
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be in (0,60]")
    if not PYTHON.is_file():
        raise ValueError("fixed Qwen Python absent")
    protected = {Path(input_path).resolve(), Path(config_path).resolve()}
    if output_path is not None:
        target = Path(output_path)
        if target.resolve() in protected or target.exists() or target.is_symlink() or not target.parent.is_dir():
            raise ValueError("output must be a new distinct file")
    arguments, config = _read(input_path), _read(config_path)
    query, safe, primary, snapshot = qc.prepare(arguments)
    invocation_id = secrets.token_hex(16)
    lifecycle = {"child_started": False, "pid": None, "reaped": False, "timed_out": False,
                 "exit_code": None, "retained_workdir": None}
    if primary["status"] != "ok":
        outcome = {"primary": primary,
                   "recommendation": qc.unavailable(snapshot, {"backend": qc.BACKEND}, "primary_budget", invocation_id),
                   "lifecycle": {**lifecycle, "elapsed_seconds": time.monotonic()-started},
                   "execution_authorized": False, "training_allowed": False, "primary_changed": False}
        return _bounded(outcome, snapshot, invocation_id)
    identity = qc.validate_config(config)
    reason, recommendation = None, None
    owned = None
    process = None
    try:
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        owned = Path(tempfile.mkdtemp(prefix="jevlocal-qwen-rank-", dir=temp_root)).resolve(strict=True)
        if owned.parent != temp_root or not owned.name.startswith("jevlocal-qwen-rank-"):
            raise ValueError("temporary job ownership invalid")
        payload = owned/"payload.json"
        result = owned/"result.json"
        _write_new(payload, {"query": query, "safe": safe, "snapshot_sha256": snapshot,
                             "config": config, "identity": identity, "invocation_id": invocation_id})
        if time.monotonic()-started >= timeout:
            lifecycle["timed_out"] = True
            reason = "timeout"
        if reason is None:
            environment = os.environ.copy()
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
                environment.pop(key, None)
            environment["PYTHONNOUSERSITE"] = "1"
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TORCH_FORCE_WEIGHTS_ONLY_LOAD"):
                environment[key] = "1"
            command = [str(PYTHON), "-I", "-B", "-X", "utf8", str(Path(__file__).resolve()),
                       "--worker", str(payload), str(result)]
            process = popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=environment,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            lifecycle["child_started"] = True
            lifecycle["pid"] = getattr(process, "pid", None)
            try:
                remaining = timeout - (time.monotonic()-started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                lifecycle["exit_code"] = process.wait(timeout=remaining)
                lifecycle["reaped"] = True
            except subprocess.TimeoutExpired:
                lifecycle["timed_out"] = True
                reason = "timeout"
            if reason is None and lifecycle["exit_code"] == 0:
                child = _read(result)
                recommendation = qc.compose(snapshot, safe, identity, invocation_id, child)
            elif reason is None:
                reason = "worker_failed"
    except Exception:
        reason = "worker_or_binding_failed"
    finally:
        if process is not None and not lifecycle["reaped"]:
            # One cleanup deadline for this owned child. Never restart a full
            # five-second wait after a first cleanup timeout.
            cleanup_deadline = time.monotonic() + 5.0
            try:
                process.kill()
                lifecycle["exit_code"] = process.wait(timeout=max(0.0, cleanup_deadline-time.monotonic()))
                lifecycle["reaped"] = True
            except Exception:
                reason = "cleanup_failed"
        if owned is not None:
            # Delete only the absolute directory this invocation created, and
            # never while its child may still have the payload open.
            if not lifecycle["child_started"] or lifecycle["reaped"]:
                try:
                    if owned.parent != temp_root or not owned.name.startswith("jevlocal-qwen-rank-"):
                        raise ValueError("temporary job ownership changed")
                    shutil.rmtree(owned)
                except Exception:
                    reason = "cleanup_failed"
            else:
                reason = "cleanup_failed"
                lifecycle["retained_workdir"] = str(owned)
    if reason == "cleanup_failed":
        recommendation = None
    if recommendation is None:
        recommendation = qc.unavailable(snapshot, identity, reason or "unavailable", invocation_id)
    outcome = {"primary": primary, "recommendation": recommendation,
            "lifecycle": {**lifecycle, "elapsed_seconds": time.monotonic()-started},
            "execution_authorized": False, "training_allowed": False, "primary_changed": False}
    return _bounded(outcome, snapshot, invocation_id)


def _bounded(outcome, snapshot, invocation_id):
    if len(qc.encoded(outcome)) <= MAX_JSON:
        return outcome
    fallback = dict(outcome)
    identity = outcome["recommendation"]["identity"]
    fallback["recommendation"] = qc.unavailable(snapshot, identity, "output_budget", invocation_id)
    if len(qc.encoded(fallback)) > MAX_JSON:
        raise ValueError("lexical primary exceeds JSON transport capacity")
    return fallback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--worker", nargs=2, metavar=("PAYLOAD", "RESULT"))
    args = parser.parse_args(argv)
    if args.worker:
        try:
            _worker(*args.worker)
            return 0
        except Exception:
            return 2
    try:
        if not args.input or not args.config:
            raise ValueError("explicit input and config required")
        outcome = invoke(args.input, args.config, output_path=args.output, timeout=args.timeout)
        if args.output:
            _write_new(args.output, outcome)
        print(qc.encoded(outcome).decode("utf-8"))
        return 0 if outcome["recommendation"]["status"] == "observed" else 2
    except Exception as exc:
        print(json.dumps({"status": "rejected", "error_type": type(exc).__name__,
                          "execution_authorized": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
