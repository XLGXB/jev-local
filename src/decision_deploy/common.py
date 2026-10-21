from __future__ import annotations
import hashlib, importlib.metadata, json, math, os, platform
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np

KINDS = {"choice": 0, "score": 1, "noul": 2}

def dumps(value, *, sorted_keys=False):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=sorted_keys, separators=(",", ":"))

def digest(value):
    return hashlib.sha256(dumps(value, sorted_keys=True).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""): h.update(chunk)
    return h.hexdigest()

def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f: f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")

def unique_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise ValueError("duplicate JSON key")
        out[k] = v
    return out

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)

def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "peft", "onnx", "onnxruntime", "onnxruntime-gpu", "tensorrt", "playwright"):
        try: out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: out[pkg] = None
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = {"name": p.name, "cc": [p.major, p.minor], "torch_cuda": torch.version.cuda}
    except ImportError: pass
    return out

def loopback_url(url):
    s = urlsplit(url)
    if s.scheme not in ("http", "ws") or s.hostname not in ("127.0.0.1", "localhost", "::1") or s.username or s.password:
        raise ValueError("only explicit loopback http/ws endpoints are allowed")
    return url

def origin(url):
    s = urlsplit(url)
    if s.scheme not in ("http", "https") or not s.hostname or s.username or s.password:
        raise ValueError("unsupported page URL")
    default = 80 if s.scheme == "http" else 443
    h = f"[{s.hostname}]" if ":" in s.hostname else s.hostname
    return f"{s.scheme}://{h}" + (f":{s.port}" if s.port and s.port != default else "")

def probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all(): raise ValueError("invalid decision logits")
    if not math.isfinite(temperature) or temperature <= 0: raise ValueError("invalid temperature")
    z = z / temperature; z -= z.max(); p = np.exp(z); return p / p.sum()

def atoms_for(request):
    """Reuse v1's public-question/Atom contracts; dummy targets are never encoded."""
    from decision_ft.data import Atom, option_keys
    from .protocol import validate_request
    validate_request(request)
    if set(request) != {"state", "questions"}: raise ValueError("request needs state and questions only")
    qs = request["questions"]
    if not isinstance(qs, dict) or not 1 <= len(qs) <= 8: raise ValueError("1..8 questions required")
    out = []
    for qid, q in qs.items():
        if not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"}: raise ValueError("labels/extra fields forbidden")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip(): raise ValueError("instructions required")
        keys = option_keys(q)
        if len(keys) > 16: raise ValueError("deployment profile supports at most 16 candidates")
        out.append((qid, Atom(qid, "inference", request["state"], q, keys, [], 0, "inference", "not_a_label")))
    dumps(request)  # reject nonfinite JSON
    return out
