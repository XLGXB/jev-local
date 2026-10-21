"""Explicit, source-bound native Kev consumer for the state-JSON experiment."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import re
import stat
from pathlib import Path

import numpy as np

from decision_deploy.common import digest, sha, versions
from decision_deploy.runtime import Predictor
from .state_json_binding import prepare_inference_atoms, read_checkpoint_marker, validate_binding


SOURCE_MODULES = frozenset({
    "decision_ft.state_json_native", "decision_ft.state_json_binding",
    "decision_ft.state_json", "decision_ft.backends", "decision_ft.data", "decision_ft.train",
    "decision_deploy.common", "decision_deploy.protocol", "decision_deploy.runtime",
    "decision_deploy.identity", "kev.api", "kev.checkpoint", "kev.model", "kev.data",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPEC_KEYS = {"backend", "mode", "purpose", "checkpoint", "device", "weights_dtype", "amp"}


def _digest_map(value, *, keys=None):
    if type(value) is not dict or (keys is not None and set(value) != keys):
        raise ValueError("invalid frozen hash map")
    if not value or any(type(k) is not str or type(v) is not str or not _SHA256.fullmatch(v)
                        for k, v in value.items()):
        raise ValueError("invalid frozen hash map")


def _source_paths():
    """Resolve the actual module origins, not paths claimed by the freeze."""
    paths = {}
    for name in SOURCE_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"required source unavailable: {name}")
        path = Path(spec.origin)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required source is not a regular file: {name}")
        paths[name] = path
    return paths


def _checked_sources(expected):
    _digest_map(expected, keys=SOURCE_MODULES)
    actual = {name: sha(path) for name, path in _source_paths().items()}
    if actual != expected:
        raise ValueError("native consumer source hash mismatch")
    return actual


def _is_link_or_reparse(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _checked_checkpoint(checkpoint, expected):
    _digest_map(expected)
    root = Path(checkpoint)
    if (not root.is_absolute() or any(_is_link_or_reparse(p) for p in (root, *root.parents))
            or not root.is_dir()):
        raise ValueError("checkpoint must be a regular directory")
    if "state_representation.json" not in expected:
        raise ValueError("checkpoint inventory lacks representation marker")
    for rel in expected:
        path = Path(rel)
        if (not rel or "\\" in rel or path.is_absolute() or path.as_posix() != rel
                or any(part in ("", ".", "..") for part in rel.split("/"))):
            raise ValueError("invalid checkpoint inventory path")
    allowed_directories = {"."}
    for rel in expected:
        allowed_directories.update(p.as_posix() for p in Path(rel).parents)
    actual = {}
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            if _is_link_or_reparse(path):
                raise ValueError("checkpoint symlink/reparse point forbidden")
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = sha(path)
            elif path.is_dir():
                if path.relative_to(root).as_posix() not in allowed_directories:
                    raise ValueError("checkpoint tree has an unlisted directory")
                pending.append(path)
            else:
                raise ValueError("unsupported checkpoint tree entry")
    if actual != expected:
        raise ValueError("checkpoint tree inventory mismatch")
    return actual


class StateJsonNativeRuntime:
    """Never selected by legacy Runtime or by an implicit backend fallback."""

    def __init__(self, spec, *, expected_binding, expected_checkpoint_files, expected_sources):
        if type(spec) is not dict or set(spec) != _SPEC_KEYS:
            raise ValueError("explicit native development spec fields required")
        if (spec["backend"], spec["mode"], spec["purpose"]) != (
            "kev", "native", "development_evaluation"
        ):
            raise ValueError("state JSON consumer supports explicit development Kev native only")
        if (type(spec["checkpoint"]) is not str or not spec["checkpoint"]
                or spec["device"] not in ("cpu", "cuda")
                or spec["weights_dtype"] != "fp32" or spec["amp"] not in ("none", "bf16")
                or (spec["amp"] == "bf16" and spec["device"] != "cuda")):
            raise ValueError("unsupported native checkpoint/device/precision spec")

        # All frozen binding, source, and complete tree checks precede load_backend.
        validate_binding(expected_binding)
        sources = _checked_sources(expected_sources)
        checkpoint_files = _checked_checkpoint(spec["checkpoint"], expected_checkpoint_files)
        marker = read_checkpoint_marker(spec["checkpoint"], expected_binding=expected_binding)

        import torch
        from .backends import load_backend
        from .train import validate_amp

        validate_amp(spec["device"], spec["amp"])
        native = load_backend("kev", spec["checkpoint"], spec["device"], spec["weights_dtype"])
        native.model.eval()
        self.spec = dict(spec)
        self.native = native
        self.expected_binding = copy.deepcopy(expected_binding)
        self.identity = {
            "representation": {"binding": copy.deepcopy(self.expected_binding), "marker": marker},
            "consumer_source_sha256": sources["decision_ft.state_json_native"],
            "checkpoint_files_sha256": checkpoint_files,
            "sources_sha256": sources,
            "spec": {k: v for k, v in spec.items() if k != "checkpoint"},
            "environment": versions(),
        }
        self.fingerprint = digest(self.identity)

    def logits(self, request):
        # The cache is scoped to exactly one forward, including preprocessing errors.
        try:
            pairs = prepare_inference_atoms(request, expected_binding=self.expected_binding)
            import torch
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.spec["amp"] == "bf16"
                   else contextlib.nullcontext())
            with torch.inference_mode(), ctx:
                outputs = self.native.forward([atom for _, atom in pairs])
            if len(outputs) != len(pairs):
                raise RuntimeError("native result count does not match the request")
            result = []
            for (qid, atom), output in zip(pairs, outputs):
                logits = np.asarray(output.detach().float().cpu().numpy())
                if logits.ndim != 1 or len(logits) != len(atom.keys):
                    raise RuntimeError("logit shape does not match the candidate set")
                if not np.isfinite(logits).all():
                    raise RuntimeError("nonfinite model output")
                result.append((qid, atom, logits))
            return result
        finally:
            self.native.cache.clear()

    def predict(self, request, calibration=None):
        return Predictor(self, calibration).predict(request)
