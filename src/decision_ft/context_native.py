"""Explicit, source-bound native Kev consumer for the context experiment."""

from __future__ import annotations

import copy
import importlib.util
import re
import sys
from pathlib import Path

from decision_deploy.common import digest, sha, versions
from .state_json_binding import read_checkpoint_marker, validate_binding
from .state_json_native import (
    StateJsonNativeRuntime, _checked_checkpoint, _checked_sources, _digest_map,
)


CONTEXT_SOURCE_MODULES = frozenset({
    "decision_ft.context_backend", "decision_ft.context_native",
})
CHECKPOINT_FILES = frozenset({
    "adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
    "head.pt", "local_checkpoint.json", "README.md",
    "state_representation.json", "context_policy.json", "tokenizer.json",
    "tokenizer_config.json",
})
_SPEC_KEYS = {
    "backend", "mode", "purpose", "checkpoint", "device", "weights_dtype",
    "amp", "context_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _checked_context_sources(expected):
    _digest_map(expected, keys=CONTEXT_SOURCE_MODULES)
    actual = {}
    for name in CONTEXT_SOURCE_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"required context source unavailable: {name}")
        path = Path(spec.origin)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required context source is not a regular file: {name}")
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is None or Path(origin).resolve() != path.resolve():
                raise ValueError(f"loaded context source origin differs: {name}")
        actual[name] = sha(path)
    if actual != expected:
        raise ValueError("context consumer source hash mismatch")
    return actual


class ContextStateJsonNativeRuntime(StateJsonNativeRuntime):
    """Use inherited logits/predict only; never construct a legacy runtime."""

    def __init__(self, spec, *, expected_binding, expected_checkpoint_files,
                 expected_sources, context, expected_context_sources):
        if type(spec) is not dict or set(spec) != _SPEC_KEYS:
            raise ValueError("explicit context native development spec fields required")
        if any(type(spec[key]) is not str for key in _SPEC_KEYS):
            raise ValueError("context native spec values must be exact strings")
        if (spec["backend"], spec["mode"], spec["purpose"]) != (
            "kev", "native", "development_evaluation"
        ):
            raise ValueError("context consumer supports explicit development Kev native only")
        if (not spec["checkpoint"]
                or spec["device"] not in ("cpu", "cuda")
                or spec["weights_dtype"] != "fp32" or spec["amp"] != "none"
                or not _SHA256.fullmatch(spec["context_sha256"])):
            raise ValueError("unsupported context checkpoint/device/precision spec")

        # Every source, marker and complete tree check precedes load_backend.
        from .context_backend import (
            ContextKevBackend, context_digest, read_context_marker, validate_context,
        )

        checked_context = validate_context(context)
        if spec["context_sha256"] != context_digest(checked_context):
            raise ValueError("context spec digest mismatch")
        validate_binding(expected_binding)
        sources = _checked_sources(expected_sources)
        context_sources = _checked_context_sources(expected_context_sources)
        _digest_map(expected_checkpoint_files, keys=CHECKPOINT_FILES)
        checkpoint_files = _checked_checkpoint(spec["checkpoint"], expected_checkpoint_files)
        marker = read_checkpoint_marker(
            spec["checkpoint"], expected_binding=expected_binding,
        )
        context_marker = read_context_marker(
            spec["checkpoint"], expected_context=checked_context,
            expected_backend_sha256=context_sources["decision_ft.context_backend"],
            expected_encoder_sources={name: sources[name] for name in (
                "kev.model", "kev.data", "decision_ft.backends",
            )},
        )

        import torch  # noqa: F401 - import deferred until all frozen checks pass
        from .backends import load_backend
        from .train import validate_amp

        validate_amp(spec["device"], spec["amp"])
        native = load_backend("kev", spec["checkpoint"], spec["device"],
                              spec["weights_dtype"])
        contextual = ContextKevBackend(native, checked_context,
                                       expected_encoder_sources={name: sources[name] for name in (
                                           "kev.model", "kev.data", "decision_ft.backends",
                                       )})
        contextual.model.eval()
        self.spec = dict(spec)
        self.native = contextual
        self.expected_binding = copy.deepcopy(expected_binding)
        self.identity = {
            "representation": {"binding": copy.deepcopy(self.expected_binding),
                               "marker": marker},
            "context": {"policy": copy.deepcopy(checked_context),
                        "marker": copy.deepcopy(context_marker)},
            "consumer_source_sha256": context_sources["decision_ft.context_native"],
            "context_sources_sha256": context_sources,
            "checkpoint_files_sha256": checkpoint_files,
            "sources_sha256": sources,
            "spec": {k: v for k, v in spec.items() if k != "checkpoint"},
            "environment": versions(),
        }
        self.fingerprint = digest(self.identity)
