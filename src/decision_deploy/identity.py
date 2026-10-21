"""Hash the actual shared candidate semantics, not only the deployment package."""
from __future__ import annotations

import inspect

import decision_ft.data as decision_data

from .common import sha


def decision_source_hashes(*, native_backend: bool = False) -> dict[str, str]:
    sources = {"decision_ft_data": sha(inspect.getfile(decision_data))}
    if native_backend:
        import decision_ft.backends as backends
        sources["decision_ft_backends"] = sha(inspect.getfile(backends))
    return sources
