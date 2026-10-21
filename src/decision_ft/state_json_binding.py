"""Explicit source-bound state representation shared by training and inference."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import Atom, json_bytes, option_keys
from . import state_json


_MARKER_NAME = "state_representation.json"
_STATUS = "experimental_requires_explicit_consumer"
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def binding_manifest() -> dict[str, Any]:
    """Describe the current code; callers must pin this in a separate freeze."""
    return {
        "schema_version": 1,
        "policy": state_json.policy_manifest(),
        "binding_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], expected[key]) for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def validate_binding(expected_binding: dict[str, Any]) -> None:
    """Fail closed unless a supplied frozen binding matches this source exactly."""
    if not _strict_equal(expected_binding, binding_manifest()):
        raise ValueError("state representation binding mismatch")


def _binding_digest(expected_binding: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(expected_binding, sort_keys=True)).hexdigest()


@dataclass
class RepresentedAtom(Atom):
    representation_digest: str
    source_fingerprint: str

    @property
    def fingerprint(self) -> str:
        payload = {
            "state": self.state,
            "question": self.question,
            "representation_digest": self.representation_digest,
        }
        return hashlib.sha256(json_bytes(payload)).hexdigest()


def _represent_atom(atom: Atom, binding: dict[str, Any]) -> RepresentedAtom:
    if type(atom) is not Atom:
        raise TypeError("only exact original Atom input is accepted")
    if option_keys(atom.question) != atom.keys:
        raise ValueError("Atom candidate keys differ from public question order")
    public = {"state": atom.state, "questions": {"question": atom.question}}
    prepared = state_json.prepare_public_request(public, expected_policy=binding["policy"])
    return RepresentedAtom(
        uid=atom.uid,
        group_id=atom.group_id,
        state=prepared["state"],
        question=prepared["questions"]["question"],
        keys=copy.deepcopy(atom.keys),
        target=copy.deepcopy(atom.target),
        label=atom.label,
        task=atom.task,
        label_source=atom.label_source,
        representation_digest=_binding_digest(binding),
        source_fingerprint=atom.fingerprint,
    )


def prepare_training_atoms(
    atoms: list[Atom], *, expected_binding: dict[str, Any]
) -> list[RepresentedAtom]:
    validate_binding(expected_binding)
    return [_represent_atom(atom, expected_binding) for atom in atoms]


def prepare_inference_atoms(
    request: dict[str, Any], *, expected_binding: dict[str, Any]
) -> list[tuple[str, RepresentedAtom]]:
    validate_binding(expected_binding)
    from decision_deploy.common import atoms_for

    return [(qid, _represent_atom(atom, expected_binding)) for qid, atom in atoms_for(request)]


def _validate_freeze_and_epoch(training_freeze_sha256: Any, epoch: Any) -> None:
    if type(training_freeze_sha256) is not str or not _LOWER_SHA256.fullmatch(training_freeze_sha256):
        raise ValueError("training freeze must be lowercase SHA256")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be a positive exact integer")


def write_checkpoint_marker(
    directory: str | Path, *, expected_binding: dict[str, Any],
    training_freeze_sha256: str, epoch: int,
) -> Path:
    validate_binding(expected_binding)
    _validate_freeze_and_epoch(training_freeze_sha256, epoch)
    marker = {
        "schema_version": 1,
        "status": _STATUS,
        "binding": copy.deepcopy(expected_binding),
        "training_freeze_sha256": training_freeze_sha256,
        "epoch": epoch,
    }
    path = Path(directory) / _MARKER_NAME
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(marker, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return path


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint marker key")
        result[key] = value
    return result


def read_checkpoint_marker(
    directory: str | Path, *, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    validate_binding(expected_binding)
    path = Path(directory) / _MARKER_NAME
    marker = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if type(marker) is not dict or set(marker) != {
        "schema_version", "status", "binding", "training_freeze_sha256", "epoch"
    }:
        raise ValueError("invalid checkpoint marker fields")
    if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
        raise ValueError("invalid checkpoint marker schema")
    if type(marker["status"]) is not str or marker["status"] != _STATUS:
        raise ValueError("invalid checkpoint marker status")
    if not _strict_equal(marker["binding"], expected_binding):
        raise ValueError("checkpoint marker binding mismatch")
    _validate_freeze_and_epoch(marker["training_freeze_sha256"], marker["epoch"])
    return marker
