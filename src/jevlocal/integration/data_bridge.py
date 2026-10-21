"""Lossless bridge from canonical samples to label-safe decision_ft records.

The public state is passed through unchanged. Future-state leakage must be
prevented by an upstream state projection, not by generic token deletion here.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal
from typing import Any

from decision_ft.data import VERIFIED, validate_record
from jevlocal.schema import Label, Question, Sample

BRIDGE_VERSION = 1
_FORWARD_SPLITS = {"train": "train", "dev": "dev", "calib": "calibration", "test": "test"}
_REVERSE_SPLITS = {value: key for key, value in _FORWARD_SPLITS.items()}


class BridgeError(ValueError):
    """A sample or record cannot be represented by this bridge without loss."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeError(message)


def _source_gate(split: str, label_source: str) -> None:
    _require(isinstance(label_source, str) and bool(label_source.strip()),
             "label_source must be explicit and nonempty")
    if split in ("calibration", "test"):
        _require(label_source in VERIFIED,
                 f"{split} requires a verified label_source")


def _noul_style(criteria: Any, qid: str) -> str:
    if criteria is None:
        return "none"
    _require(isinstance(criteria, dict), f"{qid}: noul criteria must be a mapping")
    keys = set(criteria)
    if keys == {"yes", "no"}:
        return "yes_no"
    if keys == {"true", "false"}:
        return "true_false"
    raise BridgeError(f"{qid}: noul criteria keys must be yes/no or true/false")


def _noul_public(criteria: Any, style: str) -> Any:
    if style == "none":
        return None
    if style == "yes_no":
        return {"true": copy.deepcopy(criteria["yes"]),
                "false": copy.deepcopy(criteria["no"])}
    return copy.deepcopy(criteria)


def _noul_canonical(criteria: Any, style: str, qid: str) -> Any:
    _require(style in ("none", "yes_no", "true_false"),
             f"{qid}: invalid noul key style in sidecar")
    if criteria is None:
        _require(style == "none", f"{qid}: public criteria missing; cannot restore descriptions")
        return None
    _require(isinstance(criteria, dict) and set(criteria) == {"true", "false"},
             f"{qid}: invalid public noul criteria")
    if style == "yes_no":
        return {"yes": copy.deepcopy(criteria["true"]),
                "no": copy.deepcopy(criteria["false"])}
    return copy.deepcopy(criteria)


def to_training_record(
    sample: Sample, *, group_id: str, label_source: str, task: str,
    training_allowed: bool,
) -> dict:
    """Convert one complete canonical sample; never omit an unknown question."""
    _require(isinstance(sample, Sample), "sample must be a Sample")
    for name, value in (("group_id", group_id), ("task", task)):
        _require(isinstance(value, str) and bool(value.strip()),
                 f"{name} must be explicit and nonempty")
    _require(training_allowed is True, "explicit local-training authorization is required")
    try:
        sample.validate()
        _require(sample.split in _FORWARD_SPLITS, "OOD or unsupported split cannot enter training partitions")
        split = _FORWARD_SPLITS[sample.split]
        _source_gate(split, label_source)
        _require(isinstance(sample.source.get("ref"), str) and bool(sample.source["ref"].strip()),
                 "source.ref must be an explicit string")

        questions: dict[str, dict] = {}
        styles: dict[str, str] = {}
        teachers: dict[str, dict] = {}
        for q in sample.questions:
            lab = sample.labels[q.qid]
            _require(not lab.unanswerable and lab.hard is not None,
                     f"{q.qid}: unknown/unanswerable label cannot enter training")
            _require(q.qid not in questions, f"duplicate question ID: {q.qid}")
            entry: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if q.type == "noul":
                style = _noul_style(q.criteria, q.qid)
                styles[q.qid] = style
                criteria = _noul_public(q.criteria, style)
                if criteria is not None:
                    entry["criteria"] = criteria
                _require(type(lab.hard) is int and lab.hard in (0, 1),
                         f"{q.qid}: noul hard label must be 0 or 1")
                p = float(lab.soft)
                _require(math.isfinite(p) and 0.0 <= p <= 1.0,
                         f"{q.qid}: noul probability must be finite in [0,1]")
                entry["label"] = bool(lab.hard)
                entry["target"] = {"false": float(Decimal(1) - Decimal(str(p))), "true": p}
            elif q.type == "choice":
                _require(isinstance(q.criteria, dict) and len(q.criteria) >= 2,
                         f"{q.qid}: choice needs at least two candidates")
                entry["criteria"] = copy.deepcopy(q.criteria)
                entry["label"] = lab.hard
                entry["target"] = copy.deepcopy(lab.soft)
            elif q.type == "score":
                entry["criteria"] = copy.deepcopy(q.criteria)
                entry["label"] = lab.hard
                entry["target"] = {str(i): value for i, value in enumerate(lab.soft)}
            else:
                raise BridgeError(f"{q.qid}: unsupported question type")
            if lab.teacher is not None:
                _require(isinstance(lab.teacher, dict), f"{q.qid}: teacher must be a mapping")
                teachers[q.qid] = copy.deepcopy(lab.teacher)
            questions[q.qid] = entry

        row = {
            "schema_version": 1,
            "state": copy.deepcopy(sample.state),
            "questions": questions,
            "_meta": {
                "id": sample.sample_id,
                "group_id": group_id,
                "source": sample.source["ref"],
                "task": task,
                "label_source": label_source,
                "training_allowed": True,
                "split": split,
                "bridge_version": BRIDGE_VERSION,
                "canonical": {
                    "domain": sample.domain,
                    "source": copy.deepcopy(sample.source),
                    "provenance": copy.deepcopy(sample.provenance),
                    "noul_key_style": styles,
                    "teacher": teachers,
                },
            },
        }
        validate_record(row)
        return row
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError(str(exc)) from exc


def from_training_record(record: dict) -> Sample:
    """Rebuild labels and public questions from the *current* record content."""
    try:
        validate_record(record)
        meta = record["_meta"]
        _require(meta.get("bridge_version") == BRIDGE_VERSION,
                 "missing or unsupported bridge version")
        split = meta.get("split")
        _require(split in _REVERSE_SPLITS, "OOD or unsupported split cannot be reversed")
        _source_gate(split, meta["label_source"])
        sidecar = meta.get("canonical")
        _require(isinstance(sidecar, dict), "missing canonical sidecar")
        _require(isinstance(sidecar.get("domain"), str), "missing canonical domain")
        source = sidecar.get("source")
        provenance = sidecar.get("provenance")
        styles = sidecar.get("noul_key_style")
        teachers = sidecar.get("teacher")
        _require(isinstance(source, dict) and source.get("ref") == meta["source"],
                 "canonical source and public metadata disagree")
        _require(isinstance(provenance, dict) and isinstance(styles, dict)
                 and isinstance(teachers, dict), "malformed canonical sidecar")
        _require(set(styles) <= set(record["questions"]) and set(teachers) <= set(record["questions"]),
                 "stale question IDs in canonical sidecar")

        questions: list[Question] = []
        labels: dict[str, Label] = {}
        for qid, entry in record["questions"].items():
            _require(isinstance(qid, str) and bool(qid), "question ID must be nonempty string")
            kind = entry["type"]
            criteria = copy.deepcopy(entry.get("criteria"))
            target = entry.get("target")
            # A one-hot default would erase original soft targets; require explicit targets.
            _require(isinstance(target, dict), f"{qid}: bridge requires explicit soft target")
            teacher = copy.deepcopy(teachers.get(qid))
            if kind == "noul":
                _require(qid in styles, f"{qid}: missing noul key style")
                criteria = _noul_canonical(criteria, styles[qid], qid)
                hard = int(entry["label"])
                soft = float(target["true"])
            elif kind == "choice":
                _require(qid not in styles, f"{qid}: conflicting noul sidecar")
                hard = entry["label"]
                soft = {key: float(target[key]) for key in criteria}
            elif kind == "score":
                _require(qid not in styles, f"{qid}: conflicting noul sidecar")
                hard = entry["label"]
                soft = [float(target[str(i)]) for i in range(len(criteria))]
            else:
                raise BridgeError(f"{qid}: unsupported question type")
            questions.append(Question(qid, kind, entry["instructions"], criteria))
            labels[qid] = Label(hard, soft, teacher=teacher)
        restored = Sample(meta["id"], sidecar["domain"], copy.deepcopy(source),
                          copy.deepcopy(record["state"]), questions, labels,
                          _REVERSE_SPLITS[split], copy.deepcopy(provenance))
        restored.validate()
        return restored
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError(str(exc)) from exc
