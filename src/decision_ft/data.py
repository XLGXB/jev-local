"""Strict, label-safe request conversion and leakage-aware dataset partitioning."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KINDS = {"choice": 0, "score": 1, "noul": 2}
PARTITIONS = ("train", "dev", "calibration", "test")
VERIFIED = {"human_verified", "program_verified", "observed_outcome"}


def json_bytes(value: Any, *, sort_keys: bool = False) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=sort_keys, separators=(",", ":")).encode("utf-8")


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, values: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(json_bytes(value).decode("utf-8") + "\n")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_pairs)
                validate_record(row)
                rows.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    if len({r['_meta']['id'] for r in rows}) != len(rows):
        raise ValueError("duplicate record id")
    return rows


def option_keys(q: dict) -> list[str]:
    kind = q.get("type")
    crit = q.get("criteria")
    if kind == "choice":
        if not isinstance(crit, dict) or not 2 <= len(crit) <= 255:
            raise ValueError("choice requires 2..255 explicitly described options")
        if any(not isinstance(k, str) or not k for k in crit):
            raise ValueError("choice keys must be nonempty strings")
        return list(crit)
    if kind == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 255:
            raise ValueError("score requires 2..255 ordered rubric levels")
        return [str(i) for i in range(len(crit))]
    if kind == "noul":
        if crit is not None and (not isinstance(crit, dict) or set(crit) != {"false", "true"}):
            raise ValueError("noul criteria must explicitly contain false and true")
        return ["false", "true"]
    raise ValueError("type must be choice, score or noul")


def target_vector(q: dict) -> tuple[list[str], list[float], int]:
    keys = option_keys(q)
    if "label" not in q:
        raise ValueError("missing verified label; unknown is not false")
    label = q["label"]
    if q["type"] == "choice":
        if not isinstance(label, str) or label not in keys:
            raise ValueError("choice label must be a present option key")
        index = keys.index(label)
    elif q["type"] == "noul":
        if type(label) is not bool:
            raise ValueError("noul label must be JSON true/false, not a string or integer")
        index = int(label)
    else:
        if type(label) is not int or not 0 <= label < len(keys):
            raise ValueError("score label must be an integer level index, starting at zero")
        index = label
    if "target" in q:
        target = q["target"]
        if not isinstance(target, dict) or set(target) != set(keys):
            raise ValueError("soft target must name every option exactly; use explicit zeros")
        if any(type(target[k]) not in (float, int) for k in keys):
            raise ValueError("target probabilities must be numeric, not booleans")
        values = [float(target[k]) for k in keys]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("target must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("target must sum to one; no silent normalization/uniform fallback")
    else:
        values = [float(i == index) for i in range(len(keys))]
    return keys, values, index


def public_question(q: dict) -> dict:
    return {k: copy.deepcopy(q[k]) for k in ("type", "instructions", "criteria") if k in q}


def public_request(row: dict) -> dict:
    return {"state": copy.deepcopy(row["state"]),
            "questions": {key: public_question(q) for key, q in row["questions"].items()}}


def validate_record(row: dict) -> None:
    if not isinstance(row, dict) or row.get("schema_version", 1) != 1:
        raise ValueError("unsupported request schema")
    if "state" not in row or not isinstance(row.get("questions"), dict) or not row["questions"]:
        raise ValueError("state and nonempty questions are required")
    meta = row.get("_meta", {})
    for key in ("id", "group_id", "source", "task", "label_source"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise ValueError(f"_meta.{key} must be a nonempty string")
    if meta.get("training_allowed") is not True:
        raise ValueError("local-training authorization is required")
    for q in row["questions"].values():
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError("question instructions must be a nonempty string")
        target_vector(q)
    json_bytes(row)  # rejects NaN/Infinity even in state/metadata


@dataclass
class Atom:
    uid: str
    group_id: str
    state: Any
    question: dict                     # strictly public fields only
    keys: list[str]
    target: list[float]
    label: int
    task: str
    label_source: str

    @property
    def kind(self) -> int:
        return KINDS[self.question["type"]]

    @property
    def fingerprint(self) -> str:
        # Preserve option order: a reference distribution is valid for this exact input.
        return hashlib.sha256(json_bytes({"state": self.state, "question": self.question})).hexdigest()


def atomize(rows: list[dict]) -> list[Atom]:
    atoms = []
    for row in rows:
        validate_record(row)
        m = row["_meta"]
        for qid, q in row["questions"].items():
            keys, target, label = target_vector(q)
            atoms.append(Atom(f"{m['id']}::{qid}", m["group_id"], copy.deepcopy(row["state"]),
                              public_question(q), keys, target, label, m["task"], m["label_source"]))
    return atoms


def permutation(row: dict, seed: int) -> dict:
    """Only choice may be shuffled. Keyed targets remain correctly aligned."""
    result = copy.deepcopy(row)
    rng = random.Random(seed)
    for q in result["questions"].values():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            rng.shuffle(keys)
            q["criteria"] = {k: q["criteria"][k] for k in keys}
    validate_record(result)
    return result


def _leakage_keys(row: dict) -> set[str]:
    m = row["_meta"]
    tokens = {f"group:{m['group_id']}"}
    # Equal trajectory, template, or source artifact joins rows even with different group_id.
    for field in ("trajectory_id", "template_id", "source_artifact_sha256"):
        if m.get(field):
            tokens.add(f"{field}:{m[field]}")
    for atom in atomize([row]):
        # Sort only for duplicate detection, never for tokenization/reference-cache alignment.
        key = hashlib.sha256(json_bytes({"state": atom.state, "question": atom.question},
                                       sort_keys=True)).hexdigest()
        tokens.add("content:" + key)
    return tokens


def partition_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for i, row in enumerate(rows):
        for token in _leakage_keys(row):
            if token in owners:
                parent[find(i)] = find(owners[token])
            else:
                owners[token] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    components = sorted(groups.values(), key=lambda rs: min(r['_meta']['id'] for r in rs))
    if len(components) < 4:
        raise ValueError("need at least four independent leakage components for four partitions")
    random.Random(seed).shuffle(components)
    hold = max(1, len(components) // 10)
    counts = [len(components) - 3 * hold, hold, hold, hold]
    out = {name: [] for name in PARTITIONS}
    start = 0
    for name, count in zip(PARTITIONS, counts):
        for component in components[start:start + count]:
            for row in component:
                item = copy.deepcopy(row)
                item['_meta']['split'] = name
                out[name].append(item)
        start += count
    return out


def leakage_profile(rows: list[dict]) -> dict:
    tokens = set().union(*(_leakage_keys(row) for row in rows))
    return {"tokens": sorted(hashlib.sha256(t.encode()).hexdigest() for t in tokens),
            "ids": sorted(hashlib.sha256(r['_meta']['id'].encode()).hexdigest() for r in rows)}


def verify_dataset(directory: str | Path, partitions=PARTITIONS) -> dict[str, list[dict]]:
    """Check byte hashes and frozen leakage certificates for all partitions.
    Parse labels only for requested partitions. Training requests train/dev only;
    calibration requests calibration only. Full parsing happens during preparation.
    """
    directory = Path(directory)
    requested = set(partitions)
    if not requested <= set(PARTITIONS):
        raise ValueError("unknown dataset partition")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result, owner, ids_owner = {}, {}, {}
    for name in PARTITIONS:
        path = directory / f"{name}.jsonl"
        expected = manifest["files"][name]
        if file_sha(path) != expected["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {name}")
        profile = expected["leakage_profile"]
        for token in profile["tokens"]:
            if token in owner and owner[token] != name:
                raise ValueError("cross-partition leakage certificate collision")
            owner[token] = name
        for uid in profile["ids"]:
            if uid in ids_owner:
                raise ValueError("duplicate record id certificate")
            ids_owner[uid] = name
        if name not in requested:
            continue
        rows = read_jsonl(path)
        if any(row['_meta'].get('split') != name for row in rows):
            raise ValueError("partition metadata mismatch")
        if leakage_profile(rows) != profile:
            raise ValueError("content no longer matches its leakage certificate")
        result[name] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        p.error("output directory must not exist")
    rows = read_jsonl(a.input)
    splits = partition_rows(rows, a.seed)
    out.mkdir(parents=True)
    manifest = {"schema_version": 1, "seed": a.seed, "input_sha256": file_sha(a.input), "files": {}}
    for name, records in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, records)
        manifest["files"][name] = {"sha256": file_sha(path), "records": len(records),
                                    "questions": len(atomize(records)), "leakage_profile": leakage_profile(records)}
    write_json(out / "manifest.json", manifest)
    verify_dataset(out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
