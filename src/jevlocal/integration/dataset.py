"""Prepare already-split training records without random repartitioning.

The input hash covers ordered, normalized JSON records, not an input file.
All output is staged beside the destination and verified before publication.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from decision_ft.data import (
    PARTITIONS,
    VERIFIED,
    atomize,
    file_sha,
    json_bytes,
    leakage_profile,
    validate_record,
    verify_dataset,
    write_json,
    write_jsonl,
)


def prepare_presplit_dataset(records: list[dict], output_dir: str | Path) -> dict:
    """Write all four explicit partitions; fail rather than dropping or moving rows.

    Returns a ``verify_dataset``-compatible manifest. ``input_sha256`` hashes
    the ordered input records serialized by ``json_bytes``; it is *not* a hash
    of an original JSONL file. The destination must not already exist.
    """
    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")

    # Freeze caller-owned objects once. Neither validation nor publication may
    # alter labels, choice order, metadata, or the explicit split.
    rows = copy.deepcopy(records)
    partitions: dict[str, list[dict]] = {name: [] for name in PARTITIONS}
    ids: set[str] = set()
    for row in rows:
        validate_record(row)
        if any("target" not in question for question in row["questions"].values()):
            raise ValueError("presplit bridge records require explicit soft targets")
        meta = row["_meta"]
        split = meta.get("split")
        if not isinstance(split, str) or split not in partitions:
            raise ValueError(f"unknown or OOD split: {split!r}")
        if split in ("calibration", "test") and meta["label_source"] not in VERIFIED:
            raise ValueError(f"{split} requires a verified label_source")
        record_id = meta["id"]
        if record_id in ids:
            raise ValueError(f"duplicate record id: {record_id}")
        ids.add(record_id)
        partitions[split].append(row)
    if any(not part for part in partitions.values()):
        missing = [name for name, part in partitions.items() if not part]
        raise ValueError(f"all four partitions must be nonempty; missing {missing}")

    # json_bytes is strict about NaN/Infinity. This serialization is compact
    # and order-preserving, including choice key order and row order.
    input_sha256 = hashlib.sha256(json_bytes(rows)).hexdigest()
    manifest: dict = {
        "schema_version": 1,
        "partitioning": "presplit",
        "input_hash_kind": "ordered_records_json_v1",
        "input_sha256": input_sha256,
        "files": {},
    }

    parent = output.parent.resolve()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        for name in PARTITIONS:
            path = staging / f"{name}.jsonl"
            part = partitions[name]
            write_jsonl(path, part)
            manifest["files"][name] = {
                "sha256": file_sha(path),
                "records": len(part),
                "questions": len(atomize(part)),
                "leakage_profile": leakage_profile(part),
            }
        write_json(staging / "manifest.json", manifest)
        verified = verify_dataset(staging)
        if any(verified[name] != partitions[name] for name in PARTITIONS):
            raise ValueError("verified dataset differs from explicit input partitions")
        if os.path.lexists(output):
            raise FileExistsError(f"output directory appeared during preparation: {output}")
        staging.rename(output)
        return manifest
    finally:
        # Only the temporary directory created above may be removed. It is
        # always a direct child of the intended output parent.
        if staging.exists():
            if staging.resolve().parent != parent:
                raise RuntimeError("unsafe staging cleanup path")
            shutil.rmtree(staging)
