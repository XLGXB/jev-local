"""Offline CPU evidence-ranking evaluation; predictions are advice, not authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevlocal.integration import context_tools, qwen_context as qc

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_CARD = "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B"
ASSET_NAMES = (
    ".gitattributes", "1_LogitScore/config.json", "README.md", "chat_template.jinja",
    "config.json", "config_sentence_transformers.json", "generation_config.json",
    "merges.txt", "model.safetensors", "modules.json", "sentence_bert_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
         "TORCH_FORCE_WEIGHTS_ONLY_LOAD")
MAX_JSON = 4 * 1024 * 1024
CATEGORIES = ("direct", "paraphrase", "lexical_distractor")


def at_stage(stage, function, *args, **kwargs):
    """Annotate and re-raise; only the CLI chooses whether to show a traceback."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        if not hasattr(exc, "demo_stage"):
            exc.demo_stage = stage
        raise


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open("rb") as stream:
        return qc.strict_json(stream.read(MAX_JSON + 1), max_bytes=MAX_JSON)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("invalid public identifier")
    return value


def validate_cases(dataset):
    if (not isinstance(dataset, dict) or set(dataset) != {"schema_version", "dataset_id", "provenance", "cases"}
            or type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1):
        raise ValueError("invalid cases schema")
    _id(dataset["dataset_id"])
    provenance = dataset["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {"kind", "license", "created", "private_data_used"}
            or provenance["kind"] != "new_synthetic_ai_assisted_examples"
            or provenance["license"] != "Apache-2.0" or provenance["private_data_used"] is not False
            or not isinstance(provenance["created"], str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", provenance["created"])):
        raise ValueError("public synthetic provenance required")
    if not isinstance(dataset["cases"], list) or not 1 <= len(dataset["cases"]) <= 256:
        raise ValueError("nonempty bounded cases required")
    prepared, ids = [], set()
    for case in dataset["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "query", "candidates", "max_chars"}:
            raise ValueError("invalid case fields")
        cid = _id(case["id"])
        if cid in ids:
            raise ValueError("duplicate case ID")
        ids.add(cid)
        arguments = {key: case[key] for key in ("query", "candidates", "max_chars")}
        prepare_started = time.perf_counter()
        query, safe, primary, snapshot = qc.prepare(arguments)
        validation_and_lexical_seconds = time.perf_counter() - prepare_started
        if primary["status"] != "ok":
            raise ValueError("case output budget insufficient")
        if any(not c["source"].startswith("public-example:") for c in safe):
            raise ValueError("public-example sources required; do not provide local paths")
        prepared.append((case["id"], query, safe, primary, snapshot, validation_and_lexical_seconds))
    return prepared


def validate_labels(labels, dataset):
    if (not isinstance(labels, dict) or set(labels) != {"schema_version", "dataset_id", "labels"}
            or type(labels["schema_version"]) is not int or labels["schema_version"] != 1
            or labels["dataset_id"] != dataset["dataset_id"] or not isinstance(labels["labels"], list)):
        raise ValueError("invalid labels schema or dataset identity")
    candidates = {c["id"]: {x["id"] for x in c["candidates"]} for c in dataset["cases"]}
    result = {}
    for row in labels["labels"]:
        if (not isinstance(row, dict) or set(row) != {"id", "expected_id", "category", "rationale"}
                or not isinstance(row["id"], str) or row["id"] not in candidates or row["id"] in result
                or not isinstance(row["expected_id"], str) or row["expected_id"] not in candidates[row["id"]]
                or row["category"] not in CATEGORIES
                or not isinstance(row["rationale"], str) or not row["rationale"].strip()):
            raise ValueError("invalid, duplicate, or unrelated label")
        result[row["id"]] = row
    if set(result) != set(candidates):
        raise ValueError("labels must exactly cover cases")
    return result


def no_link(path):
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if part.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("linked model path forbidden")


def verify_assets(model_dir, manifest):
    root = Path(model_dir)
    if not root.is_absolute() or str(root).startswith(("\\\\", "//")):
        raise ValueError("absolute local model directory required")
    no_link(root)
    if not root.is_dir():
        raise ValueError("model directory absent")
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "model_id", "revision", "model_card", "files"}
            or manifest["schema_version"] != 1 or manifest["model_id"] != MODEL_ID
            or manifest["revision"] != REVISION or manifest["model_card"] != MODEL_CARD
            or not isinstance(manifest["files"], list)):
        raise ValueError("invalid fixed model manifest")
    current, seen = [], set()
    for asset in manifest["files"]:
        if (not isinstance(asset, dict) or set(asset) != {"path", "bytes", "sha256"}
                or not isinstance(asset["path"], str) or asset["path"] not in ASSET_NAMES
                or asset["path"] in seen or type(asset["bytes"]) is not int or asset["bytes"] < 0
                or not isinstance(asset["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])):
            raise ValueError("invalid fixed asset record")
        seen.add(asset["path"])
        target = root.joinpath(*asset["path"].split("/"))
        no_link(target)
        if not target.is_file() or root.resolve() not in target.resolve().parents:
            raise ValueError("model asset absent or escaped")
        actual = {"path": asset["path"], "bytes": target.stat().st_size, "sha256": file_sha(target)}
        if actual != asset:
            raise ValueError("model asset bytes or SHA-256 mismatch")
        current.append(actual)
    if seen != set(ASSET_NAMES) or not qc.CRITICAL <= seen:
        raise ValueError("fixed model asset set incomplete")
    current.sort(key=lambda row: row["path"])
    return {"model_id": MODEL_ID, "revision": REVISION, "model_card": MODEL_CARD,
            "files_verified": True, "files": current, "assets_sha256": qc.digest(current),
            "prompt_version": qc.PROMPT_VERSION, "calibrated": False}


def load_native(model_dir):
    # Set before lazy imports. Refuse a previously imported online HF runtime.
    for module, attr in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                         ("transformers.utils.hub", "_is_offline_mode")):
        loaded = sys.modules.get(module)
        if loaded is not None and hasattr(loaded, attr) and getattr(loaded, attr) is not True:
            raise ValueError("native dependencies already imported online")
    for flag in FLAGS:
        os.environ[flag] = "1"

    def denied(*_args, **_kwargs):
        raise OSError("offline public demo")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, padding_side="left")
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if len(no_ids) != 1 or len(yes_ids) != 1 or no_ids == yes_ids:
        raise ValueError("yes/no must be distinct single tokens")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
        attn_implementation="sdpa").to("cpu").eval()
    if any(p.device.type != "cpu" or (p.is_floating_point() and p.dtype != torch.float32)
           for p in model.parameters()):
        raise ValueError("CPU FP32 model invariant failed")
    return torch, tokenizer, model, no_ids[0], yes_ids[0]


def validate_logits(logits, count):
    if (not isinstance(logits, list) or len(logits) != count
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                   for pair in logits)):
        raise ValueError("native logit shape or finite check failed")


def infer_native(query, safe, native):
    started = time.perf_counter()
    torch, tokenizer, model, no_id, yes_id = native
    prompts = qc.prompts(query, safe)
    lengths = qc.check_prompt_lengths(prompts, tokenizer.encode)
    scores, forward_seconds = [], 0.0
    by_id = {c["id"]: c for c in safe}
    for offset in range(0, len(prompts), 4):
        batch = prompts[offset:offset + 4]
        inputs = tokenizer([row["text"] for row in batch], padding=True, truncation=False,
                           add_special_tokens=False, return_tensors="pt").to("cpu")
        before = time.perf_counter()
        with torch.inference_mode():
            result = model(**inputs, use_cache=False, logits_to_keep=1).logits
            if tuple(result.shape[:2]) != (len(batch), 1):
                raise ValueError("native model output shape invalid")
            pairs = result[:, -1, [no_id, yes_id]].float().tolist()
        forward_seconds += time.perf_counter() - before
        validate_logits(pairs, len(batch))
        for row, pair in zip(batch, pairs):
            qc.score_pair(*pair)  # shared finite scoring contract
            scores.append({"id": row["id"], "source_sha256": hashlib.sha256(
                by_id[row["id"]]["source"].encode("utf-8")).hexdigest(),
                "no_logit": pair[0], "yes_logit": pair[1]})
    usage = {"judgments": len(safe), "forwards": (len(safe) + 3) // 4,
             "returned": len(scores), "token_lengths": dict(zip((p["id"] for p in prompts), lengths))}
    return {"scores": scores, "usage": usage}, time.perf_counter() - started, forward_seconds


def validate_output(output, cases, labels, model_dir):
    output = Path(output)
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("output must be a new file in an existing directory")
    resolved = output.resolve()
    protected = {Path(p).resolve() for p in (cases, labels, __file__, HERE / "model-manifest.json")}
    model = Path(model_dir).resolve()
    if resolved in protected or resolved == model or model in resolved.parents:
        raise ValueError("output must not overwrite inputs or model assets")
    return output


def runtime_metadata():
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "safetensors", "numpy", "tokenizers", "huggingface-hub")}
    return {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine()}, "dependencies": versions,
            "device": "cpu", "weights_dtype": "float32", "attention": "sdpa", "threads": 4,
            "seed": 42, "batch_max": 4, "offline": True, "local_files_only": True,
            "trust_remote_code": False, "use_safetensors": True}


def add_metrics(rows, labels):
    for row in rows:
        label = labels[row["id"]]
        expected = label["expected_id"]
        actual = {}
        for method, ranking in (("native", row["recommendation"]["ranking"]),
                                ("lexical", row["primary"]["results"])):
            ordered = [item["id"] for item in ranking]
            rank = ordered.index(expected) + 1
            actual[method] = {"selected_id": ordered[0], "expected_rank": rank, "correct": rank == 1}
        row["evaluation"] = {"expected_id": expected, "category": label["category"], **actual}

    def summarize(subset, method):
        count = len(subset)
        return {"count": count,
                "top1_accuracy": sum(r["evaluation"][method]["correct"] for r in subset) / count if count else None,
                "mrr": sum(1.0 / r["evaluation"][method]["expected_rank"] for r in subset) / count if count else None,
                "errors": [r["id"] for r in subset if not r["evaluation"][method]["correct"]]}

    return {method: {**summarize(rows, method), "per_category": {
        category: summarize([r for r in rows if r["evaluation"]["category"] == category], method)
        for category in CATEGORIES}} for method in ("native", "lexical")}


def evaluate(cases_path, labels_path, model_dir, output_path):
    started_utc, started = utc_now(), time.perf_counter()
    output = at_stage("output_validation", validate_output, output_path, cases_path, labels_path, model_dir)
    cases_hash = at_stage("cases_validation", file_sha, cases_path)
    dataset = at_stage("cases_validation", read_json, cases_path)
    prepared = at_stage("cases_validation", validate_cases, dataset)
    if file_sha(cases_path) != cases_hash:
        raise ValueError("cases changed during validation")
    manifest = at_stage("model_assets", read_json, HERE / "model-manifest.json")
    identity = at_stage("model_assets", verify_assets, model_dir, manifest)
    load_started = time.perf_counter()
    native = at_stage("model_load", load_native, Path(model_dir))
    load_seconds = time.perf_counter() - load_started
    rows = []
    for index, (case_id, query, safe, primary, snapshot, lexical_seconds) in enumerate(prepared):
        invocation_id = secrets.token_hex(16)
        child, inference_seconds, forward_seconds = at_stage("native_inference", infer_native, query, safe, native)
        child.update(snapshot_sha256=snapshot, identity=identity, invocation_id=invocation_id)
        recommendation = at_stage("prediction_binding", qc.compose, snapshot, safe, identity, invocation_id, child)
        rows.append({"id": case_id, "snapshot_sha256": snapshot, "primary": primary,
                     "recommendation": recommendation, "timing": {
                         "validation_and_lexical_seconds": lexical_seconds,
                         "cold_start_load_seconds": load_seconds if index == 0 else 0.0,
                         "native_inference_seconds": inference_seconds,
                         "native_forward_seconds": forward_seconds}})
    # Labels are first opened only after every model prediction is complete.
    labels_hash = at_stage("labels_evaluation", file_sha, labels_path)
    labels = at_stage("labels_evaluation", validate_labels,
                      at_stage("labels_evaluation", read_json, labels_path), dataset)
    if file_sha(labels_path) != labels_hash:
        raise ValueError("labels changed during evaluation")
    metrics = at_stage("labels_evaluation", add_metrics, rows, labels)
    report = {"schema_version": 1, "status": "evaluated", "dataset": {"id": dataset["dataset_id"],
        "cases_sha256": cases_hash, "labels_sha256": labels_hash, "provenance": dataset["provenance"]},
        "sources": {"demos/evidence_ranking/run.py": file_sha(__file__),
                    "jevlocal/integration/qwen_context.py": file_sha(qc.__file__),
                    "jevlocal/integration/context_tools.py": file_sha(context_tools.__file__),
                    "model_manifest_sha256": file_sha(HERE / "model-manifest.json")},
        "model": identity, "runtime": at_stage("runtime_metadata", runtime_metadata), "started_utc": started_utc,
        "ended_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
        "load_seconds": load_seconds,
        "total_validation_and_lexical_seconds": sum(r["timing"]["validation_and_lexical_seconds"] for r in rows),
        "total_native_inference_seconds": sum(r["timing"]["native_inference_seconds"] for r in rows),
        "total_native_forward_seconds": sum(r["timing"]["native_forward_seconds"] for r in rows),
        "cases": rows, "metrics": metrics, "calibrated": False,
        "execution_authorized": False, "training_allowed": False, "primary_changed": False,
        "limitations": ["Fixed synthetic examples are not a general semantic-quality benchmark.",
                        "validation_and_lexical_seconds measures the complete qc.prepare call, not isolated lexical algorithm performance.",
                        "Uncalibrated relevance scores confer no authority or speedup claim."]}
    payload = at_stage("report_serialization", qc.encoded, report) + b"\n"
    # Exclusive creation is the final operation; prediction failures produce no report.
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            stream.write(payload)
    except Exception as exc:
        exc.demo_stage = "report_write"
        if created:
            output.unlink(missing_ok=True)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--labels", type=Path, default=HERE / "labels.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate public cases/labels and structural budgets; no model or semantic check")
    parser.add_argument("--debug", action="store_true",
                        help="Print a local failure traceback; it may contain machine paths and should not be published verbatim")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            dataset = at_stage("cases_validation", read_json, args.cases)
            prepared = at_stage("cases_validation", validate_cases, dataset)
            at_stage("labels_validation", validate_labels,
                     at_stage("labels_validation", read_json, args.labels), dataset)
            print(json.dumps({"status": "structure_valid", "cases": len(prepared),
                              "semantic_evaluation": False, "model_loaded": False}))
            return 0
        if args.model_dir is None or args.output is None:
            raise ValueError("model-dir and new output are required")
        report = evaluate(args.cases, args.labels, args.model_dir, args.output)
        print(json.dumps({"status": "evaluated", "cases": len(report["cases"]),
                          "metrics": report["metrics"], "calibrated": False}))
        return 0
    except Exception as exc:
        # Default diagnostics contain stable stages, never arbitrary exception text.
        reason = ("dependency_unavailable" if isinstance(exc, (ImportError, importlib.metadata.PackageNotFoundError))
                  else "stage_failed")
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "stage": getattr(exc, "demo_stage", "invocation"), "reason": reason,
                          "execution_authorized": False}), file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
