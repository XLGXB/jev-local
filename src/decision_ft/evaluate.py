"""Evaluate raw and optionally calibrated logits; locked test needs explicit access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .backends import load_backend
from .data import atomize, verify_dataset, write_json, file_sha
from .metrics import report
from .train import infer, model_digest, validate_amp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("laya", "kev", "toy"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--partition", choices=("dev", "test"), default="dev")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--calibration")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--weights-dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--amp", choices=("none", "bf16"), default="none")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    if a.batch < 1:
        p.error("batch must be positive")
    if a.partition == "test" and not a.allow_test:
        p.error("locked test access needs --allow-test after candidate/threshold freeze")
    if Path(a.out).exists():
        p.error("refusing to overwrite an evaluation receipt")
    validate_amp(a.device, a.amp)
    atoms = atomize(verify_dataset(a.data, (a.partition,))[a.partition])
    backend = load_backend(a.backend, a.checkpoint, a.device, a.weights_dtype)
    digest = model_digest(a.checkpoint)
    logits = infer(backend, atoms, a.batch, a.amp)
    result = {"partition": a.partition, "checkpoint_sha256": digest,
              "dataset_manifest_sha256": file_sha(Path(a.data) / "manifest.json"),
              "raw": report(atoms, logits)}
    if a.calibration:
        calibration = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
        spec = {k: getattr(a, k) for k in ("backend", "device", "weights_dtype", "amp")}
        if calibration["checkpoint_sha256"] != digest or calibration["inference_spec"] != spec:
            raise ValueError("calibration belongs to another checkpoint or inference configuration")
        if calibration["upstream_source_sha256"] != backend.source_manifest():
            raise ValueError("native source changed after calibration")
        if calibration["dataset_manifest_sha256"] != result["dataset_manifest_sha256"]:
            raise ValueError("calibration dataset mismatch")
        for kind in {str(atom.kind) for atom in atoms}:
            if calibration["fits"][kind]["status"] == "not_present":
                raise ValueError(f"type {kind} has no fitted calibration")
        result["calibrated"] = report(atoms, logits, calibration["temperatures"])
        result["calibration_sha256"] = file_sha(a.calibration)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
