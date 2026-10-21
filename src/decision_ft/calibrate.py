"""Fit per-type temperatures on the independent calibration partition only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .backends import load_backend
from .data import atomize, verify_dataset, write_json, file_sha, VERIFIED
from .metrics import tensor_batch, report
from .objectives import soft_ce
from .train import infer, model_digest, validate_amp


def fit_temperature(atoms, logits, min_count=30):
    temperatures, details = {str(i): 1.0 for i in range(3)}, {}
    for kind in range(3):
        ix = [i for i, atom in enumerate(atoms) if atom.kind == kind]
        if not ix:
            details[str(kind)] = {"status": "not_present", "n": 0}
            continue
        if len(ix) < min_count:
            raise ValueError(f"type {kind}: {len(ix)} calibration items < required {min_count}")
        selected = [atoms[i] for i in ix]
        z, target, mask, _ = tensor_batch(selected, [logits[i].detach().cpu() for i in ix])
        # Bounded grid search is stable, reproducible, and avoids fragile scalar optimizer convergence.
        grid = torch.exp(torch.linspace(torch.log(torch.tensor(0.05)), torch.log(torch.tensor(20.0)), 241))
        losses = torch.stack([soft_ce(z / t, target, mask).mean() for t in grid])
        best = int(losses.argmin())
        temperature = float(grid[best])
        temperatures[str(kind)] = temperature
        details[str(kind)] = {"status": "bound_hit" if best in (0, len(grid)-1) else "fitted",
                              "n": len(ix), "temperature": temperature,
                              "raw_nll": soft_ce(z, target, mask).mean().item(),
                              "calibrated_nll": losses[best].item()}
    return temperatures, details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("laya", "kev", "toy"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--weights-dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--amp", choices=("none", "bf16"), default="none")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--min-per-type", type=int, default=30)
    a = p.parse_args()
    if min(a.batch, a.min_per_type) < 1:
        p.error("positive batch and calibration count required")
    if Path(a.out).exists():
        p.error("refusing to overwrite calibration sidecar")
    if Path(a.checkpoint).resolve() in Path(a.out).resolve().parents:
        p.error("store calibration outside the immutable checkpoint directory")
    validate_amp(a.device, a.amp)
    atoms = atomize(verify_dataset(a.data, ("calibration",))["calibration"])
    if any(atom.label_source not in VERIFIED for atom in atoms):
        raise ValueError("calibration requires verified labels, not unverified teacher self-confidence")
    backend = load_backend(a.backend, a.checkpoint, a.device, a.weights_dtype)
    logits = infer(backend, atoms, a.batch, a.amp)
    temperatures, details = fit_temperature(atoms, logits, a.min_per_type)
    output = {"schema_version": 1, "checkpoint_sha256": model_digest(a.checkpoint),
              "dataset_manifest_sha256": file_sha(Path(a.data) / "manifest.json"),
              "calibration_file_sha256": file_sha(Path(a.data) / "calibration.jsonl"),
              "inference_spec": {k: getattr(a, k) for k in ("backend", "device", "weights_dtype", "amp")},
              "upstream_source_sha256": backend.source_manifest(), "temperatures": temperatures,
              "fits": details, "min_per_type": a.min_per_type,
              "raw_calibration_partition_metrics": report(atoms, logits),
              "fitted_calibration_partition_metrics": report(atoms, logits, temperatures),
              "warning": "Fit-set metrics are not held-out performance; evaluate locked test separately. Sidecar is not automatically applied by native servers."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, output)
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
