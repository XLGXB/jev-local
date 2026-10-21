"""Single-device native-backend SFT / direct-score / RLCD-style training."""
from __future__ import annotations
import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
from .backends import load_backend
from .data import atomize, verify_dataset, file_sha, write_json, write_jsonl
from .metrics import tensor_batch, report
from .objectives import check, soft_ce, proper_score, reference_kl, rlcd_loss


def forward_context(device: str, amp: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" else contextlib.nullcontext()


def validate_amp(device: str, amp: str):
    if amp == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--amp bf16 needs supported CUDA hardware; use --amp none otherwise")


@torch.no_grad()
def infer(backend, atoms, batch_size: int, amp: str):
    backend.model.eval()
    result = []
    for start in range(0, len(atoms), batch_size):
        with forward_context(backend.device, amp):
            logits = backend.forward(atoms[start:start + batch_size])
        result.extend(z.detach().float().cpu() for z in logits)
    return result


def environment():
    names = ("torch", "transformers", "peft", "safetensors", "laya", "kev")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions,
            "scaffold_sha256": {p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")}}


def model_digest(path: str | Path):
    root = Path(path)
    # Sidecar calibration is deliberately outside the native checkpoint directory.
    files = {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    import hashlib
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("laya", "kev", "toy"))
    parser.add_argument("--stage", required=True, choices=("sft", "direct", "rlcd"))
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True, help="directory emitted by decision_ft.data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--weights-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--rl-weight", type=float, default=0.1)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--spherical-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-weight", type=float, default=1.0)
    parser.add_argument("--checkpointing", action="store_true", help="SFT only; RL uses eval mode with gradients")
    parser.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()
    if min(a.batch, a.accum, a.epochs) < 1 or min(a.lr, a.head_lr) <= 0:
        parser.error("positive batch/accum/epochs/learning rates are required")
    if a.group_size < 2 or a.sigma <= 0 or not all(math.isfinite(x) and x >= 0 for x in
        (a.weight_decay, a.rl_weight, a.ce_weight, a.kl_weight, a.spherical_weight, a.ordinal_weight)):
        parser.error("invalid objective configuration")
    validate_amp(a.device, a.amp)
    if a.checkpointing and a.stage != "sft":
        parser.error("this minimal RL path uses eval mode for a deterministic policy; checkpointing is SFT-only")
    out = Path(a.out)
    if out.exists():
        parser.error("refusing to overwrite an existing run")
    if a.stage != "sft":
        marker = Path(a.init) / "local_checkpoint.json"
        if not marker.exists() or json.loads(marker.read_text())["stage"] != "sft":
            parser.error("direct/RLCD must start from this scaffold's recorded SFT checkpoint")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rows = verify_dataset(a.data, ("train", "dev"))
    train_atoms, dev_atoms = atomize(rows["train"]), atomize(rows["dev"])
    backend = load_backend(a.backend, a.init, a.device, a.weights_dtype)
    # Preflight all admitted train/dev examples before any optimizer update.
    for atom in train_atoms + dev_atoms:
        backend.encode(atom)
    if a.checkpointing:
        trunk = getattr(backend.model, "encoder", getattr(backend.model, "lm", None))
        if trunk is None or not hasattr(trunk, "gradient_checkpointing_enable"):
            parser.error("backend has no supported gradient-checkpointing interface")
        trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out.mkdir(parents=True)
    manifest = {"status": "started", "args": vars(a), "environment": environment(),
                "upstream_source_sha256": backend.source_manifest(), "init_provenance": backend.provenance,
                "dataset_manifest_sha256": file_sha(Path(a.data) / "manifest.json"),
                "optimizer_resume": False, "raw_logit_training": True,
                "unverified_native_integration": a.backend != "toy"}
    write_json(out / "run_manifest.json", manifest)
    reference = {}
    if a.stage != "sft":
        initial = infer(backend, train_atoms, a.batch, a.amp)
        reference = {atom.uid: (atom.fingerprint, z) for atom, z in zip(train_atoms, initial)}
        write_jsonl(out / "reference_logits.jsonl", [
            {"id": atom.uid, "input_sha256": atom.fingerprint, "keys": atom.keys,
             "raw_logits": z.tolist()} for atom, z in zip(train_atoms, initial)])
        manifest["reference_sha256"] = file_sha(out / "reference_logits.jsonl")
    groups = backend.train_groups(a.lr, a.head_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    trainable = [p for p in backend.model.parameters() if p.requires_grad]
    manifest["trainable_parameters"] = sum(p.numel() for p in trainable)
    write_json(out / "run_manifest.json", manifest)
    history, best, step = [], float("inf"), 0
    start_time = time.monotonic()
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(train_atoms)))
        random.Random(a.seed + epoch).shuffle(order)
        # eval() disables dropout, NOT autograd. Reference and actor share inference semantics.
        backend.model.train(a.stage == "sft")
        aggregate, seen = {}, 0
        effective_batch = a.batch * a.accum
        for offset in range(0, len(order), effective_batch):
            window = order[offset:offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for sub in range(0, len(window), a.batch):
                atoms = [train_atoms[i] for i in window[sub:sub + a.batch]]
                with forward_context(a.device, a.amp):
                    outputs = backend.forward(atoms)
                logits, target, mask, kinds = tensor_batch(atoms, outputs)
                check(logits, target, mask)
                if a.stage == "sft":
                    loss = soft_ce(logits, target, mask).mean()
                    stats = {"ce": loss.detach().item()}
                else:
                    refs = []
                    for atom in atoms:
                        fingerprint, ref = reference[atom.uid]
                        if atom.fingerprint != fingerprint:
                            raise ValueError("stale SFT reference or changed option order")
                        refs.append(ref.to(logits.device))
                    ref_logits = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True)
                    if a.stage == "rlcd":
                        loss, stats = rlcd_loss(logits, target, mask, kinds,
                            group_size=a.group_size, sigma=a.sigma, rl_weight=a.rl_weight,
                            ce_weight=a.ce_weight, reference_logits=ref_logits, kl_weight=a.kl_weight,
                            spherical_weight=a.spherical_weight, ordinal_weight=a.ordinal_weight)
                    else:
                        # Required ablation: the same scoring rule, differentiated directly, is NOT RL.
                        ce = soft_ce(logits, target, mask).mean()
                        score = proper_score(logits, target, mask, kinds, a.spherical_weight, a.ordinal_weight).mean()
                        anchor = reference_kl(logits, ref_logits, mask).mean()
                        loss = a.ce_weight * ce - a.rl_weight * score + a.kl_weight * anchor
                        stats = {"ce": ce.detach().item(), "score": score.detach().item(), "reference_kl": anchor.detach().item()}
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss; no optimizer update performed")
                # Correct normalization for the FINAL partial accumulation window too.
                (loss * len(atoms) / len(window)).backward()
                for key, value in stats.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value * len(atoms)
                seen += len(atoms)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
        dev_logits = infer(backend, dev_atoms, a.batch, a.amp)
        metrics = report(dev_atoms, dev_logits)
        checkpoint = out / f"epoch-{epoch}"
        backend.export(checkpoint)
        write_json(checkpoint / "local_checkpoint.json", {"stage": a.stage, "backend": a.backend,
            "epoch": epoch, "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "weights_dtype": a.weights_dtype, "amp": a.amp, "training_seed": a.seed})
        result = {"epoch": epoch, "optimizer_steps": step,
                  "training": {k: v / seen for k, v in aggregate.items()}, "dev": metrics,
                  "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": model_digest(checkpoint)}
        history.append(result)
        if metrics["all"]["nll_target"] < best:
            best = metrics["all"]["nll_target"]
            write_json(out / "best_dev.json", result)
        write_json(out / "history.json", history)
        print(json.dumps({"epoch": epoch, "stage": a.stage, "dev": metrics["all"]}, ensure_ascii=False), flush=True)
    manifest.update(status="completed_training_not_deployment_approval", optimizer_steps=step,
                    wall_seconds=time.monotonic() - start_time)
    write_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
