"""Explicit one-batch native forward/backward + save/reload parity probe.
Does not execute optimizer.step(), train on test data, or grant deployment credit.
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import torch
from .backends import load_backend
from .data import atomize, verify_dataset, write_json
from .metrics import tensor_batch
from .objectives import soft_ce
from .train import infer, forward_context, validate_amp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True, choices=('laya', 'kev', 'toy'))
    p.add_argument('--init', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--amp', choices=('none', 'bf16'), default='none')
    p.add_argument('--weights-dtype', choices=('fp32', 'bf16'), default='fp32')
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('refusing to overwrite integration receipt')
    validate_amp(a.device, a.amp)
    atoms = atomize(verify_dataset(a.data, ('train',))['train'])[:2]
    backend = load_backend(a.backend, a.init, a.device, a.weights_dtype)
    initial = infer(backend, atoms, 2, a.amp)
    # eval mode keeps dropout deterministic but allows gradient computation.
    with forward_context(a.device, a.amp):
        outputs = backend.forward(atoms)
    z, target, mask, _ = tensor_batch(atoms, outputs)
    loss = soft_ce(z, target, mask).mean()
    loss.backward()
    gradients = {name: float(param.grad.detach().float().norm())
                 for name, param in backend.model.named_parameters()
                 if param.requires_grad and param.grad is not None}
    if not gradients or not all(torch.isfinite(torch.tensor(v)) for v in gradients.values()) or max(gradients.values()) <= 0:
        raise RuntimeError('missing or invalid gradients')
    head_grad = any(v > 0 for n, v in gradients.items() if not n.startswith(('encoder.', 'lm.')))
    trunk_grad = any(v > 0 for n, v in gradients.items() if n.startswith(('encoder.', 'lm.')))
    if not head_grad or (a.backend != 'toy' and not trunk_grad):
        raise RuntimeError('both the native head and backbone/LoRA need nonzero gradients')
    source = backend.source_manifest()
    with tempfile.TemporaryDirectory(prefix='decision-ft-parity-') as tmp:
        saved = Path(tmp) / 'checkpoint'
        backend.export(saved)
        del loss, outputs, z, target, mask, backend
        gc.collect()
        if a.device == 'cuda':
            torch.cuda.empty_cache()
        reloaded = load_backend(a.backend, str(saved), a.device, a.weights_dtype)
        after = infer(reloaded, atoms, 2, a.amp)
        difference = max((x.softmax(-1) - y.softmax(-1)).abs().max().item() for x, y in zip(initial, after))
    tolerance = 5e-3 if 'bf16' in (a.amp, a.weights_dtype) else 1e-5
    if difference > tolerance:
        raise RuntimeError(f'save/reload probability mismatch: {difference} > {tolerance}')
    result = {'scope': 'single_batch_forward_backward_and_reload_only', 'backend': a.backend,
              'atomic_questions': len(atoms), 'head_gradient_nonzero': head_grad,
              'backbone_or_lora_gradient_nonzero': trunk_grad,
              'save_reload_max_probability_difference': difference, 'tolerance': tolerance,
              'upstream_source_sha256': source, 'optimizer_updates': 0,
              'not_a_quality_or_safety_acceptance': True}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
