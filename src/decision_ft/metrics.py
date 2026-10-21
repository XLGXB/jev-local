"""Raw/calibrated metrics; empirical calibration is reported only on verified labels."""
from __future__ import annotations
from collections import defaultdict
import torch
from .data import Atom, VERIFIED
from .objectives import log_probs, soft_ce, ranked_error


def tensor_batch(atoms: list[Atom], logits: list[torch.Tensor]):
    if len(atoms) != len(logits) or not atoms:
        raise ValueError("empty or misaligned batch")
    if any(z.ndim != 1 or z.numel() != len(a.keys) for a, z in zip(atoms, logits)):
        raise ValueError("model output/candidate count mismatch")
    z = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0).float()
    mask = torch.arange(z.shape[-1], device=z.device)[None] < torch.tensor([len(a.keys) for a in atoms], device=z.device)[:, None]
    target = torch.zeros_like(z)
    for i, a in enumerate(atoms):
        target[i, :len(a.target)] = torch.tensor(a.target, device=z.device)
    kinds = torch.tensor([a.kind for a in atoms], dtype=torch.long, device=z.device)
    return z, target, mask, kinds


def report(atoms: list[Atom], logits: list[torch.Tensor], temperatures: dict | None = None) -> dict:
    z, target, mask, kinds = tensor_batch(atoms, [x.detach().cpu() for x in logits])
    temps = temperatures or {str(i): 1.0 for i in range(3)}
    t = torch.tensor([float(temps[str(a.kind)]) for a in atoms])[:, None]
    if (t <= 0).any() or not torch.isfinite(t).all():
        raise ValueError("invalid temperatures")
    z = z / t
    p = log_probs(z, mask).exp() * mask
    pred = p.argmax(-1)
    labels = torch.tensor([a.label for a in atoms])
    confidence = p.max(-1).values
    correct = pred == labels
    verified = torch.tensor([a.label_source in VERIFIED for a in atoms])
    nll = soft_ce(z, target, mask)
    brier = ((p - target) ** 2).sum(-1)
    rps = ranked_error(p, target, mask)
    groups = {"all": list(range(len(atoms)))}
    for i, atom in enumerate(atoms):
        groups.setdefault("type:" + atom.question["type"], []).append(i)
        groups.setdefault("task:" + atom.task, []).append(i)
    result = {}
    for name, indices in groups.items():
        ix = torch.tensor(indices)
        empirical = ix[verified[ix]]
        entry = {"n": len(indices), "n_verified": int(len(empirical)),
                 "accuracy_against_supplied_label": correct[ix].float().mean().item(),
                 "nll_target": nll[ix].mean().item(), "brier_target": brier[ix].mean().item()}
        score_ix = ix[kinds[ix] == 1]
        if len(score_ix):
            levels = torch.arange(p.shape[-1])
            pred_mean = (p[score_ix] * levels).sum(-1)
            target_mean = (target[score_ix] * levels).sum(-1)
            entry["rps_target"] = rps[score_ix].mean().item()
            entry["score_mean_mae_target"] = (pred_mean - target_mean).abs().mean().item()
        if len(empirical):
            cf, cr = confidence[empirical], correct[empirical].float()
            ece = 0.0
            for b in range(15):
                inside = (cf >= b / 15) & ((cf < (b + 1) / 15) if b < 14 else (cf <= 1))
                if inside.any():
                    ece += inside.float().mean().item() * abs(cf[inside].mean().item() - cr[inside].mean().item())
            entry["ece_verified_15_bins"] = ece
            entry["risk_coverage_verified"] = {}
            for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
                accepted = cf >= threshold
                entry["risk_coverage_verified"][str(threshold)] = {
                    "accepted": int(accepted.sum()), "coverage": accepted.float().mean().item(),
                    "error_rate": (1 - cr[accepted].mean().item()) if accepted.any() else None}
        result[name] = entry
    return result
