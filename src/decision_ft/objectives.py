"""Original RLCD-style estimator, NOT a reproduction of TypeSafe's private trainer.
A Gaussian policy reports a probability vector via softmax. A score-function
estimator optimizes its proper-scoring reward. No generated text, PPO ratios,
real browser actions, or fabricated outcome rewards are used.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def check(logits: Tensor, target: Tensor, mask: Tensor) -> None:
    if logits.ndim != 2 or target.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("expected aligned [batch, options] tensors")
    if mask.dtype != torch.bool or not torch.all(mask.sum(-1) >= 2):
        raise ValueError("each row needs at least two valid options")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("non-finite logits/targets")
    if (target < 0).any() or (target.masked_select(~mask) != 0).any():
        raise ValueError("invalid or masked target probability")
    if not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6):
        raise ValueError("target probabilities must sum to one")


def log_probs(logits: Tensor, mask: Tensor) -> Tensor:
    # Finite sentinel avoids 0 * -inf in CE. Padded positions have exactly zero
    # contribution in all losses, and their gradients are masked out.
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)


def soft_ce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return -(target * log_probs(logits, mask)).sum(-1)


def ranked_error(prob: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    # Include only the K-1 nontrivial cumulative boundaries, even in padded batches.
    k = mask.sum(-1)
    boundaries = torch.arange(mask.shape[-1], device=mask.device) < (k - 1).unsqueeze(-1)
    return (((prob.cumsum(-1) - target.cumsum(-1)) ** 2) * boundaries).sum(-1) / (k - 1)


def proper_score(logits: Tensor, target: Tensor, mask: Tensor, kind: Tensor,
                 spherical_weight: float = 0.5, ordinal_weight: float = 1.0) -> Tensor:
    """Expected log + spherical score; subtract normalized RPS on ordinal tasks.
    Shapes may be [G,B,K] or [B,K]; masks/targets/kind broadcast from B.
    No reward clipping: clipping a log score can break strict propriety.
    """
    if spherical_weight < 0 or ordinal_weight < 0:
        raise ValueError("proper score weights must be nonnegative")
    lp = log_probs(logits, mask)
    p = lp.exp() * mask
    reward = (target * lp).sum(-1)
    spherical = (target * p).sum(-1) / torch.linalg.vector_norm(p, dim=-1).clamp_min(1e-12)
    return reward + spherical_weight * spherical - ordinal_weight * (kind == 1) * ranked_error(p, target, mask)


def reference_kl(logits: Tensor, reference_logits: Tensor, mask: Tensor) -> Tensor:
    log_ref = log_probs(reference_logits.detach(), mask)
    p_ref = log_ref.exp() * mask
    return (p_ref * (log_ref - log_probs(logits, mask))).sum(-1)


def rlcd_loss(logits: Tensor, target: Tensor, mask: Tensor, kind: Tensor,
              *, group_size: int = 8, sigma: float = 0.25,
              rl_weight: float = 0.1, ce_weight: float = 1.0,
              reference_logits: Tensor | None = None, kl_weight: float = 0.02,
              spherical_weight: float = 0.5, ordinal_weight: float = 1.0) -> tuple[Tensor, dict]:
    check(logits, target, mask)
    if group_size < 2 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("group_size >= 2 and positive finite sigma are required")
    if any(not math.isfinite(x) or x < 0 for x in (rl_weight, ce_weight, kl_weight)):
        raise ValueError("invalid loss coefficient")
    if reference_logits is None and kl_weight:
        raise ValueError("nonzero KL weight requires a frozen SFT reference")
    mu = logits.float().masked_fill(~mask, 0.0)
    # Independent Gaussian logits (not Laya's zero-mean projected noise).
    # Common-logit shifts do not affect probabilities; they add harmless variance.
    with torch.no_grad():
        noise = torch.randn((group_size,) + mu.shape, device=mu.device) * mask
        z = mu.detach().unsqueeze(0) + sigma * noise
        rewards = proper_score(z, target, mask, kind, spherical_weight, ordinal_weight)
        # Leave-one-out baseline is independent of sample g conditional on input.
        # Unlike self-inclusive group mean, it does not shrink the expected gradient.
        baseline = (rewards.sum(0, keepdim=True) - rewards) / (group_size - 1)
        advantage = rewards - baseline
    # Gaussian log likelihood, omitting parameter-independent constants.
    # z MUST be detached; otherwise z-mu cancels the score-function gradient.
    log_policy = -0.5 * ((((z.detach() - mu.unsqueeze(0)) / sigma) ** 2) * mask).sum(-1)
    policy = -(advantage.detach() * log_policy).mean()
    ce = soft_ce(mu, target, mask).mean()
    anchor = (reference_kl(mu, reference_logits, mask).mean()
              if reference_logits is not None else mu.sum() * 0.0)
    loss = ce_weight * ce + rl_weight * policy + kl_weight * anchor
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite RLCD loss")
    return loss, {"ce": ce.detach().item(), "policy_loss": policy.detach().item(),
                  "reference_kl": anchor.detach().item(), "reward": rewards.mean().item()}
