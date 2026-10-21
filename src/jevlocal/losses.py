"""训练目标的各个分量（ModernBERT 路径用）。

配方来源与取舍
--------------
openJev-verdict-2.0 的原话是"目标混合了软交叉熵、一个 Brier 项、以及一个小权重的
硬标签项"，并由 ``--lambda_brier`` / ``--perm_kl`` 暴露出来。kev 侧的默认配方更朴素：
**正确答案上的交叉熵**，且发布的模型**没有**用 ``--perm_kl`` / ``--ord_w``。

所以这里的默认权重刻意偏保守（软项为主、硬项很小），并且把 permutation-KL 与序数项
做成**可选、从 0 扫起**——openJev 的 RUNBOOK §5 明确要求在 dev 上扫而不是照抄。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

EPS = 1e-9


# --------------------------------------------------------------------------- #
# 基础分量
# --------------------------------------------------------------------------- #
def soft_target_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """软标签交叉熵 ``-Σ t·log softmax(z)``。

    对软标签这**就是** KL 项（差一个与参数无关的教师熵常数），所以不用另外写 KL：
    最小化它等价于让学生的分布贴向教师分布。用 ``log_softmax`` 而不是先 softmax 再 log，
    后者在极端 logits 下会得到 ``log(0) = -inf``。
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def brier_term(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``mean Σ_i (p_i - t_i)²``。对**软目标**做，不是对 onehot。"""
    p = F.softmax(logits.float(), dim=-1)
    return ((p - target) ** 2).sum(dim=-1).mean()


def hard_ce(logits: torch.Tensor, hard_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), hard_idx.long(), reduction="mean")


def ordinal_term(logits: torch.Tensor, hard_idx: torch.Tensor) -> torch.Tensor:
    """序数辅助项：惩罚"期望层级"偏离真实层级。``Σ_i p_i·|i-y|`` 的均值。

    ⚠️ 定义说明：kev 只公布了 ``--ord_w`` 这个开关和它被启用/未启用，没有公开公式。
    这里定义成期望距离，是**我们自己的选择**，上报时必须写明，否则与别家不可比。

    另注：这一项**不是 proper scoring rule**（只约束一阶矩）。它只能当辅助项，
    不能替代 proper 奖励（见 ``local_decisions/labels.py`` 的 ``ordinal_mean_penalty``）。
    """
    k = logits.shape[-1]
    p = F.softmax(logits.float(), dim=-1)
    levels = torch.arange(k, device=logits.device, dtype=p.dtype)
    dist = (levels.unsqueeze(0) - hard_idx.unsqueeze(1).to(p.dtype)).abs()
    return (p * dist).sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# 顺序不变性
# --------------------------------------------------------------------------- #
def _sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    logp = p.clamp_min(EPS).log()
    logq = q.clamp_min(EPS).log()
    return 0.5 * ((p * (logp - logq)).sum(-1) + (q * (logq - logp)).sum(-1)).mean()


def symmetric_permutation_kl(
    logits_base: torch.Tensor,
    logits_perm: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """同一问题在两种选项顺序下分布的对称 KL。

    ``perm`` 是**拉取式**排列（``perm[i]`` = 新位置 i 上放原序的第几个），与
    ``jevlocal.permutation.apply_perm`` 同一约定。所以折回原序要**推入式**：
    ``folded[perm[i]] = permuted[i]``，即 ``folded.index_copy_(1, perm, permuted)``。

    这里若写反成 ``gather``，两个分布会各自被打乱一次，KL 依旧是个正数、训练照跑、
    loss 也会下降——**不会报任何错**，只是这一项不再惩罚顺序偏置。所以下面带一条
    单位排列自检。
    """
    p_base = F.softmax(logits_base.float(), dim=-1)
    p_perm = F.softmax(logits_perm.float(), dim=-1)
    if perm.shape[-1] != p_perm.shape[-1]:
        raise ValueError("perm 长度与分布宽度不符")
    folded = torch.empty_like(p_perm)
    folded.index_copy_(1, perm.to(p_perm.device), p_perm)
    return _sym_kl(p_base, folded)


def perm_kl_is_identity(logits: torch.Tensor, perm: torch.Tensor) -> bool:
    """喂单位排列时 KL 必须≈0（用于单测与启动自检）。"""
    return bool(symmetric_permutation_kl(logits, logits, perm).item() < 1e-6)


# --------------------------------------------------------------------------- #
# 总损失
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LossWeights:
    """默认值刻意保守：软项为主，硬项很小，两个可选项从 0 起扫。"""

    soft: float = 1.0
    brier: float = 0.0  # openJev 在 dev 上扫 {0.0, 0.25, 0.5, 1.0}
    hard: float = 0.1
    perm: float = 0.0  # openJev 默认 0.5，但要扫；kev 发布的模型未用
    ordinal: float = 0.0

    def names(self) -> Sequence[str]:
        return ("soft", "brier", "hard", "perm", "ordinal")


def compute_loss(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    hard_idx: torch.Tensor | None,
    weights: LossWeights,
    logits_perm: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    is_ordinal: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """返回 (总损失, 各分量数值)。分量都要记下来——只记总损失会让扫描失去依据。"""
    parts: dict[str, torch.Tensor] = {"soft": soft_target_ce(logits, target)}

    if weights.brier > 0:
        parts["brier"] = brier_term(logits, target)
    if weights.hard > 0 and hard_idx is not None:
        parts["hard"] = hard_ce(logits, hard_idx)
    if weights.perm > 0:
        if logits_perm is None or perm is None:
            raise ValueError("weights.perm > 0 时必须提供 logits_perm 与 perm")
        parts["perm"] = symmetric_permutation_kl(logits, logits_perm, perm)
    if weights.ordinal > 0 and is_ordinal:
        if hard_idx is None:
            raise ValueError("序数项需要 hard_idx")
        parts["ordinal"] = ordinal_term(logits, hard_idx)

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for name, value in parts.items():
        total = total + float(getattr(weights, name)) * value

    scalars = {name: float(v.detach().cpu()) for name, v in parts.items()}
    scalars["total"] = float(total.detach().cpu())
    return total, scalars
