"""RLCD：以**严格 proper scoring rule** 为奖励的 GRPO 式策略梯度。

Laya 对 RLCD 的描述是"proper-scoring-rule rewards, GRPO-style policy gradient"。
关键不在于 PPO/GRPO 的具体形式，而在于**奖励函数的性质**：

- proper scoring rule 的期望在"报出真实条件分布"处取最大 ⇒ 优化它就等于优化校准，
  而不是优化自信程度。
- 用 argmax 命中率当奖励会奖励过度自信：把 p 推到 0/1 能提命中率却破坏校准。
- 用教师 imitation 出来的伪置信度会静默毁掉下游所有阈值
  （hermes-jev-skills 记过：一条每轮被询问、答案却永远无法改变任何行为的轴是纯成本）。

本模块给出**闭式**适当性验证（不靠采样，不靠经验），可在单测里跑。
"""

from __future__ import annotations

import math
import random
from typing import Literal, Sequence

import torch

from .labels import EPS, proper_score
from .permutation import permutations_for

ScoreKind = Literal["log", "brier"]
PROPER_KINDS: tuple[str, ...] = ("log", "brier")


# --------------------------------------------------------------------------- #
# 适当性：闭式验证
# --------------------------------------------------------------------------- #
def closed_form_expected_score(kind: ScoreKind, p: Sequence[float], q: Sequence[float]) -> float:
    """在真实分布为 q 时，报出分布 p 的**期望**得分（闭式，无需采样）。

    - ``log``  ：``E_q[log p_y] = -H(q, p)``，由 Gibbs 不等式在 ``p == q`` 处取最大。
    - ``brier``：``E_q[1 - Σ(p_i - y_i)²] = 2·Σ p_i q_i - Σ p_i²``，且
      ``Σq² - (2Σpq - Σp²) = Σ(p - q)² ≥ 0``，故也在 ``p == q`` 处取最大。
    """
    if len(p) != len(q):
        raise ValueError("长度不一致")
    if kind == "log":
        return math.fsum(qi * math.log(max(float(pi), EPS)) for pi, qi in zip(p, q))
    if kind == "brier":
        return 2.0 * math.fsum(pi * qi for pi, qi in zip(p, q)) - math.fsum(float(pi) ** 2 for pi in p)
    raise ValueError(f"未知评分规则: {kind!r}。RLCD 只接受 proper scoring rule")


def verify_properness(
    kind: ScoreKind,
    k: int,
    *,
    trials: int = 200,
    seed: int = 20260922,
    tol: float = 1e-9,
) -> dict[str, float]:
    """数值复核：随机取 (p, q)，断言 ``E_q[score(q)] >= E_q[score(p)]``。

    这是把"奖励必须 proper"写成可执行断言，而不是写成一句注释。返回最坏一次的超额。
    """
    if kind not in PROPER_KINDS:
        raise ValueError(f"{kind!r} 不是 proper scoring rule，不许用于 RLCD")
    rng = random.Random(seed)
    worst = float("inf")
    for _ in range(trials):
        q = [rng.random() + 1e-3 for _ in range(k)]
        s = math.fsum(q)
        q = [x / s for x in q]
        p = [rng.random() + 1e-3 for _ in range(k)]
        s = math.fsum(p)
        p = [x / s for x in p]

        opt = closed_form_expected_score(kind, q, q)
        alt = closed_form_expected_score(kind, p, q)
        worst = min(worst, opt - alt)
    if worst < -tol:
        raise AssertionError(f"{kind} 未通过适当性复核：超额 {worst:.3e}")
    return {"kind_min_advantage": worst, "trials": float(trials)}


# --------------------------------------------------------------------------- #
# 组内采样与优势
# --------------------------------------------------------------------------- #
def build_group(k: int, *, group_size: int, seed: int) -> list[list[int]]:
    """一组 prompt 变体：第一个恒为单位排列。

    用**选项顺序**当分组维度，而不是重采样同一顺序：这样同一份 GRPO 分组同时充当
    permutation-KL 的配对，一组数据办两件事；而且顺序扰动本身是对真实世界不确定性的
    合理代理（下游候选顺序本来就可能变）。
    """
    return permutations_for(k, group_size=group_size, seed=seed)


def grpo_advantage(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """组内标准化优势 ``(R - mean) / std``。

    用 ``unbiased=False``：组大小通常只有 4–8，无偏标准差会把小样本的优势放大成噪声。
    组标准差为 0（整组同分）时直接返回全 0——此时"没有任何信号"，不该硬造梯度。
    """
    if rewards.ndim != 1:
        raise ValueError("rewards 应为一维（组内各样本）")
    std = rewards.std(unbiased=False)
    if float(std) < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def rewards_from_distributions(
    kind: ScoreKind,
    dists: Sequence[Sequence[float]],
    hard_idx: int,
) -> torch.Tensor:
    """把每个顺序下的分布折成标量奖励（在 Python 里算，避免梯度穿过奖励）。

    ⚠️ 奖励必须 ``detach``：让奖励可导会把策略梯度变成"直接对奖励求导"，
    梯度含义完全变了，而且不会报错。
    """
    values = [proper_score(kind, d, hard_idx) for d in dists]
    return torch.tensor(values, dtype=torch.float32, requires_grad=False)


# --------------------------------------------------------------------------- #
# 策略梯度
# --------------------------------------------------------------------------- #
def rlcd_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    ref_log_probs: torch.Tensor | None = None,
    beta_kl: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO 式策略梯度 + 到 SFT 参照模型的 KL 锚。

    ``log_probs`` / ``ref_log_probs`` / ``rewards`` 形状均为 ``(G,)``（G = 组大小）。

    KL 锚不可省：RLCD 只拿一个标量奖励做信号，若不加锚，策略会缓慢漂到"分布更极端但
    奖励略高"的解上——准确率看着没掉，校准全坏。锚就是 SFT 结束时的快照（冻结）。
    """
    if log_probs.ndim != 1 or rewards.ndim != 1:
        raise ValueError("log_probs 与 rewards 都应为一维")
    if log_probs.shape != rewards.shape:
        raise ValueError("形状不一致")
    if torch.is_tensor(rewards) and rewards.requires_grad:
        raise ValueError("rewards 必须 detach，否则梯度含义会变")

    adv = grpo_advantage(rewards)
    pg = -(adv * log_probs).mean()

    parts = {"policy_gradient": float(pg.detach().cpu())}
    total = pg
    if beta_kl > 0:
        if ref_log_probs is None:
            raise ValueError("beta_kl > 0 时必须提供 ref_log_probs（冻结的 SFT 快照）")
        kl = (log_probs - ref_log_probs.detach()).mean()
        total = total + float(beta_kl) * kl
        parts["kl_to_ref"] = float(kl.detach().cpu())
    parts["mean_reward"] = float(rewards.mean().detach().cpu())
    parts["std_reward"] = float(rewards.std(unbiased=False).detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts


def rlcd_readiness_check(temperature_logits_by_type: dict[str, int]) -> list[str]:
    """开始 RLCD 前的前置检查，返回应被修复的问题清单（空 = 可开工）。

    这三条都来自别处的实测教训：
    1. 奖励必须是 proper scoring rule —— 否则优化目标本身就错了；
    2. 每个问题类型都要有足够样本 —— 否则类型路由的温度拟合是在拟合噪声；
    3. 必须有 calib 切分 —— openJev 的双通道校准全部依赖 out-of-fold 数据，
       没有独立校准集就没法报置信度，而"只报分布 ECE 不报置信 ECE"会被读成挑指标。
    """
    problems: list[str] = []
    for kind in PROPER_KINDS:
        try:
            verify_properness(kind, 4, trials=8)
        except AssertionError as exc:
            problems.append(f"评分规则 {kind} 未通过适当性复核: {exc}")
    for qtype, n in temperature_logits_by_type.items():
        if n < 50:
            problems.append(f"类型 {qtype} 样本仅 {n} 条，不足以拟合路由温度（阈值 50）")
    return problems
